# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Adopted from
# https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/videollava/train/llava_trainer.py
# Its original license is Apache-2.0 License.


"""Reward Trainer for training the model with the reward signal."""

from __future__ import annotations

import os
from typing import Any, Generator, Iterator

import bitsandbytes
import torch
import torch.nn.functional as F
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from torch import nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from safe_sora.models import ScoreModelOutput
from safe_sora.utils import get_all_reduce_mean


def maybe_zero_3(
    param: torch.Tensor,
    ignore_status: bool = False,
    name: str | None = None,
) -> torch.Tensor:
    """Gate the parameter with zero stage 3."""
    if hasattr(param, 'ds_id'):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(
    named_params: list[tuple[str, torch.Tensor]],
    keys_to_match: list[str],
) -> dict[str, torch.Tensor]:
    """Get the state of the adapter with zero stage 3."""
    to_return = {
        k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)
    }
    return {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}


def split_to_even_chunks(
    indices: list[int],
    lengths: list[int],
    num_chunks: int,
) -> list[list[int]]:
    """Split a list of indices into `chunks` chunks of roughly equal lengths."""

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float('inf')

    return chunks


# pylint: disable=too-many-locals
def get_modality_length_grouped_indices(
    lengths: list[int],
    batch_size: int,
    world_size: int,
    generator: Generator | None = None,
) -> list[int]:
    """Get indices grouped by modality and length."""
    # We need to use torch for the random part
    # as a distributed sampler will set the random seed for torch.
    assert all(length != 0 for length in lengths), 'Should not have zero length.'
    if all(length > 0 for length in lengths) or all(length < 0 for length in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, length) for i, length in enumerate(lengths) if length > 0])
    lang_indices, lang_lengths = zip(
        *[(i, -length) for i, length in enumerate(lengths) if length < 0],
    )

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths: list[int],
    batch_size: int,
    world_size: int,
    generator: Generator | None = None,
) -> list[int]:
    """Get indices grouped by length."""
    # We need to use torch for the random part
    # as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """
    Sampler that samples indices in a way that groups together
    features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(  # pylint: disable=too-many-arguments,super-init-not-called
        self,
        batch_size: int,
        world_size: int,
        lengths: list[int] | None = None,
        generator: Generator | None = None,
        group_by_modality: bool = False,
    ) -> None:
        if lengths is None:
            raise ValueError('Lengths must be provided.')

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Iterator[int]:
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths,
                self.batch_size,
                self.world_size,
                generator=self.generator,
            )
        return iter(indices)


class RewardTrainer(Trainer):
    """Reward Trainer for training the model with the reward signal."""

    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )

        return super()._get_train_sampler()  # pylint: disable=no-member

    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can
        pass a tuple in the Trainer's init through `optimizers`, or subclass and override this
        method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()  # pylint: disable=no-member
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if 'bias' not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if 'mm_projector' in name
                ]
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        'weight_decay': self.args.weight_decay,
                    },
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        'weight_decay': 0.0,
                    },
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        'weight_decay': self.args.weight_decay,
                        'lr': self.args.mm_projector_lr,
                    },
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        'weight_decay': 0.0,
                        'lr': self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay': self.args.weight_decay,
                    },
                    {
                        'params': [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        'weight_decay': 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == 'Adam8bit':

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {p.data_ptr(): p.numel() for p in module.parameters()}.values(),
                        )
                        logger.info('skipped: %sM params', skipped / 2**20)
                        manager.register_module_override(module, 'weight', {'optim_bits': 32})
                        logger.debug('bitsandbytes: will optimize %s in fp32', module)
                # logger.info(f'skipped: {skipped/2**20}M params')
                logger.info('skipped: %sM params', skipped / 2**20)

        return self.optimizer

    def _save_checkpoint(self, model: Any, trial: Any, metrics: Any | None = None) -> None:
        if getattr(self.args, 'tune_mm_mlp_adapter', False):

            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, 'use_im_start_end', False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(),
                keys_to_match,
            )

            # if self.args.local_rank == 0 or self.args.local_rank == -1:
            if self.args.local_rank in (-1, 0):
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
        else:
            super()._save_checkpoint(model, trial, metrics)  # pylint: disable=no-member

    def _save(self, output_dir: str | None = None, state_dict: bool | None = None) -> None:
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super()._save(output_dir, state_dict)  # pylint: disable=no-member

    def compute_loss(
        self,
        model: Any,
        inputs: torch.Tensor,
        return_outputs: bool = False,
    ) -> float | tuple[float, Any]:
        assert inputs['input_ids'].size(0) % 2 == 0, 'Batch size should be even.'

        outputs: ScoreModelOutput = model(**inputs)
        end_scores = outputs.end_scores
        higher_end_scores, lower_end_scores = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
        loss = -F.logsigmoid(  # pylint: disable=not-callable
            higher_end_scores - lower_end_scores,
        ).mean()
        loss = loss + 0.001 * end_scores.square().mean()
        accuracy = (higher_end_scores > lower_end_scores).float().mean()
        accuracy = get_all_reduce_mean(accuracy)
        self.log({'train_accuracy': accuracy.item()})

        outputs.accuracy = accuracy
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,  # pylint: disable=unused-argument
        ignore_keys: list[str] | None = None,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            end_scores = outputs.end_scores
            higher_end_scores, lower_end_scores = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
            labels = (higher_end_scores > lower_end_scores).float()

        return (
            None,
            end_scores,
            labels,
        )
