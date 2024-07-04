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
# Adopted from https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/videollava/train/train.py
# Its original license is Apache-2.0 license.

"""Train a reward model for text-to-video generation."""

from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import transformers
from torch.utils.data import Dataset

from safe_sora.conversations import conv_templates
from safe_sora.datasets.base import VideoPairSample, VideoSample
from safe_sora.datasets.pair import PairDataset
from safe_sora.models import LlavaLlamaForScore
from safe_sora.models.constants import MAX_VIDEO_LENGTH
from safe_sora.models.video_llava import safe_save_model_for_hf_trainer
from safe_sora.trainers import RewardTrainer
from safe_sora.utils import order_pick_k
from utils import preprocess_multimodal, preprocess_text


local_rank = None
chat_template = conv_templates['video_rm']


@dataclass
class ModelArguments:
    model_name_or_path: str | None = field(default='facebook/opt-125m')
    version: str | None = field(default='v0')
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: str | None = field(default=None)
    mm_vision_select_layer: int | None = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: str | None = field(default=None)
    mm_projector_type: str | None = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: str | None = field(default='patch')

    # ===================================================================
    image_tower: str | None = field(default=None)
    video_tower: str | None = field(default=None)
    # ===================================================================


@dataclass
class DataArguments:
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    train_data_path: str | None = field(
        default=None,
        metadata={'help': 'Path to the training data.'},
    )
    eval_data_path: str | None = field(
        default=None,
        metadata={'help': 'Path to the evaluation data.'},
    )
    image_dir: str | None = field(default=None)
    video_dir: str | None = field(default=None)
    preference_dimension: Literal[
        'helpfulness',
        'harmlessness',
        'instruction_following',
        'correctness',
        'informativeness',
        'aesthetics',
    ] = 'helpfulness'
    num_frames: int = 8
    # ===================================================================


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: str | None = field(default=None)
    optim: str = field(default='adamw_torch')
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: str | None = field(default='triton')
    model_max_length: int = field(
        default=512,
        metadata={
            'help': 'Maximum list length. lists will be right padded (and possibly truncated).',
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={'help': 'Compress the quantization statistics through double quantization.'},
    )
    quant_type: str = field(
        default='nf4',
        metadata={'help': 'Quantization data type to use. Should be one of `fp4` or `nf4`.'},
    )
    bits: int = field(default=16, metadata={'help': 'How many bits to use.'})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    mm_projector_lr: float | None = None
    group_by_modality_length: bool = field(default=False)

    # ================================================
    tokenizer_model_max_length: int | None = None
    # ================================================


class LazyRewardPairDataset(Dataset):
    """Pair dataset for training reward model."""

    def __init__(
        self,
        config_path: str,
        video_dir: str,
        tokenizer: transformers.PreTrainedTokenizer,
        video_processor: transformers.VideoLlavaProcessor,
        data_args: DataArguments,
        preference_dimension: Literal[
            'helpfulness',
            'harmlessness',
            'instruction_following',
            'correctness',
            'informativeness',
            'aesthetics',
        ] = 'helpfulness',
    ) -> None:
        super().__init__()

        self.dataset = PairDataset.load(config_path, video_dir=video_dir)
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.preference_dimension = preference_dimension
        self.data_args = data_args

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def modality_lengths(self) -> list[int]:
        return [len(pair_config['prompt_text']) for pair_config in self.dataset]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pair_config: VideoPairSample = self.dataset[index]
        if self.preference_dimension in ['helpfulness', 'harmlessness']:
            higher_score_id = pair_config[self.preference_dimension]
        elif self.preference_dimension in [
            'instruction_following',
            'correctness',
            'informativeness',
            'aesthetics',
        ]:
            higher_score_id = pair_config['sub_preferences'][self.preference_dimension]
        else:
            raise ValueError('Unknown preference dimension.')
        lower_score_id = 'video_0' if higher_score_id == 'video_1' else 'video_1'

        def process_multimodal_data(video_id: str) -> dict:
            """Process VideoSample data into a dictionary for training."""
            video_config: VideoSample = pair_config[video_id]
            conversation = [
                {
                    'from': 'human',
                    'value': f"##Video Generation Prompt: {pair_config['prompt_text']}",
                },
                {'from': 'gpt', 'value': '##Generated Video: \n<video>'},
            ]
            video_file = video_config['video_path']
            video_file = video_file if isinstance(video_file, list) else [video_file]
            video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
            image = [
                self.video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video_file
            ]

            sources = preprocess_multimodal(
                copy.deepcopy([conversation]),
                num_frames=self.data_args.num_frames,
                mm_use_im_start_end=self.data_args.mm_use_im_start_end,
            )
            data_dict = preprocess_text(sources, self.tokenizer, has_image=True)
            return {
                'input_ids': data_dict['input_ids'][0],
                'labels': data_dict['labels'][0],
                'image': image,
            }

        return {
            'higher_score': process_multimodal_data(higher_score_id),
            'lower_score': process_multimodal_data(lower_score_id),
        }


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: list[dict]) -> dict[str, torch.Tensor]:
        higher_scores = [instance['higher_score'] for instance in instances]
        lower_scores = [instance['lower_score'] for instance in instances]
        all_scores = higher_scores + lower_scores

        input_ids = [instance['input_ids'] for instance in all_scores]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        images = [instance['image'] for instance in all_scores]
        new_images = []
        for image in images:
            if isinstance(image, list):
                for i in image:
                    new_images.append(i)
            else:
                new_images.append(image)
        images = new_images
        return {
            'input_ids': input_ids,
            'images': images,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }


def make_supervised_data_module(
    data_args: DataArguments,
    tokenizer: transformers.PreTrainedTokenizer,
    video_processor: transformers.VideoLlavaProcessor,
) -> dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = LazyRewardPairDataset(
        config_path=data_args.train_data_path,
        video_dir=data_args.video_dir,
        tokenizer=tokenizer,
        video_processor=video_processor,
        preference_dimension=data_args.preference_dimension,
        data_args=data_args,
    )
    eval_dataset = LazyRewardPairDataset(
        config_path=data_args.eval_data_path,
        video_dir=data_args.video_dir,
        tokenizer=tokenizer,
        video_processor=video_processor,
        preference_dimension=data_args.preference_dimension,
        data_args=data_args,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return {
        'train_dataset': train_dataset,
        'eval_dataset': eval_dataset,
        'data_collator': data_collator,
    }


def compute_metrics(data: transformers.EvalPrediction) -> dict:
    max_score = data.predictions.max().item()
    min_score = data.predictions.min().item()
    accuracy = data.label_ids.mean().item()
    return {'accuracy': accuracy, 'max_score': max_score, 'min_score': min_score}


def train() -> None:
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    if model_args.image_tower is None and model_args.video_tower is None:
        raise ValueError('At least one of `image_tower` and `video_tower` should be set.')

    model = LlavaLlamaForScore.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module: Any, input: Any, output: Any) -> None:
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='right',
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    # =============================================================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None:
        # print(model_args)
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            image_tower.to(
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                device=training_args.device,
            )

            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            video_tower.to(
                dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
                device=training_args.device,
            )

            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
        # =============================================================================================================

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side

        # =============================================================================================================
        tokenizer_model_max_length = training_args.tokenizer_model_max_length
        model.config.tokenizer_model_max_length = (
            tokenizer.model_max_length
            if tokenizer_model_max_length is None
            else tokenizer_model_max_length
        )
        # =============================================================================================================

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(
        data_args=data_args,
        tokenizer=tokenizer,
        video_processor=data_args.video_processor,
    )
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob('checkpoint-*')):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train(
            ignore_keys_for_eval=[
                'scores',
                'last_hidden_state',
                'end_last_hidden_state',
                'end_index',
            ],
        )
    trainer.evaluate()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == '__main__':
    train()
