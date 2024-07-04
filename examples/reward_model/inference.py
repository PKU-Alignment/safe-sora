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
# Adopted from https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/videollava/eval/video/run_inference_benchmark_general.py
# Its original license is Apache-2.0 license.

"""Inference script for reward model."""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Sequence

import torch
import transformers
from torch import distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from safe_sora.conversations import conv_templates
from safe_sora.datasets.video import VideoDataset
from safe_sora.models import LlavaLlamaForScore
from safe_sora.models.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN,
    MAX_VIDEO_LENGTH,
)
from safe_sora.models.score_model import ScoreModelOutput
from safe_sora.utils import order_pick_k
from utils import preprocess_multimodal, preprocess_text


chat_template = conv_templates['video_rm']


def distributed_max(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the maximum value of a tensor across all workers."""
    if not dist.is_initialized():
        logging.warning('Max without distributed initialization.')
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def distributed_gather(to_gather: torch.Tensor) -> torch.Tensor:
    """Gather tensors from all workers."""
    if not dist.is_initialized():
        logging.warning('Gathering without distributed initialization.')
        return to_gather

    if not to_gather.is_cuda:
        to_gather = to_gather.to(f'cuda:{dist.get_rank()}')

    world_size = dist.get_world_size()

    length = torch.tensor([to_gather.shape[0]], device=to_gather.device)
    lengths = [torch.zeros_like(length) for _ in range(world_size)]
    dist.all_gather(lengths, length)
    max_length = max(lengths).item()

    if to_gather.shape[0] < max_length:
        padding = torch.zeros(
            max_length - to_gather.shape[0],
            *to_gather.shape[1:],
            device=to_gather.device,
        )
        to_gather = torch.cat([to_gather, padding], dim=0)

    gathered = [torch.zeros_like(to_gather) for _ in range(world_size)]
    dist.all_gather(gathered, to_gather)

    for i, length in enumerate(lengths):
        gathered[i] = gathered[i][: length.item()]

    return torch.cat(gathered, dim=0)


@dataclass
class EvalArguments:
    model_name_or_path: str | None = field(default='facebook/opt-125m')
    cache_dir: str | None = field(default=None)
    output_dir: str | None = field(default=None)
    model_max_length: int | None = field(default=2048)
    batch_size: int | None = field(default=4)


@dataclass
class DataArguments:
    is_multimodal: bool = True
    image_aspect_ratio: str = 'square'
    # ===================================================================
    eval_data_path: str | None = field(
        default=None,
        metadata={'help': 'Path to the evaluation data.'},
    )
    image_dir: str | None = field(default=None)
    video_dir: str | None = field(default=None)
    num_frames: int = 8
    # ===================================================================


class LazyVideoDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        config_path: str,
        video_dir: str,
        tokenizer: transformers.PreTrainedTokenizer,
        video_processor: transformers.VideoLlavaProcessor,
        data_args: DataArguments,
    ) -> None:
        super().__init__()

        self.dataset = VideoDataset.load(config_path, video_dir=video_dir)
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.data_args = data_args

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def modality_lengths(self) -> list[int]:
        return [len(video_config['prompt_text'] for video_config in self.dataset)]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        video_config = self.dataset[index]
        conversation = [
            {
                'from': 'human',
                'value': f"##Video Generation Prompt: {video_config['prompt_text']}",
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
            'index': index,
            'input_ids': data_dict['input_ids'][0],
            'labels': data_dict['labels'][0],
            'image': image,
        }


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]

        images = [instance['image'] for instance in instances]
        new_images = []
        for image in images:
            if isinstance(image, list):
                for i in image:
                    new_images.append(i)
            else:
                new_images.append(image)
        return {
            'index': [instance['index'] for instance in instances],
            'input_ids': input_ids,
            'images': new_images,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }


def load_pretrained_model(
    model_path: str,
    device: str = 'cuda',
    **kwargs,  # noqa
) -> tuple[transformers.PreTrainedTokenizer, LlavaLlamaForScore, dict]:
    """Load a pretrained model."""

    kwargs['device_map'] = {'': device}
    kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForScore.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # ==========================================================================================================
    processor = {'image': None, 'video': None}

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    mm_use_im_patch_token = getattr(model.config, 'mm_use_im_patch_token', True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if model.config.mm_image_tower is not None:
        image_tower = model.get_image_tower()
        if not image_tower.is_loaded:
            image_tower.load_model()
        image_tower.to(device=device, dtype=torch.float16)
        image_processor = image_tower.image_processor
        processor['image'] = image_processor

    if model.config.mm_video_tower is not None:
        video_tower = model.get_video_tower()
        if not video_tower.is_loaded:
            video_tower.load_model()
        video_tower.to(device=device, dtype=torch.float16)
        video_processor = video_tower.video_processor
        processor['video'] = video_processor

    return tokenizer, model, processor


def main() -> None:
    parser = transformers.HfArgumentParser((EvalArguments, DataArguments))
    eval_args, data_args = parser.parse_args_into_dataclasses()

    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    eval_args.device = f'cuda:{local_rank}'

    tokenizer, model, processor = load_pretrained_model(
        eval_args.model_name_or_path,
        device=eval_args.device,
    )
    model = model.to(eval_args.device)

    data_args.video_processor = processor['video']
    data_args.mm_use_im_start_end = model.config.mm_use_im_start_end
    data_args.mm_use_im_patch_token = model.config.mm_use_im_patch_token

    dataset = LazyVideoDataset(
        config_path=data_args.eval_data_path,
        video_dir=data_args.video_dir,
        tokenizer=tokenizer,
        video_processor=processor['video'],
        data_args=data_args,
    )
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=eval_args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        shuffle=False,
    )

    bar = tqdm(total=len(dataloader), desc='Inference ...', disable=local_rank != 0)

    results = {
        'index': [],
        'end_scores': [],
    }
    for _, batch in enumerate(dataloader):
        indexes = batch['index']
        batch = {
            'input_ids': batch['input_ids'].to(model.device),
            'images': [image.half().to(model.device) for image in batch['images']],
            'attention_mask': batch['attention_mask'].to(model.device),
        }

        with torch.no_grad():
            outputs: ScoreModelOutput = model(**batch)
            end_scores = outputs.end_scores
            input_ids = batch['input_ids']
            input_ids[input_ids == -200] = tokenizer.eos_token_id

            for index, end_score in zip(indexes, end_scores):
                results['index'].append(index)
                results['end_scores'].append(end_score.item())

        bar.update(1)
    dist.barrier()
    indexes = distributed_gather(torch.tensor(results['index']).to(eval_args.device))
    end_scores = distributed_gather(torch.tensor(results['end_scores']).to(eval_args.device))
    dist.barrier()

    full_configs = dataset.dataset.configs
    for index, end_score in zip(indexes, end_scores):
        full_config = full_configs[index]
        if 'scores' not in full_config:
            full_config['scores'] = []
        full_config['scores'].append(
            {
                'value': end_score.item(),
                'from': eval_args.model_name_or_path,
            },
        )

    bar.close()

    if local_rank == 0:
        basename = os.path.basename(data_args.eval_data_path).split('.')[0]
        save_path = os.path.join(eval_args.output_dir, f'{basename}.json')
        os.makedirs(eval_args.output_dir, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(full_configs, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
