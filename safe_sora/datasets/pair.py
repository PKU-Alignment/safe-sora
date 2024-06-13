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
"""Pair dataset for Representing Human Preferences."""

from __future__ import annotations

from typing import Iterable

from safe_sora.datasets.base import (
    BaseDataset,
    VideoPairSample,
    VideoSample,
    format_video_pair_sample_from_dict,
)


class PairDataset(BaseDataset):
    """Pair dataset for Representing Human Preferences."""

    FORMAT_CONFIGS: bool = True

    def __init__(self, configs: list[dict]) -> None:
        super().__init__(configs)

        if self.FORMAT_CONFIGS:
            self.configs = [format_video_pair_sample_from_dict(config) for config in self.configs]

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, index: int) -> VideoPairSample:
        return self.configs[index]

    @property
    def num_prompts(self) -> int:
        return len({config['prompt_id'] for config in self.configs})

    @property
    def num_pairs(self) -> int:
        return len(self.configs)

    @property
    def num_videos(self) -> int:
        return len({video['video_id'] for video in self.videos})

    @property
    def videos(self) -> Iterable[VideoSample]:
        videos = []
        for config in self.configs:
            videos.append(config['video_0'])
            videos.append(config['video_1'])
        return videos
