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
"""Base Types for SafeSora."""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil
from typing import Iterable, Literal
from typing_extensions import NotRequired, Required, TypedDict  # Python 3.11+; Python 3.10+

from torch.utils.data import Dataset
from torchvision.io import read_video
from tqdm import tqdm


__all__ = [
    'HarmLabel',
    'VideoSample',
    'VideoPairSample',
    'SubPreference',
    'BaseDataset',
]


def is_complete(data_dict: dict) -> bool:
    """Check if a dictionary is complete, i.e., all values are not None."""
    for _, value in data_dict.items():
        if isinstance(value, dict) and not is_complete(value):
            return False
        if value is None:
            return False

    return True


def check_video_integrity(video_path: str) -> bool:
    """Check if video is corrupted"""
    try:
        vframes, _, _ = read_video(video_path, pts_unit='sec')
        if vframes.shape[0] == 0:
            return False
        return True
    except Exception:  # noqa: BLE001 # pylint: disable=broad-except
        return False


class HarmLabel(TypedDict, total=False):
    """Harm labels for harmfulness classification"""

    porn: Required[bool]
    violence: Required[bool]
    hate: Required[bool]
    terrorism: Required[bool]
    contraband: Required[bool]
    controversial: Required[bool]
    racism: NotRequired[bool]
    other_discrimination: Required[bool]
    animal_abuse: Required[bool]
    child_abuse: Required[bool]
    crime: Required[bool]
    other_harmful: Required[bool]


def format_harm_label_from_dict(data: dict) -> HarmLabel:
    """
    Converts a dictionary into a VideoSample object.

    This function takes a dictionary where each key corresponds to a property of a HarmLabel and
    converts it into a HarmLabel object. Fields are handled accordingly, defaulting to None if not
    provided.
    """
    return HarmLabel(
        porn=data.get('porn'),
        violence=data.get('violence'),
        hate=data.get('hate'),
        terrorism=data.get('terrorism'),
        contraband=data.get('contraband'),
        controversial=data.get('controversial'),
        racism=data.get('racism'),
        other_discrimination=data.get('other_discrimination'),
        animal_abuse=data.get('animal_abuse'),
        child_abuse=data.get('child_abuse'),
        crime=data.get('crime'),
        other_harmful=data.get('other_harmful'),
    )


class PromptSample(TypedDict):
    """Text prompt sample."""

    prompt_id: Required[str]
    prompt_text: Required[str]
    prompt_type: NotRequired[Literal['safety_critical', 'safety_neutral']]
    prompt_labels: NotRequired[HarmLabel]


def format_prompt_sample_from_dict(data: dict, contain_labels: bool = False) -> PromptSample:
    """
    Converts a dictionary into a PromptSample object.

    This function takes a dictionary where each key corresponds to a property of a PromptSample and
    converts it into a PromptSample object. Fields are handled accordingly, defaulting to None if
    not provided.
    """

    if contain_labels:
        prompt_labels = data.get('prompt_labels')
        if prompt_labels is not None:
            prompt_labels = format_harm_label_from_dict(prompt_labels)

        return PromptSample(
            prompt_id=data.get('prompt_id'),
            prompt_text=data.get('prompt_text'),
            prompt_type=data.get('prompt_type'),
            prompt_labels=prompt_labels,
        )

    return PromptSample(
        prompt_id=data.get('prompt_id'),
        prompt_text=data.get('prompt_text'),
        prompt_type=data.get('prompt_type'),
    )


class VideoSample(TypedDict):
    """Video sample"""

    prompt_id: Required[str]
    prompt_text: Required[str]
    prompt_type: NotRequired[Literal['safety_critical', 'safety_neutral']]
    prompt_labels: NotRequired[HarmLabel]
    video_id: Required[str]
    video_text: Required[str]
    video_path: Required[str]
    is_safe: NotRequired[bool]
    video_labels: NotRequired[HarmLabel]
    generated_from: NotRequired[str]


def format_video_sample_from_dict(data: dict, contain_labels: bool = False) -> VideoSample:
    """
    Converts a dictionary into a VideoSample object.

    This function takes a dictionary where each key corresponds to a property of a VideoSample and
    converts it into a VideoSample object. Fields are handled accordingly, defaulting to None if
    not provided.
    """
    if contain_labels:
        video_labels = data.get('video_labels')
        if video_labels is not None:
            video_labels = format_harm_label_from_dict(video_labels)
        prompt_labels = data.get('prompt_labels')
        if prompt_labels is not None:
            prompt_labels = format_harm_label_from_dict(prompt_labels)

        return VideoSample(
            prompt_id=data.get('prompt_id'),
            prompt_text=data.get('prompt_text'),
            prompt_type=data.get('prompt_type'),
            prompt_labels=prompt_labels,
            video_id=data.get('video_id'),
            video_text=data.get('video_text'),
            video_path=data.get('video_path'),
            is_safe=data.get('is_safe'),
            video_labels=video_labels,
            generated_from=data.get('generated_from'),
        )

    return VideoSample(
        prompt_id=data.get('prompt_id'),
        prompt_text=data.get('prompt_text'),
        prompt_type=data.get('prompt_type'),
        video_id=data.get('video_id'),
        video_text=data.get('video_text'),
        video_path=data.get('video_path'),
        is_safe=data.get('is_safe'),
        generated_from=data.get('generated_from'),
    )


class SubPreference(TypedDict, total=False):
    """Sub-preferences for helpfulness dimension"""

    instruction_following: NotRequired[Literal['video_0', 'video_1']]
    correctness: NotRequired[Literal['video_0', 'video_1']]
    informativeness: NotRequired[Literal['video_0', 'video_1']]
    aesthetics: NotRequired[Literal['video_0', 'video_1']]


def format_sub_preference_from_dict(data: dict) -> SubPreference:
    """
    Converts a dictionary into a SubPreference object.

    This function takes a dictionary where each key corresponds to a property of a SubPreference and
    converts it into a SubPreference object. Fields are handled accordingly, defaulting to None if
    not provided.
    """
    return SubPreference(
        instruction_following=data.get('instruction_following'),
        correctness=data.get('correctness'),
        informativeness=data.get('informativeness'),
        aesthetics=data.get('aesthetics'),
    )


class VideoPairSample(TypedDict):
    """A Comparison pair contains one user prompt and two video samples."""

    pair_id: Required[str]
    prompt_id: Required[str]
    prompt_text: Required[str]
    prompt_type: NotRequired[Literal['safety_critical', 'safety_neutral']]
    video_0: Required[VideoSample]
    video_1: Required[VideoSample]
    helpfulness: NotRequired[Literal['video_0', 'video_1']]
    harmlessness: NotRequired[Literal['video_0', 'video_1']]
    sub_preferences: NotRequired[SubPreference]


def format_video_pair_sample_from_dict(data: dict) -> VideoPairSample:
    """
    Converts a dictionary into a VideoPairSample object.

    This function takes a dictionary where each key corresponds to a property of a VideoPairSample
    and converts it into a VideoPairSample object. Optional fields are handled accordingly,
    defaulting to None if not provided.
    """

    video_0 = data.get('video_0')
    if video_0 is not None:
        video_0 = format_video_sample_from_dict(video_0)

    video_1 = data.get('video_1')
    if video_1 is not None:
        video_1 = format_video_sample_from_dict(video_1)

    sub_preferences = data.get('sub_preferences')
    if sub_preferences is not None:
        sub_preferences = format_sub_preference_from_dict(sub_preferences)

    return VideoPairSample(
        pair_id=data.get('pair_id'),
        prompt_id=data.get('prompt_id'),
        prompt_text=data.get('prompt_text'),
        prompt_type=data.get('prompt_type'),
        video_0=video_0,
        video_1=video_1,
        helpfulness=data.get('helpfulness'),
        harmlessness=data.get('harmlessness'),
        sub_preferences=sub_preferences,
    )


class BaseDataset(Dataset):
    """Base class for datasets"""

    def __init__(self, configs: list[dict]) -> None:
        self.configs = configs

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, index: int) -> dict:
        return self.configs[index]

    @property
    def num_prompts(self) -> int:
        """The number of prompts in the dataset"""
        raise NotImplementedError

    @property
    def num_pairs(self) -> int:
        """The number of pairs in the dataset"""
        raise NotImplementedError

    @property
    def num_videos(self) -> int:
        """The number of unique videos in the dataset"""
        raise NotImplementedError

    @property
    def videos(self) -> Iterable[VideoSample]:
        """A iterable object of all video samples"""
        raise NotImplementedError

    @classmethod
    def load(
        cls,
        path: str,
        config_filename: str = 'config.json',
        video_dir: str | None = None,
    ) -> BaseDataset:
        """Load dataset from disk.

        Args:
            path: Path to the dataset. If `path` is a directory, it will look for `config_filename`
                as the configuration file and a directory named `videos` for video files. If `path`
                is a file, it will only load the configuration from the file.
            config_filename: The name of the configuration file.
            video_dir: The directory where the video files are stored. If `video_dir` is not None,
                it will force to set the given video directory for all video samples.
        """

        if os.path.isdir(path):
            config_path = os.path.join(path, config_filename)
            if not os.path.exists(config_path):
                raise FileNotFoundError(f'`{config_filename}` not found in {path}')

            if config_filename.endswith('.json'):
                with open(config_path, encoding='utf-8') as file:
                    configs = json.load(file)
            elif config_filename.endswith('.json.gz'):
                with gzip.open(config_path, 'rt', encoding='utf-8') as file:
                    configs = json.load(file)
            else:
                raise ValueError(f'Unsupported file format: {config_filename}')

            video_dir = os.path.join(path, 'videos') if video_dir is None else video_dir

        elif path.endswith('.json'):
            with open(path, encoding='utf-8') as file:
                configs = json.load(file)

        elif path.endswith('.json.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as file:
                configs = json.load(file)

        else:
            raise ValueError(f'Unsupported file format: {path}')

        dataset = cls(configs)
        if video_dir is not None:
            dataset.set_video_dir(video_dir)

        return dataset

    def save(
        self,
        path: str,
        config_filename: str = 'config.json',
        copy_videos: bool = False,
    ) -> None:
        """Save dataset to disk.

        Args:
            path: Path to save the dataset. If `path` is a directory, it will save the configuration
                file as `config_filename` and, at the same time, if `copy_videos` is True, it will
                copy all video files to a directory named `videos` in the same directory. If `path`
                is a json file, it will only save the configuration to the file.
            config_filename: The name of the configuration file.
            copy_videos: Copy video files to a directory named `videos` in the same directory.
        """

        if os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

            if copy_videos:
                video_dir = os.path.join(path, 'videos')

                tqdm_bar = tqdm(total=len(self.videos), desc='Copying videos')
                for video_config in self.videos:
                    src_path = video_config.get('video_path')
                    dst_path = os.path.join(
                        video_dir,
                        video_config['prompt_id'],
                        f"{video_config['video_id']}.mp4",
                    )
                    video_config['video_path'] = os.path.relpath(dst_path, path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    if src_path is not None and os.path.exists(src_path):
                        if src_path != dst_path:
                            shutil.copy2(src_path, dst_path)
                    else:
                        with tqdm.external_write_mode():
                            print(f"Video path not found for video {video_config['video_id']}")
                    tqdm_bar.update(1)
                tqdm_bar.close()

            config_path = os.path.join(path, config_filename)

            if config_filename.endswith('.json'):
                with open(config_path, 'w', encoding='utf-8') as file:
                    json.dump(self.configs, file, indent=4, ensure_ascii=False)
            elif config_filename.endswith('.json.gz'):
                with gzip.open(config_path, 'wt', encoding='utf-8') as file:
                    json.dump(self.configs, file, indent=4, ensure_ascii=False)
            else:
                raise ValueError(f'Unsupported file format: {config_filename}')

        elif path.endswith('.json'):
            with open(path, 'w', encoding='utf-8') as file:
                json.dump(self.configs, file, indent=4, ensure_ascii=False)

        elif path.endswith('.json.gz'):
            with gzip.open(path, 'wt', encoding='utf-8') as file:
                json.dump(self.configs, file, indent=4, ensure_ascii=False)

        else:
            raise ValueError(f'Unsupported file format: {path}')

    def set_video_dir(self, video_dir: str) -> None:
        """Set video directory for all video samples"""

        for video_config in self.videos:
            video_config['video_path'] = os.path.abspath(
                os.path.join(
                    video_dir,
                    video_config['prompt_id'],
                    f"{video_config['video_id']}.mp4",
                ),
            )

    def is_complete(self) -> bool:
        """Check if the dataset is complete, i.e., all values are not None."""
        return all(is_complete(config) for config in self.configs)

    def check_video_integrity(self) -> None:
        """Check if videos are corrupted"""

        has_checked_path = set()
        num_corrupted_videos = 0
        tqdm_bar = tqdm(total=len(self.videos), desc='Number of corrupted videos: 0')

        for video_config in self.videos:
            video_path = video_config.get('video_path')

            if video_path is None:
                num_corrupted_videos += 1
                with tqdm.external_write_mode():
                    print(f"Video path not found for video {video_config['video_id']}")

            elif video_path not in has_checked_path:

                has_checked_path.add(video_path)
                if not check_video_integrity(video_path):
                    num_corrupted_videos += 1
                    with tqdm.external_write_mode():
                        print(f'Corrupted video: {video_path}')

            tqdm_bar.set_description(f'Number of corrupted videos: {num_corrupted_videos}')
            tqdm_bar.update(1)
        tqdm_bar.close()

    def shuffle(self) -> None:
        """Shuffle the dataset"""
        random.shuffle(self.configs)
