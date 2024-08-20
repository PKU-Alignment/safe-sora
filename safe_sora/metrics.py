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

"""Traditional metrics for video generation."""

import os

import clip  # pylint: disable=import-error
import cv2
import hpsv2  # pylint: disable=import-error
import numpy as np
import torch
from PIL import Image
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as psnr


def extract_frames(video_path: str) -> tuple:
    """Extract frames from video file."""
    video_capture = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    if not video_capture.isOpened():
        print(f'Error opening video file: {video_path}')
        return None
    all_frames = []
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        all_frames.append(frame)
        frame_count += 1

    video_capture.release()
    return all_frames, frame_count


def psnr_reward(video_path: str) -> float:
    """Calculate the average PSNR of a video."""
    frames, frame_num = extract_frames(video_path)
    image_0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
    psnr_sum = 0
    for i in range(1, frame_num):
        image = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
        psnr_sum += psnr(image_0, image)
    return psnr_sum / frame_num


def hpsv2_reward(prompt: str, video_path: str, cache_dir: str, sample_rate: int = 1) -> float:
    """Calculate the average HPSv2 score of a video."""
    reward = []
    temp_save_path = os.path.join(cache_dir, 'temp.png')
    frames, frame_num = extract_frames(video_path)
    index = np.linspace(0, frame_num - 1, frame_num, dtype=int)
    frames = [frames[i] for i in index[:: int(1 / sample_rate)]]
    for _, frame in enumerate(frames):
        image = Image.fromarray(frame)
        image.save(temp_save_path)
        reward.append(hpsv2.score(temp_save_path, prompt, hps_version='v2.1'))
        os.remove(temp_save_path)
    mean_reward = np.mean(reward)
    return mean_reward.item()


class ClipReward:  # pylint: disable=too-few-public-methods
    """Calculate the average CLIP score of a video."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        print('device:', self.device)

    def __call__(self, prompt: str, video_path: str) -> float:
        frames, _ = extract_frames(video_path)
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        reward = []
        for i in len(frames):
            image = Image.fromarray(frames[i])
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, _ = self.model(image, text)
            reward.append(logits_per_image[0][0].item())
        return np.mean(reward)
