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
# Adopted from https://github.com/PKU-YuanGroup/Video-LLaVA project
# Its original license is Apache-2.0 License.

"""Constants used in the video-llava model."""

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = '.'

# Model Constants
IGNORE_INDEX = -100

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IMAGE_PATCH_TOKEN = '<im_patch>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'
IMAGE_PLACEHOLDER = '<image-placeholder>'

# ==============================
DEFAULT_VIDEO_TOKEN = '<video>'
DEFAULT_VIDEO_PATCH_TOKEN = '<im_patch>'
DEFAULT_VID_START_TOKEN = '<vid_start>'
DEFAULT_VID_END_TOKEN = '<vid_end>'
VIDEO_PLACEHOLDER = '<video-placeholder>'
# ==============================

MAX_IMAGE_LENGTH = 16
MAX_VIDEO_LENGTH = 1  # current video datasets only have 1 video?

PAD_LENGTH = 620
