#!/bin/bash
#
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

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

MODEL_NAME_OR_PATH="./checkpoints/reward-helpfulness"
EVAL_DATA_PATH="./SafeSora-Label/config-test.json.gz"
IMAGE_DIR="./SafeSora/videos"
VIDEO_DIR="./SafeSora/videos"

OUTPUT_DIR="examples/outputs"

while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
	--model_name_or_path)
		MODEL_NAME_OR_PATH="$1"
		shift
		;;
	--model_name_or_path=*)
		MODEL_NAME_OR_PATH="${arg#*=}"
		;;
	--eval_data_path)
		EVAL_DATA_PATH="$1"
		shift
		;;
	--image_dir=*)
		IMAGE_DIR="${arg#*=}"
		;;
	--image_dir)
		IMAGE_DIR="$1"
		shift
		;;
	--video_dir=*)
		VIDEO_DIR="${arg#*=}"
		;;
	--video_dir)
		VIDEO_DIR="$1"
		shift
		;;
	--eval_data_path=*)
		EVAL_DATA_PATH="${arg#*=}"
		;;
	--output_dir)
		OUTPUT_DIR="$1"
		shift
		;;
	--output_dir=*)
		OUTPUT_DIR="${arg#*=}"
		;;
	*)
		echo "Unknown parameter passed: '${arg}'" >&2
		exit 1
		;;
	esac
done

torchrun --nproc_per_node=8 \
	examples/reward_model/inference.py \
	--model_name_or_path "$MODEL_NAME_OR_PATH" \
	--cache_dir "./cache_dir" \
	--output_dir "$OUTPUT_DIR" \
	--model_max_length "2048" \
	--is_multimodal "True" \
	--image_aspect_ratio "pad" \
	--eval_data_path "$EVAL_DATA_PATH" \
	--image_dir "$IMAGE_DIR" \
	--video_dir "$VIDEO_DIR" \
	--num_frames "8" \
	--batch_size "8"
