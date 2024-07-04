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

VIDEO_DIR="./SafeSora/videos"
TRAIN_DATA_PATH="./SafeSora/config-train.json.gz"
EVAL_DATA_PATH="./SafeSora/config-test.json.gz"
MODEL_NAME_OR_PATH="LanguageBind/Video-LLaVA-7B"
MM_MLP_ADAPTER_PATH="LanguageBind/Video-LLaVA-Pretrain-7B/mm_projector.bin"
OUTPUT_DIR="./outputs"
DIMENSION="helpfulness"

while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
    --video_dir)
        VIDEO_DIR="$1"
        shift
        ;;
    --video_dir=*)
        VIDEO_DIR="${arg#*=}"
        ;;
    --train_data_path)
        TRAIN_DATA_PATH="$1"
        shift
        ;;
    --train_data_path=*)
        TRAIN_DATA_PATH="${arg#*=}"
        ;;
    --eval_data_path)
        EVAL_DATA_PATH="$1"
        shift
        ;;
    --eval_data_path=*)
        EVAL_DATA_PATH="${arg#*=}"
        ;;
    --model_name_or_path)
        MODEL_NAME_OR_PATH="$1"
        shift
        ;;
    --model_name_or_path=*)
        MODEL_NAME_OR_PATH="${arg#*=}"
        ;;
    --mm_mlp_adapter_path)
        MM_MLP_ADAPTER_PATH="$1"
        shift
        ;;
    --mm_mlp_adapter_path=*)
        MM_MLP_ADAPTER_PATH="${arg#*=}"
        ;;
    --output_dir)
        OUTPUT_DIR="$1"
        shift
        ;;
    --output_dir=*)
        OUTPUT_DIR="${arg#*=}"
        ;;
    --dimension)
        DIMENSION="$1"
        shift
        ;;
    --dimension=*)
        DIMENSION="${arg#*=}"
        ;;
    *)
        echo "Unknown parameter passed: '${arg}'" >&2
        exit 1
        ;;
    esac
done

if [[ ! "helpfulness harmlessness instruction_following correctness informativeness aesthetics" =~ (^|[[:space:]])"${DIMENSION}"($|[[:space:]]) ]]; then
    echo "Invalid dimension: ${DIMENSION}, should be one of 'helpfulness', 'harmlessness', 'instruction_following', 'correctness', 'informativeness', 'aesthetics'." >&2
    exit 1
fi

IMAGE_DIR="${VIDEO_DIR}"
RUN_NAME="reward-${DIMENSION}"
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
    echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
    comm -23 \
        <(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
        <(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
        shuf | head -n 1
)"

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

deepspeed --master_port="${MASTER_PORT}" examples/reward_model/train_reward.py \
    --deepspeed examples/scripts/ds_zero2.json \
    --version v1 \
    --run_name "${RUN_NAME}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --eval_data_path "${EVAL_DATA_PATH}" \
    --preference_dimension "${DIMENSION}" \
    --image_dir "${IMAGE_DIR}" \
    --video_dir "${VIDEO_DIR}" \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter "${MM_MLP_ADAPTER_PATH}" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "./models/cache_dir" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 0.0499 \
    --load_best_model_at_end True \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --logging_first_step True \
    --save_strategy "steps" \
    --save_steps 0.0499 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --bf16 True \
    --tf32 True \
    --num_frames 8
