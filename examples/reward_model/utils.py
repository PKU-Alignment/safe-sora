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

"""Some utility functions for reward model training."""

import torch
import transformers

from safe_sora.conversations import SeparatorStyle, conv_templates
from safe_sora.models.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_IMAGE_LENGTH,
    MAX_VIDEO_LENGTH,
)


chat_template = conv_templates['video_rm']


def tokenizer_image_token(
    prompt: str,
    tokenizer: transformers.PreTrainedTokenizer,
    image_token_index: int = IMAGE_TOKEN_INDEX,
    return_tensors: str | None = None,
) -> torch.Tensor:
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X: list, sep: list) -> list:
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_multimodal(
    sources: list[str],
    num_frames: int,
    mm_use_im_start_end: bool,
) -> dict:
    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) or sentence['value'].startswith(
                DEFAULT_VIDEO_TOKEN,
            ):  # run with multi-im, multi-vid, multi-im & multi-vid
                # <video><video><image><image>\nxxxxxxxxxxxxx  # must <video> first
                # <image>\nxxxxxxxxxxxxx -> <image>\nxxxxxxxxxxxxx
                # <video>\nxxxxxxxxxxxxx -> <video>\nxxxxxxxxxxxxx

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = (
                        sentence['value']
                        .replace(
                            DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM,
                            DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH,
                        )
                        .strip()
                    )
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")

            # a <video> is treated as `num_frames * <image>`
            replace_token, vid_replace_token = (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IMAGE_TOKEN * num_frames,
            )
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                vid_replace_token = (
                    DEFAULT_VID_START_TOKEN + vid_replace_token + DEFAULT_VID_END_TOKEN
                )

            # <video><video><image><image>\nxxxxxxxxxxxxx -> `num_frames*<image>``num_frames*<image>`<image><image>\nxxxxxxxxxxxxx
            # <video>\nxxxxxxxxxxxxx -> `num_frames*<image>`\nxxxxxxxxxxxxx
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, vid_replace_token)
            # ======================================================================================================

    return sources


def preprocess_text(
    sources: list[list[dict[str, str]]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = chat_template.copy()
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors='pt')
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors='pt',
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ': '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for _, rou in enumerate(rounds):
            if rou == '':
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)')

    return {
        'input_ids': input_ids,
        'labels': targets,
    }
