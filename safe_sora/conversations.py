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
# Adopted from https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/videollava/conversation.py
# and https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
# Its original license is Apache-2.0 License.

"""Conversations"""

from __future__ import annotations

import base64
import dataclasses
from enum import Enum, auto
from io import BytesIO
from typing import Any

from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    # pylint: disable=too-many-instance-attributes

    system: str
    roles: list[str]
    messages: list[list[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = '###'
    sep2: str = None
    version: str = 'Unknown'

    skip_next: bool = False

    def get_prompt(self) -> str:
        """
        Get the prompt of the conversation.
        """
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        messages = self.messages
        if len(messages) > 0 and isinstance(messages[0][1], tuple):
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace('<image>', '').strip()
            if 'mmtag' in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], '<Image><image></Image>'))
                messages.insert(1, (self.roles[1], 'Received.'))
            else:
                messages[0] = (init_role, '<image>\n' + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:

            def wrap_sys(msg: str) -> str:
                return f'<<SYS>>\n{msg}\n<</SYS>>\n\n'

            def wrap_inst(msg: str) -> str:
                return f'[INST] {msg} [/INST]'

            # wrap_sys = lambda msg: f'<<SYS>>\n{msg}\n<</SYS>>\n\n'
            # wrap_inst = lambda msg: f'[INST] {msg} [/INST]'

            ret = ''

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, 'first message should not be none'
                    assert role == self.roles[0], 'first message should come from user'
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += ' ' + message + ' ' + self.sep2
                else:
                    ret += ''
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (_role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ''
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

        return ret

    def append_message(self, role: str, message: Any) -> None:
        """
        Append role and message into self.messages.
        """
        self.messages.append([role, message])

    def get_images(self, return_pil: bool = False) -> list[Image.Image | str]:
        """
        Get images from self.messages.
        """
        # pylint: disable=too-many-locals
        images = []
        for i, (_role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0 and isinstance(msg, tuple):

                msg, image, image_process_mode = msg
                if image_process_mode == 'Pad':

                    def expand2square(
                        pil_img: Image.Image,
                        background_color: tuple = (122, 116, 104),
                    ) -> Image.Image:
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        if width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                    image = expand2square(image)
                elif image_process_mode in ['Default', 'Crop']:
                    pass
                elif image_process_mode == 'Resize':
                    image = image.resize((336, 336))
                else:
                    raise ValueError(f'Invalid image_process_mode: {image_process_mode}')
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 800, 400
                shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                longest_edge = int(shortest_edge * aspect_ratio)
                width, height = image.size
                if longest_edge != max(image.size):
                    if height > width:
                        height, width = longest_edge, shortest_edge
                    else:
                        height, width = shortest_edge, longest_edge
                    image = image.resize((width, height))
                if return_pil:
                    images.append(image)
                else:
                    buffered = BytesIO()
                    image.save(buffered, format='PNG')
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self) -> list[list[str]]:
        """
        Convert the conversation to a format that can be used in Gradio chatbot.
        """
        # pylint: disable=too-many-locals
        ret = []
        for i, (_role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if isinstance(msg, tuple):
                    msg, image = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    width, height = image.size
                    if height > width:
                        height, width = longest_edge, shortest_edge
                    else:
                        height, width = shortest_edge, longest_edge
                    image = image.resize((width, height))
                    buffered = BytesIO()
                    image.save(buffered, format='JPEG')
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = (
                        f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self) -> Conversation:
        """
        Return a copy of the conversation.
        """
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self) -> dict:
        """
        Return a dictionary representation of the conversation.
        """
        if len(self.get_images()) > 0:
            return {
                'system': self.system,
                'roles': self.roles,
                'messages': [[x, y[0] if isinstance(y, tuple) else y] for x, y in self.messages],
                'offset': self.offset,
                'sep': self.sep,
                'sep2': self.sep2,
            }
        return {
            'system': self.system,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
            'sep': self.sep,
            'sep2': self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system='A chat between a curious human and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=('Human', 'Assistant'),
    # pylint: disable=line-too-long
    messages=(
        (
            'Human',
            'What are the key differences between renewable and non-renewable energy sources?',
        ),
        (
            'Assistant',
            'Renewable energy sources are those that can be replenished naturally in a relatively '
            'short amount of time, such as solar, wind, hydro, geothermal, and biomass. '
            'Non-renewable energy sources, on the other hand, are finite and will eventually be '
            'depleted, such as coal, oil, and natural gas. Here are some key differences between '
            'renewable and non-renewable energy sources:\n'
            '1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable '
            'energy sources are finite and will eventually run out.\n'
            '2. Environmental impact: Renewable energy sources have a much lower environmental impact '
            'than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, '
            'and other negative effects.\n'
            '3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically '
            'have lower operational costs than non-renewable sources.\n'
            '4. Reliability: Renewable energy sources are often more reliable and can be used in more remote '
            'locations than non-renewable sources.\n'
            '5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different '
            'situations and needs, while non-renewable sources are more rigid and inflexible.\n'
            '6. Sustainability: Renewable energy sources are more sustainable over the long term, while '
            'non-renewable sources are not, and their depletion can lead to economic and social instability.\n',
        ),
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep='###',
)

conv_vicuna_v1 = Conversation(
    system='A chat between a curious user and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=('USER', 'ASSISTANT'),
    version='v1',
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=' ',
    sep2='</s>',
)

default_conversation = conv_vicuna_v1
conv_templates = {
    'default': conv_vicuna_v0,
    'vicuna_v0': conv_vicuna_v0,
    'vicuna_v1': conv_vicuna_v1,
    'video_rm': conv_vicuna_v1,
}

if __name__ == '__main__':
    print(default_conversation.get_prompt())
