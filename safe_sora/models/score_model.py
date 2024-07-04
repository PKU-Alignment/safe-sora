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

"""Score model for the video captioning task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import LlamaConfig, LlamaModel
from transformers.modeling_utils import ModelOutput, PretrainedConfig
from transformers.models.llama import LlamaPreTrainedModel

from safe_sora.models.video_llava import LlavaMetaForCausalLM, LlavaMetaModel


@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, score_dim)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
        last_hidden_state (:obj:`torch.FloatTensor` of shape :
                obj:`(batch_size, sequence_length, hidden_dim)`
                ):
            Sequence of hidden-states at the output of the last layer of the model.
        end_last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_dim)`):
            Last hidden state of the sequence at the output of the last layer of the model.
        end_index (`torch.LongTensor` of shape `(batch_size,)`):
            Indices of the end of the sequence.
    """

    scores: torch.FloatTensor | None = None  # size = (B, L, D)
    end_scores: torch.FloatTensor | None = None  # size = (B, D)
    last_hidden_state: torch.FloatTensor | None = None  # size = (B, L, E)
    end_last_hidden_state: torch.FloatTensor | None = None  # size = (B, E)
    end_index: torch.LongTensor | None = None  # size = (B,)


class LlamaForScore(LlamaPreTrainedModel):  # pylint: disable=abstract-method
    """LlamaForScore is a model that predicts the score of the input sequence."""

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> Any:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: Any) -> Any:
        self.model.embed_tokens = value

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,  # pylint: disable=unused-argument
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ScoreModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ...,
            config.num_labels - 1]`.
                If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = transformer_outputs.last_hidden_state
        scores = self.score(last_hidden_state)  # size = (B, L, 1)

        end_index = torch.cat([m.nonzero()[-1] for m in attention_mask])  # size = (B,)
        end_last_hidden_state = torch.gather(  # size = (B, 1, E)
            last_hidden_state,
            dim=1,
            index=(
                end_index.to(last_hidden_state.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, last_hidden_state.size(-1))
            ),
        )
        end_scores = torch.gather(  # size = (B, 1, D)
            scores,
            dim=1,
            index=(
                end_index.to(scores.device)
                .unsqueeze(dim=1)
                .unsqueeze(dim=2)
                .expand(-1, -1, scores.size(-1))
            ),
        )
        end_last_hidden_state = end_last_hidden_state.squeeze(dim=1)  # size = (B, E)
        end_scores = end_scores.squeeze(dim=1)  # size = (B, D)

        if not return_dict:
            return scores, end_scores

        return ScoreModelOutput(
            scores=scores,
            end_scores=end_scores,
            last_hidden_state=last_hidden_state,
            end_last_hidden_state=end_last_hidden_state,
            end_index=end_index,
        )


class LlavaScoreConfig(LlamaConfig):  # pylint: disable=too-many-ancestors
    """Configuration for the llava score model"""

    model_type = 'llava_score'


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):  # pylint: disable=too-many-ancestors
    """LlavaLlamaModel."""

    config_class = LlavaScoreConfig

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)


class LlavaLlamaForScore(LlamaForScore, LlavaMetaForCausalLM):  # pylint: disable=too-many-ancestors
    """LlavaLlamaForScore is a model that predicts the score of the input sequence."""

    config_class = LlavaScoreConfig

    def __init__(self, config: LlamaConfig) -> None:
        super(LlamaForScore, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self) -> Any:
        return self.model

    # pylint: disable=too-many-arguments
    # pylint: disable=arguments-renamed
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        images: torch.FloatTensor | None = None,
        return_dict: bool | None = None,
    ) -> tuple | ScoreModelOutput:

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = (
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                )
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # pylint: disable=arguments-differ
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs,  # noqa
    ) -> dict:
        images = kwargs.pop('images', None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
