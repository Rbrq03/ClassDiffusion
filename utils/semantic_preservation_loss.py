from ast import Dict
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)

from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)

from transformers import CLIPTextModel


def forward_with_custom_embeddings(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    input_modifier_embeddings: Optional[torch.Tensor] = None,
    modifier_token_id: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:

    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    use_custom_embeddings = (
        input_modifier_embeddings is not None and modifier_token_id is not None
    )

    if input_ids is None:
        raise ValueError("You have to specify input_ids")

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])

    hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

    if use_custom_embeddings:
        modifier_index = torch.where(input_ids.squeeze(0) == modifier_token_id)
        hidden_states[0, modifier_index, :] = input_modifier_embeddings

    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _create_4d_causal_attention_mask(
        input_shape, hidden_states.dtype, device=hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    if self.eos_token_id == 2:
        # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
        # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
        # ------------------------------------------------------------
        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]
    else:
        # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
            (
                input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                == self.eos_token_id
            )
            .int()
            .argmax(dim=-1),
        ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )


class SPL:

    def __init__(
        self,
        text_encoder: CLIPTextModel,
        use_attention_mask: Optional[bool] = None,
    ):
        self.text_encoder = text_encoder
        self.use_attention_mask = use_attention_mask

    def _encode(
        self,
        text_input: Optional[Dict] = None,
        input_modifier_embeddings: Optional[torch.Tensor] = None,
        modifier_token_id: Optional[torch.Tensor] = None,
    ):

        text_input_ids = text_input.input_ids
        device = self.text_encoder.device

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_input.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder.text_model.forward_with_custom_embeddings(
            text_input_ids.to(device),
            attention_mask=attention_mask,
            input_modifier_embeddings=input_modifier_embeddings,
            modifier_token_id=modifier_token_id,
        )

        prompt_embeds = prompt_embeds.pooler_output

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        return prompt_embeds

    def __call__(
        self,
        text_embeddings: Optional[torch.Tensor],
        modifier_token_id: Optional[torch.Tensor],
        modifier_cls_text_input: Optional[Dict],
        cls_text_input: Optional[Dict],
    ) -> torch.Tensor:

        modifier_embedding = text_embeddings[modifier_token_id].clone()

        modifier_cls_embedding = self._encode(
            text_input=modifier_cls_text_input,
            input_modifier_embeddings=modifier_embedding,
            modifier_token_id=modifier_token_id,
        )

        cls_embedding = self._encode(text_input=cls_text_input)

        dis = F.cosine_similarity(modifier_cls_embedding, cls_embedding)

        dis = 1 - dis

        return dis
