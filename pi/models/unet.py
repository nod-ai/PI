import importlib
import json
import logging
import sys
from pathlib import Path, PosixPath

logger = logging.getLogger(__name__)

import functools
import operator as op
import os
from dataclasses import dataclass, fields
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
from torch import Tensor, device
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4
import inspect
import math
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import warnings


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()
        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(
                torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size
            )
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[
                :, :, offsety : offsety + size, offsetx : offsetx + size
            ]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()
        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers=4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)


class AttnProcsLayers(torch.nn.Module):
    def __init__(self, state_dict: "Dict[str, torch.Tensor]"):
        super().__init__()
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = {k: v for k, v in enumerate(state_dict.keys())}
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value
            return new_state_dict

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = key.split(".processor")[0] + ".processor"
                new_key = key.replace(
                    replace_key, f"layers.{module.rev_mapping[replace_key]}"
                )
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


def is_xformers_available():
    return _xformers_available


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other. Originally ported from here, but adapted
    to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels (`int`): The number of channels in the input and output.
        num_head_channels (`int`, *optional*):
            The number of channels in each head. If None, then `num_heads` = 1.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for group norm.
        rescale_output_factor (`float`, *optional*, defaults to 1.0): The factor to rescale the output by.
        eps (`float`, *optional*, defaults to 1e-5): The epsilon value to use for group norm.
    """

    def __init__(
        self,
        channels: "int",
        num_head_channels: "Optional[int]" = None,
        norm_num_groups: "int" = 32,
        rescale_output_factor: "float" = 1.0,
        eps: "float" = 1e-05,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = (
            channels // num_head_channels if num_head_channels is not None else 1
        )
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(
            num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True
        )
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, 1)
        self._use_memory_efficient_attention_xformers = False
        self._attention_op = None

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size
        )
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.num_heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: "bool",
        attention_op: "Optional[Callable]" = None,
    ):
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers",
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU "
                )
            else:
                try:
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e
        self._use_memory_efficient_attention_xformers = (
            use_memory_efficient_attention_xformers
        )
        self._attention_op = attention_op

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(
            1, 2
        )
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)
        scale = 1 / math.sqrt(self.channels / self.num_heads)
        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)
        if self._use_memory_efficient_attention_xformers:
            hidden_states = xformers.ops.memory_efficient_attention(
                query_proj, key_proj, value_proj, attn_bias=None, op=self._attention_op
            )
            hidden_states = hidden_states
        else:
            attention_scores = torch.baddbmm(
                torch.empty(
                    query_proj.shape[0],
                    query_proj.shape[1],
                    key_proj.shape[1],
                    dtype=query_proj.dtype,
                    device=query_proj.device,
                ),
                query_proj,
                key_proj.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )
            attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(
                attention_scores.dtype
            )
            hidden_states = torch.bmm(attention_probs, value_proj)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch, channel, height, width
        )
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class LabelEmbedding(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = torch.tensor(force_drop_ids == 1)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if self.training and use_dropout or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        time_embed_dim: "int",
        act_fn: "str" = "silu",
        out_dim: "int" = None,
        post_act_fn: "Optional[str]" = None,
        cond_proj_dim=None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        if act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"{act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'"
            )
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)
        if post_act_fn is None:
            self.post_act = None
        elif post_act_fn == "silu":
            self.post_act = nn.SiLU()
        elif post_act_fn == "mish":
            self.post_act = nn.Mish()
        elif post_act_fn == "gelu":
            self.post_act = nn.GELU()
        else:
            raise ValueError(
                f"{post_act_fn} does not exist. Make sure to define one of 'silu', 'mish', or 'gelu'"
            )

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def get_timestep_embedding(
    timesteps: "torch.Tensor",
    embedding_dim: "int",
    flip_sin_to_cos: "bool" = False,
    downscale_freq_shift: "float" = 1,
    scale: "float" = 1,
    max_period: "int" = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: "int",
        flip_sin_to_cos: "bool",
        downscale_freq_shift: "float",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.class_embedder = LabelEmbedding(
            num_classes, embedding_dim, class_dropout_prob
        )

    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)
        class_labels = self.class_embedder(class_labels)
        conditioning = timesteps_emb + class_labels
        return conditioning


class AdaLayerNormZero(nn.Module):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-06)

    def forward(self, x, timestep, class_labels, hidden_dtype=None):
        emb = self.linear(
            self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype))
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
            6, dim=1
        )
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CrossAttnAddedKVProcessor:
    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class CrossAttnProcessor:
    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        down_hidden_states = self.down(hidden_states)
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states


class LoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LoRAXFormersCrossAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attention_dim,
        rank=4,
        attention_op: "Optional[Callable]" = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SlicedAttnAddedKVProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "'CrossAttention'",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )
            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual
        return hidden_states


class SlicedAttnProcessor:
    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size
            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )
            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )
            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class XFormersCrossAttnProcessor:
    def __init__(self, attention_op: "Optional[Callable]" = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: "CrossAttention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True):
    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)
    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(
            version_name
        ):
            raise ValueError(
                f"The deprecation tuple {attribute, version_name, message} should be removed since diffusers' version {__version__} is >= {version_name}"
            )
        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."
        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=2)
    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(
            f"{function} in {filename} line {line_number - 1} got an unexpected keyword argument `{key}`"
        )
    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values


class CrossAttention(nn.Module):
    """
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: "int",
        cross_attention_dim: "Optional[int]" = None,
        heads: "int" = 8,
        dim_head: "int" = 64,
        dropout: "float" = 0.0,
        bias=False,
        upcast_attention: "bool" = False,
        upcast_softmax: "bool" = False,
        cross_attention_norm: "bool" = False,
        added_kv_proj_dim: "Optional[int]" = None,
        norm_num_groups: "Optional[int]" = None,
        out_bias: "bool" = True,
        scale_qk: "bool" = True,
        processor: "Optional['AttnProcessor']" = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.cross_attention_norm = cross_attention_norm
        self.scale = dim_head ** -0.5 if scale_qk else 1.0
        self.heads = heads
        self.sliceable_head_dim = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=inner_dim,
                num_groups=norm_num_groups,
                eps=1e-05,
                affine=True,
            )
        else:
            self.group_norm = None
        if cross_attention_norm:
            self.norm_cross = nn.LayerNorm(cross_attention_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))
        if processor is None:
            processor = (
                AttnProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention") and scale_qk
                else CrossAttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: "bool",
        attention_op: "Optional[Callable]" = None,
    ):
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor, (LoRACrossAttnProcessor, LoRAXFormersCrossAttnProcessor)
        )
        if use_memory_efficient_attention_xformers:
            if self.added_kv_proj_dim is not None:
                raise NotImplementedError(
                    "Memory efficient attention with `xformers` is currently not supported when `self.added_kv_proj_dim` is defined."
                )
            elif not is_xformers_available():
                raise ModuleNotFoundError(
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers",
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only available for GPU "
                )
            else:
                try:
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e
            if is_lora:
                processor = LoRAXFormersCrossAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor
            else:
                processor = XFormersCrossAttnProcessor(attention_op=attention_op)
        elif is_lora:
            processor = LoRACrossAttnProcessor(
                hidden_size=self.processor.hidden_size,
                cross_attention_dim=self.processor.cross_attention_dim,
                rank=self.processor.rank,
            )
            processor.load_state_dict(self.processor.state_dict())
            processor
        else:
            processor = CrossAttnProcessor()
        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(
                f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}."
            )
        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = CrossAttnAddedKVProcessor()
        else:
            processor = CrossAttnProcessor()
        self.set_processor(processor)

    def set_processor(self, processor: "'AttnProcessor'"):
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(
                f"You are removing possibly trained weights of {self.processor} with {processor}"
            )
            self._modules.pop("processor")
        self.processor = processor

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **cross_attention_kwargs,
    ):
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size
        )
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1
        attention_scores = torch.baddbmm(
            baddbmm_input, query, key.transpose(-1, -2), beta=beta, alpha=self.scale
        )
        if self.upcast_softmax:
            attention_scores = attention_scores.float()
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs
        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to `prepare_attention_mask` when preparing the attention_mask.",
            )
            batch_size = 1
        head_size = self.heads
        if attention_mask is None:
            return attention_mask
        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: "int", dim_out: "int"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class GEGLU(nn.Module):
    """
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: "int", dim_out: "int"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        return F.gelu(gate.to(dtype=torch.float32))

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class GELU(nn.Module):
    """
    GELU activation function with tanh approximation support with `approximate="tanh"`.
    """

    def __init__(self, dim_in: "int", dim_out: "int", approximate: "str" = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
    """

    def __init__(
        self,
        dim: "int",
        dim_out: "Optional[int]" = None,
        mult: "int" = 4,
        dropout: "float" = 0.0,
        activation_fn: "str" = "geglu",
        final_dropout: "bool" = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim)
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out))
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: "int",
        num_attention_heads: "int",
        attention_head_dim: "int",
        dropout=0.0,
        cross_attention_dim: "Optional[int]" = None,
        activation_fn: "str" = "geglu",
        num_embeds_ada_norm: "Optional[int]" = None,
        attention_bias: "bool" = False,
        only_cross_attention: "bool" = False,
        upcast_attention: "bool" = False,
        norm_elementwise_affine: "bool" = True,
        norm_type: "str" = "layer_norm",
        final_dropout: "bool" = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm_zero"
        )
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None and norm_type == "ada_norm"
        )
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        if cross_attention_dim is not None:
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
        else:
            self.norm2 = None
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention
            else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states
        if self.attn2 is not None:
            norm_hidden_states = (
                self.norm2(hidden_states, timestep)
                if self.use_ada_layer_norm
                else self.norm2(hidden_states)
            )
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm3(hidden_states)
        if self.use_ada_layer_norm_zero:
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )
        ff_output = self.ff(norm_hidden_states)
        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = ff_output + hidden_states
        return hidden_states


class AdaGroupNorm(nn.Module):
    """
    GroupNorm layer modified to incorporate timestep embeddings.
    """

    def __init__(
        self,
        embedding_dim: "int",
        out_dim: "int",
        num_groups: "int",
        act_fn: "Optional[str]" = None,
        eps: "float" = 1e-05,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.act = None
        if act_fn == "swish":
            self.act = lambda x: F.silu(x)
        elif act_fn == "mish":
            self.act = nn.Mish()
        elif act_fn == "silu":
            self.act = nn.SiLU()
        elif act_fn == "gelu":
            self.act = nn.GELU()
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x, emb):
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: "int",
        conditioning_channels: "int" = 3,
        block_out_channels: "Tuple[int]" = (16, 32, 96, 256),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2)
            )
        self.conv_out = zero_module(
            nn.Conv2d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding


STR_OPERATION_TO_FUNC = {
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
    "<=": op.le,
    "<": op.lt,
}


def compare_versions(
    library_or_version: "Union[str, Version]",
    operation: "str",
    requirement_version: "str",
):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(
            f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}"
        )
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_transformers_version(operation: "str", version: "str"):
    """
    Args:
    Compares the current Transformers version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _transformers_available:
        return False
    return compare_versions(parse(_transformers_version), operation, version)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))
    if (
        name
        in [
            "VersatileDiffusionTextToImagePipeline",
            "VersatileDiffusionPipeline",
            "VersatileDiffusionDualGuidedPipeline",
            "StableDiffusionImageVariationPipeline",
            "UnCLIPPipeline",
        ]
        and is_transformers_version("<", "4.25.0")
    ):
        raise ImportError(
            f"You need to install `transformers>=4.25` in order to use {name}: \n```\n pip install --upgrade transformers \n```"
        )
    if (
        name
        in [
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionPix2PixZeroPipeline",
        ]
        and is_transformers_version("<", "4.26.0")
    ):
        raise ImportError(
            f"You need to install `transformers>=4.26` in order to use {name}: \n```\n pip install --upgrade transformers \n```"
        )


class DummyObject(type):
    """
    Metaclass for the dummy objects. Any class inheriting from it will return the ImportError generated by
    `requires_backend` each time a user tries to access any method of that class.
    """

    def __getattr__(cls, key):
        if key.startswith("_"):
            return super().__getattr__(cls, key)
        requires_backends(cls, cls._backends)


class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)
        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(
                f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
            )
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(
                f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance."
            )
        super().__setitem__(name, value)


HUGGINGFACE_CO_RESOLVE_ENDPOINT = "https://huggingface.co"


def extract_commit_hash(
    resolved_file: "Optional[str]", commit_hash: "Optional[str]" = None
):
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search("snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    return commit_hash if REGEX_COMMIT_HASH.match(commit_hash) else None


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

SESSION_ID = uuid4().hex

_flax_version = "N/A"

_jax_version = "N/A"

_onnxruntime_version = "N/A"

_torch_version = "N/A"


class ConfigMixin:
    """
    Base class for all configuration classes. Stores all configuration parameters under `self.config` Also handles all
    methods for loading/downloading/saving classes inheriting from [`ConfigMixin`] with
        - [`~ConfigMixin.from_config`]
        - [`~ConfigMixin.save_config`]

    Class attributes:
        - **config_name** (`str`) -- A filename under which the config should stored when calling
          [`~ConfigMixin.save_config`] (should be overridden by parent class).
        - **ignore_for_config** (`List[str]`) -- A list of attributes that should not be saved in the config (should be
          overridden by subclass).
        - **has_compatibles** (`bool`) -- Whether the class has compatible classes (should be overridden by subclass).
        - **_deprecated_kwargs** (`List[str]`) -- Keyword arguments that are deprecated. Note that the init function
          should only have a `kwargs` argument if at least one argument is deprecated (should be overridden by
          subclass).
    """

    config_name = None
    ignore_for_config = []
    has_compatibles = False
    _deprecated_kwargs = []

    def register_to_config(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(
                f"Make sure that {self.__class__} has defined a class name `config_name`"
            )
        kwargs.pop("kwargs", None)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err
        if not hasattr(self, "_internal_dict"):
            internal_dict = kwargs
        else:
            previous_dict = dict(self._internal_dict)
            internal_dict = {**self._internal_dict, **kwargs}
            logger.debug(f"Updating config from {previous_dict} to {internal_dict}")
        self._internal_dict = FrozenDict(internal_dict)

    def save_config(
        self,
        save_directory: "Union[str, os.PathLike]",
        push_to_hub: "bool" = False,
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, self.config_name)
        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_config(
        cls,
        config: "Union[FrozenDict, Dict[str, Any]]" = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        if "pretrained_model_name_or_path" in kwargs:
            config = kwargs.pop("pretrained_model_name_or_path")
        if config is None:
            raise ValueError(
                "Please make sure to provide a config as the first positional argument."
            )
        if not isinstance(config, dict):
            deprecation_message = "It is deprecated to pass a pretrained model name or path to `from_config`."
            if "Scheduler" in cls.__name__:
                deprecation_message += f"If you were trying to load a scheduler, please use {cls}.from_pretrained(...) instead. Otherwise, please make sure to pass a configuration dictionary instead. This functionality will be removed in v1.0.0."
            elif "Model" in cls.__name__:
                deprecation_message += f"If you were trying to load a model, please use {cls}.load_config(...) followed by {cls}.from_config(...) instead. Otherwise, please make sure to pass a configuration dictionary instead. This functionality will be removed in v1.0.0."
            deprecate(
                "config-passed-as-path",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            config, kwargs = cls.load_config(
                pretrained_model_name_or_path=config,
                return_unused_kwargs=True,
                **kwargs,
            )
        init_dict, unused_kwargs, hidden_dict = cls.extract_init_dict(config, **kwargs)
        if "dtype" in unused_kwargs:
            init_dict["dtype"] = unused_kwargs.pop("dtype")
        for deprecated_kwarg in cls._deprecated_kwargs:
            if deprecated_kwarg in unused_kwargs:
                init_dict[deprecated_kwarg] = unused_kwargs.pop(deprecated_kwarg)
        model = cls(**init_dict)
        model.register_to_config(**hidden_dict)
        unused_kwargs = {**unused_kwargs, **hidden_dict}
        if return_unused_kwargs:
            return model, unused_kwargs
        else:
            return model

    @classmethod
    def get_config_dict(cls, *args, **kwargs):
        deprecation_message = f" The function get_config_dict is deprecated. Please use {cls}.load_config instead. This function will be removed in version v1.0.0"
        deprecate("get_config_dict", "1.0.0", deprecation_message, standard_warn=False)
        return cls.load_config(*args, **kwargs)

    @classmethod
    def load_config(
        cls,
        pretrained_model_name_or_path: "Union[str, os.PathLike]",
        return_unused_kwargs=False,
        return_commit_hash=False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Instantiate a Python class from a config dictionary

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a model repo on huggingface.co. Valid model ids should have an
                      organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ConfigMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.
            return_unused_kwargs (`bool`, *optional*, defaults to `False):
                Whether unused keyword arguments of the config shall be returned.
            return_commit_hash (`bool`, *optional*, defaults to `False):
                Whether the commit_hash of the loaded configuration shall be returned.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        use this method in a firewalled environment.

        </Tip>
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", None)
        user_agent = kwargs.pop("user_agent", {})
        user_agent = {**user_agent, "file_type": "config"}
        user_agent = http_user_agent(user_agent)
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from `ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )
        if os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        elif os.path.isdir(pretrained_model_name_or_path):
            if os.path.isfile(
                os.path.join(pretrained_model_name_or_path, cls.config_name)
            ):
                config_file = os.path.join(
                    pretrained_model_name_or_path, cls.config_name
                )
            elif subfolder is not None and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, cls.config_name)
            ):
                config_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, cls.config_name
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {cls.config_name} found in directory {pretrained_model_name_or_path}."
                )
        else:
            pass
        try:
            config_dict = cls._dict_from_json_file(config_file)
            commit_hash = extract_commit_hash(config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{config_file}' is not a valid JSON file."
            )
        if not (return_unused_kwargs or return_commit_hash):
            return config_dict
        outputs = (config_dict,)
        if return_unused_kwargs:
            outputs += (kwargs,)
        if return_commit_hash:
            outputs += (commit_hash,)
        return outputs

    @staticmethod
    def _get_init_keys(cls):
        return set(dict(inspect.signature(cls.__init__).parameters).keys())

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        original_dict = {k: v for k, v in config_dict.items()}
        expected_keys = cls._get_init_keys(cls)
        expected_keys.remove("self")
        if "kwargs" in expected_keys:
            expected_keys.remove("kwargs")
        if hasattr(cls, "_flax_internal_args"):
            for arg in cls._flax_internal_args:
                expected_keys.remove(arg)
        if len(cls.ignore_for_config) > 0:
            expected_keys = expected_keys - set(cls.ignore_for_config)
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        if cls.has_compatibles:
            compatible_classes = [
                c for c in cls._get_compatibles() if not isinstance(c, DummyObject)
            ]
        else:
            compatible_classes = []
        expected_keys_comp_cls = set()
        for c in compatible_classes:
            expected_keys_c = cls._get_init_keys(c)
            expected_keys_comp_cls = expected_keys_comp_cls.union(expected_keys_c)
        expected_keys_comp_cls = expected_keys_comp_cls - cls._get_init_keys(cls)
        config_dict = {
            k: v for k, v in config_dict.items() if k not in expected_keys_comp_cls
        }
        orig_cls_name = config_dict.pop("_class_name", cls.__name__)
        if orig_cls_name != cls.__name__ and hasattr(diffusers_library, orig_cls_name):
            orig_cls = getattr(diffusers_library, orig_cls_name)
            unexpected_keys_from_orig = cls._get_init_keys(orig_cls) - expected_keys
            config_dict = {
                k: v
                for k, v in config_dict.items()
                if k not in unexpected_keys_from_orig
            }
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        init_dict = {}
        for key in expected_keys:
            if key in kwargs and key in config_dict:
                config_dict[key] = kwargs.pop(key)
            if key in kwargs:
                init_dict[key] = kwargs.pop(key)
            elif key in config_dict:
                init_dict[key] = config_dict.pop(key)
        if len(config_dict) > 0:
            logger.warning(
                f"The config attributes {config_dict} were passed to {cls.__name__}, but are not expected and will be ignored. Please verify your {cls.config_name} configuration file."
            )
        passed_keys = set(init_dict.keys())
        if len(expected_keys - passed_keys) > 0:
            logger.info(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )
        unused_kwargs = {**config_dict, **kwargs}
        hidden_config_dict = {
            k: v for k, v in original_dict.items() if k not in init_dict
        }
        return init_dict, unused_kwargs, hidden_config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: "Union[str, os.PathLike]"):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @property
    def config(self) -> Dict[str, Any]:
        """
        Returns the config of the class as a frozen dictionary

        Returns:
            `Dict[str, Any]`: Config of the class.
        """
        return self._internal_dict

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, PosixPath):
                value = str(value)
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: "Union[str, os.PathLike]"):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())


class ImagePositionalEmbeddings(nn.Module):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.

    For more details, see figure 10 of the dall-e paper: https://arxiv.org/abs/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(self, num_embed: "int", height: "int", width: "int", embed_dim: "int"):
        super().__init__()
        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        emb = self.emb(index)
        height_emb = self.height_emb(
            torch.arange(self.height, device=index.device).view(1, self.height)
        )
        height_emb = height_emb.unsqueeze(2)
        width_emb = self.width_emb(
            torch.arange(self.width, device=index.device).view(1, self.width)
        )
        width_emb = width_emb.unsqueeze(1)
        pos_emb = height_emb + width_emb
        pos_emb = pos_emb.view(1, self.height * self.width, -1)
        emb = emb + pos_emb[:, : emb.shape[1], :]
        return emb


CONFIG_NAME = "config.json"

FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"

SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"

WEIGHTS_NAME = "diffusion_pytorch_model.bin"


def _add_variant(weights_name: "str", variant: "Optional[str]" = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)
    return weights_name


DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]


def _get_model_file(
    pretrained_model_name_or_path,
    *,
    weights_name,
    subfolder,
    cache_dir,
    force_download,
    proxies,
    resume_download,
    local_files_only,
    use_auth_token,
    user_agent,
    revision,
    commit_hash=None,
):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(
                pretrained_model_name_or_path, subfolder, weights_name
            )
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        if (
            revision in DEPRECATED_REVISION_ARGS
            and (
                weights_name == WEIGHTS_NAME or weights_name == SAFETENSORS_WEIGHTS_NAME
            )
            and version.parse(version.parse(__version__).base_version)
            >= version.parse("0.17.0")
        ):
            try:
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=_add_variant(weights_name, revision),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision or commit_hash,
                )
                warnings.warn(
                    f"Loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` is deprecated. Loading instead from `revision='main'` with `variant={revision}`. Loading model variants via `revision='{revision}'` will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
                    FutureWarning,
                )
                return model_file
            except:
                warnings.warn(
                    f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have a {_add_variant(weights_name, revision)} file in the 'main' branch of {pretrained_model_name_or_path}. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {_add_variant(weights_name, revision)}' so that the correct variant file can be added.",
                    FutureWarning,
                )


def _load_state_dict_into_model(model_to_load, state_dict):
    state_dict = state_dict.copy()
    error_msgs = []

    def load(module: "torch.nn.Module", prefix=""):
        args = state_dict, prefix, {}, True, [], [], error_msgs
        module._load_from_state_dict(*args)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)
    return error_msgs


def get_parameter_device(parameter: "torch.nn.Module"):
    try:
        return next(parameter.parameters()).device
    except StopIteration:

        def find_tensor_attributes(
            module: "torch.nn.Module",
        ) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: "torch.nn.Module"):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:

        def find_tensor_attributes(
            module: "torch.nn.Module",
        ) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype


def is_torch_version(operation: "str", version: "str"):
    """
    Args:
    Compares the current PyTorch version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


def load_state_dict(
    checkpoint_file: "Union[str, os.PathLike]", variant: "Optional[str]" = None
):
    """
    Reads a checkpoint file, returning properly formatted errors if they arise.
    """
    try:
        if os.path.basename(checkpoint_file) == _add_variant(WEIGHTS_NAME, variant):
            return torch.load(checkpoint_file, map_location="cpu")
        else:
            return safetensors.torch.load_file(checkpoint_file, device="cpu")
    except Exception as e:
        try:
            with open(checkpoint_file) as f:
                if f.read().startswith("version"):
                    raise OSError(
                        "You seem to have cloned a repository without having git-lfs installed. Please install git-lfs and run `git lfs install` followed by `git lfs pull` in the folder you cloned."
                    )
                else:
                    raise ValueError(
                        f"Unable to locate the file {checkpoint_file} which is necessary to load this pretrained model. Make sure you have saved the model properly."
                    ) from e
        except (UnicodeDecodeError, ValueError):
            raise OSError(
                f"Unable to load weights from checkpoint file for '{checkpoint_file}' at '{checkpoint_file}'. If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True."
            )


class ModelMixin(torch.nn.Module):
    """
    Base class for all models.

    [`ModelMixin`] takes care of storing the configuration of the models and handles methods for loading, downloading
    and saving models.

        - **config_name** ([`str`]) -- A filename under which the model should be stored when calling
          [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_diffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False

    def __init__(self):
        super().__init__()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(
            hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing
            for m in self.modules()
        )

    def enable_gradient_checkpointing(self):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing."
            )
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_memory_efficient_attention_xformers(
        self, valid: "bool", attention_op: "Optional[Callable]" = None
    ) -> None:
        def fn_recursive_set_mem_eff(module: "torch.nn.Module"):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)
            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(
        self, attention_op: "Optional[Callable]" = None
    ):
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        """
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def save_pretrained(
        self,
        save_directory: "Union[str, os.PathLike]",
        is_main_process: "bool" = True,
        save_function: "Callable" = None,
        safe_serialization: "bool" = False,
        variant: "Optional[str]" = None,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `[`~models.ModelMixin.from_pretrained`]` class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `False`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
        """
        if safe_serialization and not is_safetensors_available():
            raise ImportError(
                "`safe_serialization` requires the `safetensors library: `pip install safetensors`."
            )
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = self
        if is_main_process:
            model_to_save.save_config(save_directory)
        state_dict = model_to_save.state_dict()
        weights_name = SAFETENSORS_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)
        if safe_serialization:
            safetensors.torch.save_file(
                state_dict,
                os.path.join(save_directory, weights_name),
                metadata={"format": "pt"},
            )
        else:
            torch.save(state_dict, os.path.join(save_directory, weights_name))
        logger.info(
            f"Model weights saved in {os.path.join(save_directory, weights_name)}"
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: "Optional[Union[str, os.PathLike]]",
        **kwargs,
    ):
        """
        Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
        pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
        task.

        The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
        weights are discarded.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype. If `"auto"` is passed the dtype
                will be automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading by not initializing the weights and only loading the pre-trained weights. This
                also tries to not use more than 1x model size in CPU memory (including peak memory) while loading the
                model. This is only supported when torch version >= 1.9.0. If you are using an older version of torch,
                setting this argument to `True` will raise an error.
            variant (`str`, *optional*):
                If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
                ignored when using `from_flax`.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>

        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        device_map = kwargs.pop("device_map", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip install accelerate\n```\n."
            )
        if device_map is not None and not is_accelerate_available():
            raise NotImplementedError(
                "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set `device_map=None`. You can install accelerate with `pip install accelerate`."
            )
        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set `device_map=None`."
            )
        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set `low_cpu_mem_usage=False`."
            )
        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )
        config_path = pretrained_model_name_or_path
        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            user_agent=user_agent,
            **kwargs,
        )
        model_file = None
        if from_flax:
            model = cls.from_config(config, **unused_kwargs)
        else:
            if is_safetensors_available():
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )
                except:
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=_add_variant(WEIGHTS_NAME, variant),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )
            if low_cpu_mem_usage:
                with accelerate.init_empty_weights():
                    model = cls.from_config(config, **unused_kwargs)
                if device_map is None:
                    param_device = "cpu"
                    state_dict = load_state_dict(model_file, variant=variant)
                    missing_keys = set(model.state_dict().keys()) - set(
                        state_dict.keys()
                    )
                    if len(missing_keys) > 0:
                        raise ValueError(
                            f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are missing: \n {', '.join(missing_keys)}. \n Please make sure to pass `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize those weights or else make sure your checkpoint file is correct."
                        )
                    for param_name, param in state_dict.items():
                        accepts_dtype = "dtype" in set(
                            inspect.signature(
                                set_module_tensor_to_device
                            ).parameters.keys()
                        )
                        if accepts_dtype:
                            set_module_tensor_to_device(
                                model,
                                param_name,
                                param_device,
                                value=param,
                                dtype=torch_dtype,
                            )
                        else:
                            set_module_tensor_to_device(
                                model, param_name, param_device, value=param
                            )
                else:
                    accelerate.load_checkpoint_and_dispatch(
                        model, model_file, device_map, dtype=torch_dtype
                    )
                loading_info = {
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "mismatched_keys": [],
                    "error_msgs": [],
                }
            else:
                model = cls.from_config(config, **unused_kwargs)
                state_dict = load_state_dict(model_file, variant=variant)
                (
                    model,
                    missing_keys,
                    unexpected_keys,
                    mismatched_keys,
                    error_msgs,
                ) = cls._load_pretrained_model(
                    model,
                    state_dict,
                    model_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                )
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            model = model
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        model.eval()
        if output_loading_info:
            return model, loading_info
        return model

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
    ):
        model_state_dict = model.state_dict()
        loaded_keys = [k for k in state_dict.keys()]
        expected_keys = list(model_state_dict.keys())
        original_loaded_keys = loaded_keys
        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))
        model_to_load = model

        def _find_mismatched_keys(
            state_dict, model_state_dict, loaded_keys, ignore_mismatched_sizes
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key
                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape
                        != model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (
                                checkpoint_key,
                                state_dict[checkpoint_key].shape,
                                model_state_dict[model_key].shape,
                            )
                        )
                        del state_dict[checkpoint_key]
            return mismatched_keys

        if state_dict is not None:
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            error_msgs = _load_state_dict_into_model(model_to_load, state_dict)
        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
            raise RuntimeError(
                f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}"
            )
        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).\n- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(
                f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n"
            )
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint was trained on, you can already use {model.__class__.__name__} for predictions without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {pretrained_model_name_or_path} and are newly initialized because the shapes did not match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self) -> device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(
        self, only_trainable: "bool" = False, exclude_embeddings: "bool" = False
    ) -> int:
        """
        Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """
        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter
                for name, parameter in self.named_parameters()
                if name not in embedding_param_names
            ]
            return sum(
                p.numel()
                for p in non_embedding_parameters
                if p.requires_grad or not only_trainable
            )
        else:
            return sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad or not only_trainable
            )


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        num_patches = height // patch_size * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias=bias,
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-06)
        else:
            self.norm = None
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5))
        self.register_buffer(
            "pos_embed",
            torch.from_numpy(pos_embed).float().unsqueeze(0),
            persistent=False,
        )

    def forward(self, latent):
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)
        if self.layer_norm:
            latent = self.norm(latent)
        return latent + self.pos_embed


class BaseOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `BaseOutput` directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )
        if other_fields_are_none and isinstance(first_field, dict):
            for key, value in first_field.items():
                self[key] = value
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: "torch.FloatTensor"


def register_to_config(init):
    """
    Decorator to apply on the init of classes inheriting from [`ConfigMixin`] so that all the arguments are
    automatically sent to `self.register_for_config`. To ignore a specific argument accepted by the init but that
    shouldn't be registered in the config, use the `ignore_for_config` class variable

    Warning: Once decorated, all private arguments (beginning with an underscore) are trashed and not sent to the init!
    """

    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@register_for_config` was applied to {self.__class__.__name__} init method, but this class does not inherit from `ConfigMixin`."
            )
        ignore = getattr(self, "ignore_for_config", [])
        new_kwargs = {}
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0 and name not in ignore
        }
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        new_kwargs = {**config_init_kwargs, **new_kwargs}
        getattr(self, "register_to_config")(**new_kwargs)
        init(self, *args, **init_kwargs)

    return inner_init


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    Transformer model for image-like data. Takes either discrete (classes of vector embeddings) or continuous (actual
    embeddings) inputs.

    When input is continuous: First, project the input (aka embedding) and reshape to b, t, d. Then apply standard
    transformer action. Finally, reshape to image.

    When input is discrete: First, input (classes of latent pixels) is converted to embeddings and has positional
    embeddings applied, see `ImagePositionalEmbeddings`. Then apply standard transformer action. Finally, predict
    classes of unnoised image.

    Note that it is assumed one of the input classes is the masked latent pixel. The predicted classes of the unnoised
    image do not contain a prediction for the masked pixel as the unnoised image cannot be masked.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: "int" = 16,
        attention_head_dim: "int" = 88,
        in_channels: "Optional[int]" = None,
        out_channels: "Optional[int]" = None,
        num_layers: "int" = 1,
        dropout: "float" = 0.0,
        norm_num_groups: "int" = 32,
        cross_attention_dim: "Optional[int]" = None,
        attention_bias: "bool" = False,
        sample_size: "Optional[int]" = None,
        num_vector_embeds: "Optional[int]" = None,
        patch_size: "Optional[int]" = None,
        activation_fn: "str" = "geglu",
        num_embeds_ada_norm: "Optional[int]" = None,
        use_linear_projection: "bool" = False,
        only_cross_attention: "bool" = False,
        upcast_attention: "bool" = False,
        norm_type: "str" = "layer_norm",
        norm_elementwise_affine: "bool" = True,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.is_input_continuous = in_channels is not None and patch_size is None
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None
        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config. Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for the `transformer/config.json` file"
            deprecate(
                "norm_type!=num_embeds_ada_norm",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            norm_type = "ada_norm"
        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif (
            not self.is_input_continuous
            and not self.is_input_vectorized
            and not self.is_input_patches
        ):
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size: {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )
        if self.is_input_continuous:
            self.in_channels = in_channels
            self.norm = torch.nn.GroupNorm(
                num_groups=norm_num_groups,
                num_channels=in_channels,
                eps=1e-06,
                affine=True,
            )
            if use_linear_projection:
                self.proj_in = nn.Linear(in_channels, inner_dim)
            else:
                self.proj_in = nn.Conv2d(
                    in_channels, inner_dim, kernel_size=1, stride=1, padding=0
                )
        elif self.is_input_vectorized:
            assert (
                sample_size is not None
            ), "Transformer2DModel over discrete input must provide sample_size"
            assert (
                num_vector_embeds is not None
            ), "Transformer2DModel over discrete input must provide num_embed"
            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width
            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds,
                embed_dim=inner_dim,
                height=self.height,
                width=self.width,
            )
        elif self.is_input_patches:
            assert (
                sample_size is not None
            ), "Transformer2DModel over patched input must provide sample_size"
            self.height = sample_size
            self.width = sample_size
            self.patch_size = patch_size
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for d in range(num_layers)
            ]
        )
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            if use_linear_projection:
                self.proj_out = nn.Linear(inner_dim, in_channels)
            else:
                self.proj_out = nn.Conv2d(
                    inner_dim, in_channels, kernel_size=1, stride=1, padding=0
                )
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches:
            self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-06)
            self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Linear(
                inner_dim, patch_size * patch_size * self.out_channels
            )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict: "bool" = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Optional class labels to be applied as an embedding in AdaLayerZeroNorm. Used to indicate class labels
                conditioning.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        if self.is_input_continuous:
            batch, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            if not self.use_linear_projection:
                hidden_states = self.proj_in(hidden_states)
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
            else:
                inner_dim = hidden_states.shape[1]
                hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                    batch, height * width, inner_dim
                )
                hidden_states = self.proj_in(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )
        if self.is_input_continuous:
            if not self.use_linear_projection:
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                hidden_states = self.proj_out(hidden_states)
            else:
                hidden_states = self.proj_out(hidden_states)
                hidden_states = (
                    hidden_states.reshape(batch, height, width, inner_dim)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            logits = logits.permute(0, 2, 1)
            output = F.log_softmax(logits.double(), dim=1).float()
        elif self.is_input_patches:
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            hidden_states = (
                self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            )
            hidden_states = self.proj_out_2(hidden_states)
            height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(
                    -1,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(
                    -1,
                    self.out_channels,
                    height * self.patch_size,
                    width * self.patch_size,
                )
            )
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class DualTransformer2DModel(nn.Module):
    """
    Dual transformer wrapper that combines two `Transformer2DModel`s for mixed inference.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            Pass if the input is continuous. The number of channels in the input and output.
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of encoder_hidden_states dimensions to use.
        sample_size (`int`, *optional*): Pass if the input is discrete. The width of the latent images.
            Note that this is fixed at training time as it is used for learning a number of position embeddings. See
            `ImagePositionalEmbeddings`.
        num_vector_embeds (`int`, *optional*):
            Pass if the input is discrete. The number of classes of the vector embeddings of the latent pixels.
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*): Pass if at least one of the norm_layers is `AdaLayerNorm`.
            The number of diffusion steps used during training. Note that this is fixed at training time as it is used
            to learn a number of embeddings that are added to the hidden states. During inference, you can denoise for
            up to but not more than steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the TransformerBlocks' attention should contain a bias parameter.
    """

    def __init__(
        self,
        num_attention_heads: "int" = 16,
        attention_head_dim: "int" = 88,
        in_channels: "Optional[int]" = None,
        num_layers: "int" = 1,
        dropout: "float" = 0.0,
        norm_num_groups: "int" = 32,
        cross_attention_dim: "Optional[int]" = None,
        attention_bias: "bool" = False,
        sample_size: "Optional[int]" = None,
        num_vector_embeds: "Optional[int]" = None,
        activation_fn: "str" = "geglu",
        num_embeds_ada_norm: "Optional[int]" = None,
    ):
        super().__init__()
        self.transformers = nn.ModuleList(
            [
                Transformer2DModel(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    in_channels=in_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    sample_size=sample_size,
                    num_vector_embeds=num_vector_embeds,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                )
                for _ in range(2)
            ]
        )
        self.mix_ratio = 0.5
        self.condition_lengths = [77, 257]
        self.transformer_index_for_condition = [1, 0]

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        return_dict: "bool" = True,
    ):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continuous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            attention_mask (`torch.FloatTensor`, *optional*):
                Optional attention mask to be applied in CrossAttention
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.transformer_2d.Transformer2DModelOutput`] or `tuple`:
            [`~models.transformer_2d.Transformer2DModelOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        input_states = hidden_states
        encoded_states = []
        tokens_start = 0
        for i in range(2):
            condition_state = encoder_hidden_states[
                :, tokens_start : tokens_start + self.condition_lengths[i]
            ]
            transformer_index = self.transformer_index_for_condition[i]
            encoded_state = self.transformers[transformer_index](
                input_states,
                encoder_hidden_states=condition_state,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            encoded_states.append(encoded_state - input_states)
            tokens_start += self.condition_lengths[i]
        output_states = encoded_states[0] * self.mix_ratio + encoded_states[1] * (
            1 - self.mix_ratio
        )
        output_states = output_states + input_states
        if not return_dict:
            return (output_states,)
        return Transformer2DModelOutput(sample=output_states)


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self,
        embedding_size: "int" = 256,
        scale: "float" = 1.0,
        set_W_to_weight=True,
        log=True,
        flip_sin_to_cos=False,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(embedding_size) * scale, requires_grad=False
        )
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos
        if set_W_to_weight:
            self.W = nn.Parameter(
                torch.randn(embedding_size) * scale, requires_grad=False
            )
            self.weight = self.W

    def forward(self, x):
        if self.log:
            x = torch.log(x)
        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi
        if self.flip_sin_to_cos:
            out = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        else:
            out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    Args:
        predicted_image_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: "torch.FloatTensor"


class PriorTransformer(ModelMixin, ConfigMixin):
    """
    The prior transformer from unCLIP is used to predict CLIP image embeddings from CLIP text embeddings. Note that the
    transformer predicts the image embeddings through a denoising diffusion process.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    For more details, see the original paper: https://arxiv.org/abs/2204.06125

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the CLIP embeddings. Note that CLIP
            image embeddings and text embeddings are both the same dimension.
        num_embeddings (`int`, *optional*, defaults to 77): The max number of clip embeddings allowed. I.e. the
            length of the prompt after it has been tokenized.
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected hidden_states. The actual length of the used hidden_states is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.

    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: "int" = 32,
        attention_head_dim: "int" = 64,
        num_layers: "int" = 20,
        embedding_dim: "int" = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: "float" = 0.0,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings
        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, inner_dim)
        self.proj_in = nn.Linear(embedding_dim, inner_dim)
        self.embedding_proj = nn.Linear(embedding_dim, inner_dim)
        self.encoder_hidden_states_proj = nn.Linear(embedding_dim, inner_dim)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_embeddings + additional_embeddings, inner_dim)
        )
        self.prd_embedding = nn.Parameter(torch.zeros(1, 1, inner_dim))
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(inner_dim)
        self.proj_to_clip_embeddings = nn.Linear(inner_dim, embedding_dim)
        causal_attention_mask = torch.full(
            [
                num_embeddings + additional_embeddings,
                num_embeddings + additional_embeddings,
            ],
            -10000.0,
        )
        causal_attention_mask.triu_(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.register_buffer(
            "causal_attention_mask", causal_attention_mask, persistent=False
        )
        self.clip_mean = nn.Parameter(torch.zeros(1, embedding_dim))
        self.clip_std = nn.Parameter(torch.zeros(1, embedding_dim))

    def forward(
        self,
        hidden_states,
        timestep: "Union[torch.Tensor, float, int]",
        proj_embedding: "torch.FloatTensor",
        encoder_hidden_states: "torch.FloatTensor",
        attention_mask: "Optional[torch.BoolTensor]" = None,
        return_dict: "bool" = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                x_t, the currently predicted image embeddings.
            timestep (`torch.long`):
                Current denoising step.
            proj_embedding (`torch.FloatTensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`torch.BoolTensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.prior_transformer.PriorTransformerOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.prior_transformer.PriorTransformerOutput`] or `tuple`:
            [`~models.prior_transformer.PriorTransformerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        batch_size = hidden_states.shape[0]
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=hidden_states.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps * torch.ones(
            batch_size, dtype=timesteps.dtype, device=timesteps.device
        )
        timesteps_projected = self.time_proj(timesteps)
        timesteps_projected = timesteps_projected
        time_embeddings = self.time_embedding(timesteps_projected)
        proj_embeddings = self.embedding_proj(proj_embedding)
        encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        hidden_states = self.proj_in(hidden_states)
        prd_embedding = self.prd_embedding.expand(batch_size, -1, -1)
        positional_embeddings = self.positional_embedding
        hidden_states = torch.cat(
            [
                encoder_hidden_states,
                proj_embeddings[:, None, :],
                time_embeddings[:, None, :],
                hidden_states[:, None, :],
                prd_embedding,
            ],
            dim=1,
        )
        hidden_states = hidden_states + positional_embeddings
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -10000.0
            attention_mask = F.pad(
                attention_mask, (0, self.additional_embeddings), value=0.0
            )
            attention_mask = attention_mask[:, None, :] + self.causal_attention_mask
            attention_mask = attention_mask.repeat_interleave(
                self.config.num_attention_heads, dim=0
            )
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -1]
        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)
        if not return_dict:
            return (predicted_image_embedding,)
        return PriorTransformerOutput(
            predicted_image_embedding=predicted_image_embedding
        )

    def post_process_latents(self, prior_latents):
        prior_latents = prior_latents * self.clip_std + self.clip_mean
        return prior_latents


class Upsample1D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
            channels: channels in the inputs and outputs.
            use_conv: a bool determining if a convolution is applied.
            use_conv_transpose:
            out_channels:
    """

    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=False,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample1D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(
        self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            self.conv = nn.Conv1d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1d(kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.conv(x)


class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        use_conv_transpose:
        out_channels:
    """

    def __init__(
        self,
        channels,
        use_conv=False,
        use_conv_transpose=False,
        out_channels=None,
        name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=2.0, mode="nearest"
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )
        if dtype == torch.bfloat16:
            hidden_states = hidden_states
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states


class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(
        self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = 0, 1, 0, 1
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)
        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


def upfirdn2d_native(tensor, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]
    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape
    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]
    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


class FirUpsample2D(nn.Module):
    def __init__(
        self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight: Weight tensor of the shape `[filterH, filterW, inChannels,
                outChannels]`. Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
                (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
            factor: Integer upsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
            datatype as `hidden_states`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        kernel = kernel * (gain * factor ** 2)
        if self.use_conv:
            convH = weight.shape[2]
            convW = weight.shape[3]
            inC = weight.shape[1]
            pad_value = kernel.shape[0] - factor - (convW - 1)
            stride = factor, factor
            output_shape = (hidden_states.shape[2] - 1) * factor + convH, (
                hidden_states.shape[3] - 1
            ) * factor + convW
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convH,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // inC
            weight = torch.reshape(weight, (num_groups, -1, inC, convH, convW))
            weight = torch.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
            weight = torch.reshape(weight, (num_groups * inC, -1, convH, convW))
            inverse_conv = F.conv_transpose2d(
                hidden_states,
                weight,
                stride=stride,
                output_padding=output_padding,
                padding=0,
            )
            output = upfirdn2d_native(
                inverse_conv,
                torch.tensor(kernel, device=inverse_conv.device),
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )
        return output

    def forward(self, hidden_states):
        if self.use_conv:
            height = self._upsample_2d(
                hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel
            )
            height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)
        return height


class FirDownsample2D(nn.Module):
    def __init__(
        self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = nn.Conv2d(
                channels, out_channels, kernel_size=3, stride=1, padding=1
            )
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight:
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel: FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] *
            factor`, which corresponds to average pooling.
            factor: Integer downsampling factor (default: 2).
            gain: Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output: Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and
            same datatype as `x`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = torch.outer(kernel, kernel)
        kernel /= torch.sum(kernel)
        kernel = kernel * gain
        if self.use_conv:
            _, _, convH, convW = weight.shape
            pad_value = kernel.shape[0] - factor + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                torch.tensor(kernel, device=hidden_states.device),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
        return output

    def forward(self, hidden_states):
        if self.use_conv:
            downsample_input = self._downsample_2d(
                hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel
            )
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            hidden_states = self._downsample_2d(
                hidden_states, kernel=self.fir_kernel, factor=2
            )
        return hidden_states


class KDownsample2D(nn.Module):
    def __init__(self, pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer("kernel", kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 4, self.pad_mode)
        weight = x.new_zeros(
            [x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]]
        )
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel
        return F.conv2d(x, weight, stride=2)


class KUpsample2D(nn.Module):
    def __init__(self, pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer("kernel", kernel_1d.T @ kernel_1d, persistent=False)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = x.new_zeros(
            [x.shape[1], x.shape[1], self.kernel.shape[0], self.kernel.shape[1]]
        )
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel
        return F.conv_transpose2d(x, weight, stride=2, padding=self.pad * 2 + 1)


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor: Integer downsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H // factor, W // factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states, kernel, down=factor, pad=((pad_value + 1) // 2, pad_value // 2)
    )
    return output


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states: Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel: FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor: Integer upsampling factor (default: 2).
        gain: Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output: Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = torch.tensor(kernel, dtype=torch.float32)
    if kernel.ndim == 1:
        kernel = torch.outer(kernel, kernel)
    kernel /= torch.sum(kernel)
    kernel = kernel * (gain * factor ** 2)
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel,
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


class ResnetBlock2D(nn.Module):
    """
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernal (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-06,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        if groups_out is None:
            groups_out = groups
        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        else:
            self.norm1 = torch.nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, eps=eps, affine=True
            )
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group":
                self.time_emb_proj = None
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None
        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        else:
            self.norm2 = torch.nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
            )
        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = torch.nn.Conv2d(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1
        )
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        elif non_linearity == "gelu":
            self.nonlinearity = nn.GELU()
        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = 1, 3, 3, 1
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = 1, 3, 3, 1
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name="op"
                )
        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if self.time_emb_proj is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb
        if self.time_embedding_norm == "ada_group":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)
        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


def rearrange_dims(tensor):
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv1d = nn.Conv1d(
            inp_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        x = self.conv1d(x)
        x = rearrange_dims(x)
        x = self.group_norm(x)
        x = rearrange_dims(x)
        x = self.mish(x)
        return x


class ResidualTemporalBlock1D(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)
        self.time_emb_act = nn.Mish()
        self.time_emb = nn.Linear(embed_dim, out_channels)
        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        """
        Args:
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(x) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(x)


@dataclass
class UNet1DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, sample_size)`):
            Hidden states output. Output of last layer of model.
    """

    sample: "torch.FloatTensor"


class LinearMultiDim(nn.Linear):
    def __init__(self, in_features, out_features=None, second_dim=4, *args, **kwargs):
        in_features = (
            [in_features, second_dim, 1]
            if isinstance(in_features, int)
            else list(in_features)
        )
        if out_features is None:
            out_features = in_features
        out_features = (
            [out_features, second_dim, 1]
            if isinstance(out_features, int)
            else list(out_features)
        )
        self.in_features_multidim = in_features
        self.out_features_multidim = out_features
        super().__init__(np.array(in_features).prod(), np.array(out_features).prod())

    def forward(self, input_tensor, *args, **kwargs):
        shape = input_tensor.shape
        n_dim = len(self.in_features_multidim)
        input_tensor = input_tensor.reshape(*shape[0:-n_dim], self.in_features)
        output_tensor = super().forward(input_tensor)
        output_tensor = output_tensor.view(
            *shape[0:-n_dim], *self.out_features_multidim
        )
        return output_tensor


class ResnetBlockFlat(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-06,
        time_embedding_norm="default",
        use_in_shortcut=None,
        second_dim=4,
        **kwargs,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        in_channels = (
            [in_channels, second_dim, 1]
            if isinstance(in_channels, int)
            else list(in_channels)
        )
        self.in_channels_prod = np.array(in_channels).prod()
        self.channels_multidim = in_channels
        if out_channels is not None:
            out_channels = (
                [out_channels, second_dim, 1]
                if isinstance(out_channels, int)
                else list(out_channels)
            )
            out_channels_prod = np.array(out_channels).prod()
            self.out_channels_multidim = out_channels
        else:
            out_channels_prod = self.in_channels_prod
            self.out_channels_multidim = self.channels_multidim
        self.time_embedding_norm = time_embedding_norm
        if groups_out is None:
            groups_out = groups
        self.norm1 = torch.nn.GroupNorm(
            num_groups=groups, num_channels=self.in_channels_prod, eps=eps, affine=True
        )
        self.conv1 = torch.nn.Conv2d(
            self.in_channels_prod, out_channels_prod, kernel_size=1, padding=0
        )
        if temb_channels is not None:
            self.time_emb_proj = torch.nn.Linear(temb_channels, out_channels_prod)
        else:
            self.time_emb_proj = None
        self.norm2 = torch.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels_prod, eps=eps, affine=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels_prod, out_channels_prod, kernel_size=1, padding=0
        )
        self.nonlinearity = nn.SiLU()
        self.use_in_shortcut = (
            self.in_channels_prod != out_channels_prod
            if use_in_shortcut is None
            else use_in_shortcut
        )
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = torch.nn.Conv2d(
                self.in_channels_prod,
                out_channels_prod,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor, temb):
        shape = input_tensor.shape
        n_dim = len(self.channels_multidim)
        input_tensor = input_tensor.reshape(
            *shape[0:-n_dim], self.in_channels_prod, 1, 1
        )
        input_tensor = input_tensor.view(-1, self.in_channels_prod, 1, 1)
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = input_tensor + hidden_states
        output_tensor = output_tensor.view(*shape[0:-n_dim], -1)
        output_tensor = output_tensor.view(
            *shape[0:-n_dim], *self.out_channels_multidim
        )
        return output_tensor


class CrossAttnDownBlockFlat(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    LinearMultiDim(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class DownBlockFlat(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    LinearMultiDim(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


# def get_down_block(
#     down_block_type,
#     num_layers,
#     in_channels,
#     out_channels,
#     temb_channels,
#     add_downsample,
#     resnet_eps,
#     resnet_act_fn,
#     attn_num_head_channels,
#     resnet_groups=None,
#     cross_attention_dim=None,
#     downsample_padding=None,
#     dual_cross_attention=False,
#     use_linear_projection=False,
#     only_cross_attention=False,
#     upcast_attention=False,
#     resnet_time_scale_shift="default",
# ):
#     down_block_type = (
#         down_block_type[7:]
#         if down_block_type.startswith("UNetRes")
#         else down_block_type
#     )
#     if down_block_type == "DownBlockFlat":
#         return DownBlockFlat(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             temb_channels=temb_channels,
#             add_downsample=add_downsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             downsample_padding=downsample_padding,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#         )
#     elif down_block_type == "CrossAttnDownBlockFlat":
#         if cross_attention_dim is None:
#             raise ValueError(
#                 "cross_attention_dim must be specified for CrossAttnDownBlockFlat"
#             )
#         return CrossAttnDownBlockFlat(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             temb_channels=temb_channels,
#             add_downsample=add_downsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             downsample_padding=downsample_padding,
#             cross_attention_dim=cross_attention_dim,
#             attn_num_head_channels=attn_num_head_channels,
#             dual_cross_attention=dual_cross_attention,
#             use_linear_projection=use_linear_projection,
#             only_cross_attention=only_cross_attention,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#         )
#     raise ValueError(f"{down_block_type} is not supported.")


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "ResnetDownsampleBlock2D":
        return ResnetDownsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownBlock2D":
        return AttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnDownBlock2D"
            )
        return CrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "SimpleCrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D"
            )
        return SimpleCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "SkipDownBlock2D":
        return SkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnSkipDownBlock2D":
        return AttnSkipDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "DownEncoderBlock2D":
        return DownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "AttnDownEncoderBlock2D":
        return AttnDownEncoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "KDownBlock2D":
        return KDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif down_block_type == "KCrossAttnDownBlock2D":
        return KCrossAttnDownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            add_self_attention=True if not add_downsample else False,
        )
    raise ValueError(f"{down_block_type} does not exist.")


class MidResTemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim,
        num_layers: "int" = 1,
        add_downsample: "bool" = False,
        add_upsample: "bool" = False,
        non_linearity=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_downsample = add_downsample
        resnets = [
            ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=embed_dim)
        ]
        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(out_channels, out_channels, embed_dim=embed_dim)
            )
        self.resnets = nn.ModuleList(resnets)
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None
        self.upsample = None
        if add_upsample:
            self.upsample = Downsample1D(out_channels, use_conv=True)
        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True)
        if self.upsample and self.downsample:
            raise ValueError("Block cannot downsample and upsample")

    def forward(self, hidden_states, temb):
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)
        if self.upsample:
            hidden_states = self.upsample(hidden_states)
        if self.downsample:
            self.downsample = self.downsample(hidden_states)
        return hidden_states


_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


class Downsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states):
        hidden_states = F.pad(hidden_states, (self.pad,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]]
        )
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        weight[indices, indices] = self.kernel
        return F.conv1d(hidden_states, weight, stride=2)


class ResConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.has_conv_skip = in_channels != out_channels
        if self.has_conv_skip:
            self.conv_skip = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv_1 = nn.Conv1d(in_channels, mid_channels, 5, padding=2)
        self.group_norm_1 = nn.GroupNorm(1, mid_channels)
        self.gelu_1 = nn.GELU()
        self.conv_2 = nn.Conv1d(mid_channels, out_channels, 5, padding=2)
        if not self.is_last:
            self.group_norm_2 = nn.GroupNorm(1, out_channels)
            self.gelu_2 = nn.GELU()

    def forward(self, hidden_states):
        residual = (
            self.conv_skip(hidden_states) if self.has_conv_skip else hidden_states
        )
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.group_norm_1(hidden_states)
        hidden_states = self.gelu_1(hidden_states)
        hidden_states = self.conv_2(hidden_states)
        if not self.is_last:
            hidden_states = self.group_norm_2(hidden_states)
            hidden_states = self.gelu_2(hidden_states)
        output = hidden_states + residual
        return output


class SelfAttention1d(nn.Module):
    def __init__(self, in_channels, n_head=1, dropout_rate=0.0):
        super().__init__()
        self.channels = in_channels
        self.group_norm = nn.GroupNorm(1, num_channels=in_channels)
        self.num_heads = n_head
        self.query = nn.Linear(self.channels, self.channels)
        self.key = nn.Linear(self.channels, self.channels)
        self.value = nn.Linear(self.channels, self.channels)
        self.proj_attn = nn.Linear(self.channels, self.channels, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def transpose_for_scores(self, projection: "torch.Tensor") -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel_dim, seq = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)
        scale = 1 / math.sqrt(math.sqrt(key_states.shape[-1]))
        attention_scores = torch.matmul(
            query_states * scale, key_states.transpose(-1, -2) * scale
        )
        attention_probs = torch.softmax(attention_scores, dim=-1)
        hidden_states = torch.matmul(attention_probs, value_states)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.dropout(hidden_states)
        output = hidden_states + residual
        return output


class Upsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, hidden_states, temb=None):
        hidden_states = F.pad(hidden_states, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = hidden_states.new_zeros(
            [hidden_states.shape[1], hidden_states.shape[1], self.kernel.shape[0]]
        )
        indices = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        weight[indices, indices] = self.kernel
        return F.conv_transpose1d(
            hidden_states, weight, stride=2, padding=self.pad * 2 + 1
        )


class UNetMidBlock1D(nn.Module):
    def __init__(self, mid_channels, in_channels, out_channels=None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.up = Upsample1d(kernel="cubic")
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class ValueFunctionMidBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.res1 = ResidualTemporalBlock1D(
            in_channels, in_channels // 2, embed_dim=embed_dim
        )
        self.down1 = Downsample1D(out_channels // 2, use_conv=True)
        self.res2 = ResidualTemporalBlock1D(
            in_channels // 2, in_channels // 4, embed_dim=embed_dim
        )
        self.down2 = Downsample1D(out_channels // 4, use_conv=True)

    def forward(self, x, temb=None):
        x = self.res1(x, temb)
        x = self.down1(x)
        x = self.res2(x, temb)
        x = self.down2(x)
        return x


def get_mid_block(
    mid_block_type,
    num_layers,
    in_channels,
    mid_channels,
    out_channels,
    embed_dim,
    add_downsample,
):
    if mid_block_type == "MidResTemporalBlock1D":
        return MidResTemporalBlock1D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dim=embed_dim,
            add_downsample=add_downsample,
        )
    elif mid_block_type == "ValueFunctionMidBlock1D":
        return ValueFunctionMidBlock1D(
            in_channels=in_channels, out_channels=out_channels, embed_dim=embed_dim
        )
    elif mid_block_type == "UNetMidBlock1D":
        return UNetMidBlock1D(
            in_channels=in_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
        )
    raise ValueError(f"{mid_block_type} does not exist.")


class OutConv1DBlock(nn.Module):
    def __init__(self, num_groups_out, out_channels, embed_dim, act_fn):
        super().__init__()
        self.final_conv1d_1 = nn.Conv1d(embed_dim, embed_dim, 5, padding=2)
        self.final_conv1d_gn = nn.GroupNorm(num_groups_out, embed_dim)
        if act_fn == "silu":
            self.final_conv1d_act = nn.SiLU()
        if act_fn == "mish":
            self.final_conv1d_act = nn.Mish()
        self.final_conv1d_2 = nn.Conv1d(embed_dim, out_channels, 1)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.final_conv1d_1(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_gn(hidden_states)
        hidden_states = rearrange_dims(hidden_states)
        hidden_states = self.final_conv1d_act(hidden_states)
        hidden_states = self.final_conv1d_2(hidden_states)
        return hidden_states


class OutValueFunctionBlock(nn.Module):
    def __init__(self, fc_dim, embed_dim):
        super().__init__()
        self.final_block = nn.ModuleList(
            [
                nn.Linear(fc_dim + embed_dim, fc_dim // 2),
                nn.Mish(),
                nn.Linear(fc_dim // 2, 1),
            ]
        )

    def forward(self, hidden_states, temb):
        hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        hidden_states = torch.cat((hidden_states, temb), dim=-1)
        for layer in self.final_block:
            hidden_states = layer(hidden_states)
        return hidden_states


def get_out_block(
    *, out_block_type, num_groups_out, embed_dim, out_channels, act_fn, fc_dim
):
    if out_block_type == "OutConv1DBlock":
        return OutConv1DBlock(num_groups_out, out_channels, embed_dim, act_fn)
    elif out_block_type == "ValueFunction":
        return OutValueFunctionBlock(fc_dim, embed_dim)
    return None


class CrossAttnUpBlockFlat(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        prev_output_channel: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlockFlat(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [LinearMultiDim(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlockFlat(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlockFlat(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [LinearMultiDim(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


# def get_up_block(
#     up_block_type,
#     num_layers,
#     in_channels,
#     out_channels,
#     prev_output_channel,
#     temb_channels,
#     add_upsample,
#     resnet_eps,
#     resnet_act_fn,
#     attn_num_head_channels,
#     resnet_groups=None,
#     cross_attention_dim=None,
#     dual_cross_attention=False,
#     use_linear_projection=False,
#     only_cross_attention=False,
#     upcast_attention=False,
#     resnet_time_scale_shift="default",
# ):
#     up_block_type = (
#         up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
#     )
#     if up_block_type == "UpBlockFlat":
#         return UpBlockFlat(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             prev_output_channel=prev_output_channel,
#             temb_channels=temb_channels,
#             add_upsample=add_upsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#         )
#     elif up_block_type == "CrossAttnUpBlockFlat":
#         if cross_attention_dim is None:
#             raise ValueError(
#                 "cross_attention_dim must be specified for CrossAttnUpBlockFlat"
#             )
#         return CrossAttnUpBlockFlat(
#             num_layers=num_layers,
#             in_channels=in_channels,
#             out_channels=out_channels,
#             prev_output_channel=prev_output_channel,
#             temb_channels=temb_channels,
#             add_upsample=add_upsample,
#             resnet_eps=resnet_eps,
#             resnet_act_fn=resnet_act_fn,
#             resnet_groups=resnet_groups,
#             cross_attention_dim=cross_attention_dim,
#             attn_num_head_channels=attn_num_head_channels,
#             dual_cross_attention=dual_cross_attention,
#             use_linear_projection=use_linear_projection,
#             only_cross_attention=only_cross_attention,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#         )
#     raise ValueError(f"{up_block_type} is not supported.")
#


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "ResnetUpsampleBlock2D":
        return ResnetUpsampleBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for CrossAttnUpBlock2D"
            )
        return CrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "SimpleCrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError(
                "cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D"
            )
        return SimpleCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnUpBlock2D":
        return AttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "SkipUpBlock2D":
        return SkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnSkipUpBlock2D":
        return AttnSkipUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "UpDecoderBlock2D":
        return UpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "AttnUpDecoderBlock2D":
        return AttnUpDecoderBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attn_num_head_channels=attn_num_head_channels,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "KUpBlock2D":
        return KUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
        )
    elif up_block_type == "KCrossAttnUpBlock2D":
        return KCrossAttnUpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
        )

    raise ValueError(f"{up_block_type} does not exist.")


class UNet1DModel(ModelMixin, ConfigMixin):
    """
    UNet1DModel is a 1D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int`, *optional*): Default length of sample. Should be adaptable at runtime.
        in_channels (`int`, *optional*, defaults to 2): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 2): Number of channels in the output.
        time_embedding_type (`str`, *optional*, defaults to `"fourier"`): Type of time embedding to use.
        freq_shift (`float`, *optional*, defaults to 0.0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`False`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock1D", "DownBlock1DNoSkip", "AttnDownBlock1D")`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpBlock1D", "UpBlock1DNoSkip", "AttnUpBlock1D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(32, 32, 64)`): Tuple of block output channels.
        mid_block_type (`str`, *optional*, defaults to "UNetMidBlock1D"): block type for middle of UNet.
        out_block_type (`str`, *optional*, defaults to `None`): optional output processing of UNet.
        act_fn (`str`, *optional*, defaults to None): optional activitation function in UNet blocks.
        norm_num_groups (`int`, *optional*, defaults to 8): group norm member count in UNet blocks.
        layers_per_block (`int`, *optional*, defaults to 1): added number of layers in a UNet block.
        downsample_each_block (`int`, *optional*, defaults to False:
            experimental feature for using a UNet without upsampling.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: "int" = 65536,
        sample_rate: "Optional[int]" = None,
        in_channels: "int" = 2,
        out_channels: "int" = 2,
        extra_in_channels: "int" = 0,
        time_embedding_type: "str" = "fourier",
        flip_sin_to_cos: "bool" = True,
        use_timestep_embedding: "bool" = False,
        freq_shift: "float" = 0.0,
        down_block_types: "Tuple[str]" = (
            "DownBlock1DNoSkip",
            "DownBlock1D",
            "AttnDownBlock1D",
        ),
        up_block_types: "Tuple[str]" = (
            "AttnUpBlock1D",
            "UpBlock1D",
            "UpBlock1DNoSkip",
        ),
        mid_block_type: "Tuple[str]" = "UNetMidBlock1D",
        out_block_type: "str" = None,
        block_out_channels: "Tuple[int]" = (32, 32, 64),
        act_fn: "str" = None,
        norm_num_groups: "int" = 8,
        layers_per_block: "int" = 1,
        downsample_each_block: "bool" = False,
    ):
        super().__init__()
        self.sample_size = sample_size
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=8,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0],
                flip_sin_to_cos=flip_sin_to_cos,
                downscale_freq_shift=freq_shift,
            )
            timestep_input_dim = block_out_channels[0]
        if use_timestep_embedding:
            time_embed_dim = block_out_channels[0] * 4
            self.time_mlp = TimestepEmbedding(
                in_channels=timestep_input_dim,
                time_embed_dim=time_embed_dim,
                act_fn=act_fn,
                out_dim=block_out_channels[0],
            )
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None
        output_channel = in_channels
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            if i == 0:
                input_channel += extra_in_channels
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_downsample=not is_final_block or downsample_each_block,
            )
            self.down_blocks.append(down_block)
        self.mid_block = get_mid_block(
            mid_block_type,
            in_channels=block_out_channels[-1],
            mid_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            embed_dim=block_out_channels[0],
            num_layers=layers_per_block,
            add_downsample=downsample_each_block,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if out_block_type is None:
            final_upsample_channels = out_channels
        else:
            final_upsample_channels = block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = (
                reversed_block_out_channels[i + 1]
                if i < len(up_block_types) - 1
                else final_upsample_channels
            )
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=block_out_channels[0],
                add_upsample=not is_final_block,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        self.out_block = get_out_block(
            out_block_type=out_block_type,
            num_groups_out=num_groups_out,
            embed_dim=block_out_channels[0],
            out_channels=out_channels,
            act_fn=act_fn,
            fc_dim=block_out_channels[-1] // 4,
        )

    def forward(
        self,
        sample: "torch.FloatTensor",
        timestep: "Union[torch.Tensor, float, int]",
        return_dict: "bool" = True,
    ) -> Union[UNet1DOutput, Tuple]:
        """
        Args:
            sample (`torch.FloatTensor`): `(batch_size, num_channels, sample_size)` noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_1d.UNet1DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_1d.UNet1DOutput`] or `tuple`: [`~models.unet_1d.UNet1DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timestep_embed = self.time_proj(timesteps)
        if self.config.use_timestep_embedding:
            timestep_embed = self.time_mlp(timestep_embed)
        else:
            timestep_embed = timestep_embed[..., None]
            timestep_embed = timestep_embed.repeat([1, 1, sample.shape[2]])
            timestep_embed = timestep_embed.broadcast_to(
                sample.shape[:1] + timestep_embed.shape[1:]
            )
        down_block_res_samples = ()
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=timestep_embed
            )
            down_block_res_samples += res_samples
        if self.mid_block:
            sample = self.mid_block(sample, timestep_embed)
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = upsample_block(
                sample, res_hidden_states_tuple=res_samples, temb=timestep_embed
            )
        if self.out_block:
            sample = self.out_block(sample, timestep_embed)
        if not return_dict:
            return (sample,)
        return UNet1DOutput(sample=sample)


class DownResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        conv_shortcut=False,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_downsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.add_downsample = add_downsample
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups
        resnets = [
            ResidualTemporalBlock1D(in_channels, out_channels, embed_dim=temb_channels)
        ]
        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(
                    out_channels, out_channels, embed_dim=temb_channels
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None
        self.downsample = None
        if add_downsample:
            self.downsample = Downsample1D(out_channels, use_conv=True, padding=1)

    def forward(self, hidden_states, temb=None):
        output_states = ()
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)
        output_states += (hidden_states,)
        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states)
        return hidden_states, output_states


class UpResnetBlock1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        num_layers=1,
        temb_channels=32,
        groups=32,
        groups_out=None,
        non_linearity=None,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.add_upsample = add_upsample
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups
        resnets = [
            ResidualTemporalBlock1D(
                2 * in_channels, out_channels, embed_dim=temb_channels
            )
        ]
        for _ in range(num_layers):
            resnets.append(
                ResidualTemporalBlock1D(
                    out_channels, out_channels, embed_dim=temb_channels
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()
        else:
            self.nonlinearity = None
        self.upsample = None
        if add_upsample:
            self.upsample = Upsample1D(out_channels, use_conv_transpose=True)

    def forward(self, hidden_states, res_hidden_states_tuple=None, temb=None):
        if res_hidden_states_tuple is not None:
            res_hidden_states = res_hidden_states_tuple[-1]
            hidden_states = torch.cat((hidden_states, res_hidden_states), dim=1)
        hidden_states = self.resnets[0](hidden_states, temb)
        for resnet in self.resnets[1:]:
            hidden_states = resnet(hidden_states, temb)
        if self.nonlinearity is not None:
            hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            hidden_states = self.upsample(hidden_states)
        return hidden_states


class AttnDownBlock1D(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)
        return hidden_states, (hidden_states,)


class DownBlock1D(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        self.down = Downsample1d("cubic")
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.down(hidden_states)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states, (hidden_states,)


class DownBlock1DNoSkip(nn.Module):
    def __init__(self, out_channels, in_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        resnets = [
            ResConvBlock(in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = torch.cat([hidden_states, temb], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states, (hidden_states,)


class AttnUpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = out_channels if mid_channels is None else mid_channels
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        attentions = [
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(mid_channels, mid_channels // 32),
            SelfAttention1d(out_channels, out_channels // 32),
        ]
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states)
            hidden_states = attn(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class UpBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels),
        ]
        self.resnets = nn.ModuleList(resnets)
        self.up = Upsample1d(kernel="cubic")

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        hidden_states = self.up(hidden_states)
        return hidden_states


class UpBlock1DNoSkip(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = in_channels if mid_channels is None else mid_channels
        resnets = [
            ResConvBlock(2 * in_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, mid_channels),
            ResConvBlock(mid_channels, mid_channels, out_channels, is_last=True),
        ]
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        res_hidden_states = res_hidden_states_tuple[-1]
        hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        return hidden_states


@dataclass
class UNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: "torch.FloatTensor"


class UNetMidBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        add_attention: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    AttentionBlock(
                        in_channels,
                        num_head_channels=attn_num_head_channels,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                    )
                )
            else:
                attentions.append(None)
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class UNet2DModel(ModelMixin, ConfigMixin):
    """
    UNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to :
            obj:`True`): Whether to flip sin to cos for fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, or `"identity"`.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: "Optional[Union[int, Tuple[int, int]]]" = None,
        in_channels: "int" = 3,
        out_channels: "int" = 3,
        center_input_sample: "bool" = False,
        time_embedding_type: "str" = "positional",
        freq_shift: "int" = 0,
        flip_sin_to_cos: "bool" = True,
        down_block_types: "Tuple[str]" = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types: "Tuple[str]" = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: "Tuple[int]" = (224, 448, 672, 896),
        layers_per_block: "int" = 2,
        mid_block_scale_factor: "float" = 1,
        downsample_padding: "int" = 1,
        act_fn: "str" = "silu",
        attention_head_dim: "Optional[int]" = 8,
        norm_num_groups: "int" = 32,
        norm_eps: "float" = 1e-05,
        resnet_time_scale_shift: "str" = "default",
        add_attention: "bool" = True,
        class_embed_type: "Optional[str]" = None,
        num_class_embeds: "Optional[int]" = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1)
        )
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], scale=16
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            add_attention=add_attention,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(
        self,
        sample: "torch.FloatTensor",
        timestep: "Union[torch.Tensor, float, int]",
        class_labels: "Optional[torch.Tensor]" = None,
        return_dict: "bool" = True,
    ) -> Union[UNet2DOutput, Tuple]:
        """
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int): (batch) timesteps
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb
        emb = self.time_embedding(t_emb)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when doing class conditioning"
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb
        skip_sample = sample
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        sample = self.mid_block(sample, emb)
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(
                    sample, res_samples, emb, skip_sample
                )
            else:
                sample = upsample_block(sample, res_samples, emb)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if skip_sample is not None:
            sample += skip_sample
        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps
        if not return_dict:
            return (sample,)
        return UNet2DOutput(sample=sample)


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class UNetMidBlock2DSimpleCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.num_heads = in_channels // self.attn_num_head_channels
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                CrossAttention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=attn_num_head_channels,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    processor=CrossAttnAddedKVProcessor(),
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class AttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
            output_states += (hidden_states,)
        return hidden_states, output_states


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class AttnDownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = attn(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class AttnSkipDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=np.sqrt(2.0),
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min(in_channels // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            self.attentions.append(
                AttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                )
            )
        if add_downsample:
            self.resnet_down = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                down=True,
                kernel="fir",
            )
            self.downsamplers = nn.ModuleList(
                [FirDownsample2D(out_channels, out_channels=out_channels)]
            )
            self.skip_conv = nn.Conv2d(
                3, out_channels, kernel_size=(1, 1), stride=(1, 1)
            )
        else:
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    def forward(self, hidden_states, temb=None, skip_sample=None):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.resnet_down(hidden_states, temb)
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)
            hidden_states = self.skip_conv(skip_sample) + hidden_states
            output_states += (hidden_states,)
        return hidden_states, output_states, skip_sample


class SkipDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_pre_norm: "bool" = True,
        output_scale_factor=np.sqrt(2.0),
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min(in_channels // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        if add_downsample:
            self.resnet_down = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                down=True,
                kernel="fir",
            )
            self.downsamplers = nn.ModuleList(
                [FirDownsample2D(out_channels, out_channels=out_channels)]
            )
            self.skip_conv = nn.Conv2d(
                3, out_channels, kernel_size=(1, 1), stride=(1, 1)
            )
        else:
            self.resnet_down = None
            self.downsamplers = None
            self.skip_conv = None

    def forward(self, hidden_states, temb=None, skip_sample=None):
        output_states = ()
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            hidden_states = self.resnet_down(hidden_states, temb)
            for downsampler in self.downsamplers:
                skip_sample = downsampler(skip_sample)
            hidden_states = self.skip_conv(skip_sample) + hidden_states
            output_states += (hidden_states,)
        return hidden_states, output_states, skip_sample


class ResnetDownsampleBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb)
            output_states += (hidden_states,)
        return hidden_states, output_states


class SimpleCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_downsample=True,
    ):
        super().__init__()
        self.has_cross_attention = True
        resnets = []
        attentions = []
        self.attn_num_head_channels = attn_num_head_channels
        self.num_heads = out_channels // self.attn_num_head_channels
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                CrossAttention(
                    query_dim=out_channels,
                    cross_attention_dim=out_channels,
                    heads=self.num_heads,
                    dim_head=attn_num_head_channels,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    processor=CrossAttnAddedKVProcessor(),
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        output_states = ()
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb)
            output_states += (hidden_states,)
        return hidden_states, output_states


class KDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 4,
        resnet_eps: "float" = 1e-05,
        resnet_act_fn: "str" = "gelu",
        resnet_group_size: "int" = 32,
        add_downsample=False,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=groups,
                    groups_out=groups_out,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm="ada_group",
                    conv_shortcut_bias=False,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_downsample:
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states, output_states


class KAttentionBlock(nn.Module):
    """
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
        self,
        dim: "int",
        num_attention_heads: "int",
        attention_head_dim: "int",
        dropout: "float" = 0.0,
        cross_attention_dim: "Optional[int]" = None,
        attention_bias: "bool" = False,
        upcast_attention: "bool" = False,
        temb_channels: "int" = 768,
        add_self_attention: "bool" = False,
        cross_attention_norm: "bool" = False,
        group_size: "int" = 32,
    ):
        super().__init__()
        self.add_self_attention = add_self_attention
        if add_self_attention:
            self.norm1 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=None,
                cross_attention_norm=None,
            )
        self.norm2 = AdaGroupNorm(temb_channels, dim, max(1, dim // group_size))
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            cross_attention_norm=cross_attention_norm,
        )

    def _to_3d(self, hidden_states, height, weight):
        return hidden_states.permute(0, 2, 3, 1).reshape(
            hidden_states.shape[0], height * weight, -1
        )

    def _to_4d(self, hidden_states, height, weight):
        return hidden_states.permute(0, 2, 1).reshape(
            hidden_states.shape[0], -1, height, weight
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        emb=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        if self.add_self_attention:
            norm_hidden_states = self.norm1(hidden_states, emb)
            height, weight = norm_hidden_states.shape[2:]
            norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
            attn_output = self.attn1(
                norm_hidden_states, encoder_hidden_states=None, **cross_attention_kwargs
            )
            attn_output = self._to_4d(attn_output, height, weight)
            hidden_states = attn_output + hidden_states
        norm_hidden_states = self.norm2(hidden_states, emb)
        height, weight = norm_hidden_states.shape[2:]
        norm_hidden_states = self._to_3d(norm_hidden_states, height, weight)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )
        attn_output = self._to_4d(attn_output, height, weight)
        hidden_states = attn_output + hidden_states
        return hidden_states


class KCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        cross_attention_dim: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 4,
        resnet_group_size: "int" = 32,
        add_downsample=True,
        attn_num_head_channels: "int" = 64,
        add_self_attention: "bool" = False,
        resnet_eps: "float" = 1e-05,
        resnet_act_fn: "str" = "gelu",
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=groups,
                    groups_out=groups_out,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm="ada_group",
                    conv_shortcut_bias=False,
                )
            )
            attentions.append(
                KAttentionBlock(
                    out_channels,
                    out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    cross_attention_dim=cross_attention_dim,
                    temb_channels=temb_channels,
                    attention_bias=True,
                    add_self_attention=add_self_attention,
                    cross_attention_norm=True,
                    group_size=resnet_group_size,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_downsample:
            self.downsamplers = nn.ModuleList([KDownsample2D()])
        else:
            self.downsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        output_states = ()
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    emb=temb,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            if self.downsamplers is None:
                output_states += (None,)
            else:
                output_states += (hidden_states,)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states, output_states


class AttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        prev_output_channel: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class AttnUpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AttentionBlock(
                    out_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = attn(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class AttnSkipUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=np.sqrt(2.0),
        upsample_padding=1,
        add_upsample=True,
    ):
        super().__init__()
        self.attentions = nn.ModuleList([])
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min(resnet_in_channels + res_skip_channels // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions.append(
            AttentionBlock(
                out_channels,
                num_head_channels=attn_num_head_channels,
                rescale_output_factor=output_scale_factor,
                eps=resnet_eps,
            )
        )
        self.upsampler = FirUpsample2D(in_channels, out_channels=out_channels)
        if add_upsample:
            self.resnet_up = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                groups_out=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                up=True,
                kernel="fir",
            )
            self.skip_conv = nn.Conv2d(
                out_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.skip_norm = torch.nn.GroupNorm(
                num_groups=min(out_channels // 4, 32),
                num_channels=out_channels,
                eps=resnet_eps,
                affine=True,
            )
            self.act = nn.SiLU()
        else:
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, skip_sample=None
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states)
        if skip_sample is not None:
            skip_sample = self.upsampler(skip_sample)
        else:
            skip_sample = 0
        if self.resnet_up is not None:
            skip_sample_states = self.skip_norm(hidden_states)
            skip_sample_states = self.act(skip_sample_states)
            skip_sample_states = self.skip_conv(skip_sample_states)
            skip_sample = skip_sample + skip_sample_states
            hidden_states = self.resnet_up(hidden_states, temb)
        return hidden_states, skip_sample


class SkipUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_pre_norm: "bool" = True,
        output_scale_factor=np.sqrt(2.0),
        add_upsample=True,
        upsample_padding=1,
    ):
        super().__init__()
        self.resnets = nn.ModuleList([])
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=min((resnet_in_channels + res_skip_channels) // 4, 32),
                    groups_out=min(out_channels // 4, 32),
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.upsampler = FirUpsample2D(in_channels, out_channels=out_channels)
        if add_upsample:
            self.resnet_up = ResnetBlock2D(
                in_channels=out_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=min(out_channels // 4, 32),
                groups_out=min(out_channels // 4, 32),
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                use_in_shortcut=True,
                up=True,
                kernel="fir",
            )
            self.skip_conv = nn.Conv2d(
                out_channels, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
            )
            self.skip_norm = torch.nn.GroupNorm(
                num_groups=min(out_channels // 4, 32),
                num_channels=out_channels,
                eps=resnet_eps,
                affine=True,
            )
            self.act = nn.SiLU()
        else:
            self.resnet_up = None
            self.skip_conv = None
            self.skip_norm = None
            self.act = None

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, skip_sample=None
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
        if skip_sample is not None:
            skip_sample = self.upsampler(skip_sample)
        else:
            skip_sample = 0
        if self.resnet_up is not None:
            skip_sample_states = self.skip_norm(hidden_states)
            skip_sample_states = self.act(skip_sample_states)
            skip_sample_states = self.skip_conv(skip_sample_states)
            skip_sample = skip_sample + skip_sample_states
            hidden_states = self.resnet_up(hidden_states, temb)
        return hidden_states, skip_sample


class ResnetUpsampleBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        prev_output_channel: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None
    ):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb)
        return hidden_states


class SimpleCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        prev_output_channel: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        self.num_heads = out_channels // self.attn_num_head_channels
        for i in range(num_layers):
            res_skip_channels = in_channels if i == num_layers - 1 else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                CrossAttention(
                    query_dim=out_channels,
                    cross_attention_dim=out_channels,
                    heads=self.num_heads,
                    dim_head=attn_num_head_channels,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    processor=CrossAttnAddedKVProcessor(),
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock2D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        for resnet, attn in zip(self.resnets, self.attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb)
        return hidden_states


class KUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 5,
        resnet_eps: "float" = 1e-05,
        resnet_act_fn: "str" = "gelu",
        resnet_group_size: "Optional[int]" = 32,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        k_in_channels = 2 * out_channels
        k_out_channels = in_channels
        num_layers = num_layers - 1
        for i in range(num_layers):
            in_channels = k_in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=k_out_channels
                    if i == num_layers - 1
                    else out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=groups,
                    groups_out=groups_out,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm="ada_group",
                    conv_shortcut_bias=False,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        if add_upsample:
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None
    ):
        res_hidden_states_tuple = res_hidden_states_tuple[-1]
        if res_hidden_states_tuple is not None:
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
            else:
                hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class KCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        out_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 4,
        resnet_eps: "float" = 1e-05,
        resnet_act_fn: "str" = "gelu",
        resnet_group_size: "int" = 32,
        attn_num_head_channels=1,
        cross_attention_dim: "int" = 768,
        add_upsample: "bool" = True,
        upcast_attention: "bool" = False,
    ):
        super().__init__()
        resnets = []
        attentions = []
        is_first_block = in_channels == out_channels == temb_channels
        is_middle_block = in_channels != out_channels
        add_self_attention = True if is_first_block else False
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        k_in_channels = out_channels if is_first_block else 2 * out_channels
        k_out_channels = in_channels
        num_layers = num_layers - 1
        for i in range(num_layers):
            in_channels = k_in_channels if i == 0 else out_channels
            groups = in_channels // resnet_group_size
            groups_out = out_channels // resnet_group_size
            if is_middle_block and i == num_layers - 1:
                conv_2d_out_channels = k_out_channels
            else:
                conv_2d_out_channels = None
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_2d_out_channels=conv_2d_out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=groups,
                    groups_out=groups_out,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm="ada_group",
                    conv_shortcut_bias=False,
                )
            )
            attentions.append(
                KAttentionBlock(
                    k_out_channels if i == num_layers - 1 else out_channels,
                    k_out_channels // attn_num_head_channels
                    if i == num_layers - 1
                    else out_channels // attn_num_head_channels,
                    attn_num_head_channels,
                    cross_attention_dim=cross_attention_dim,
                    temb_channels=temb_channels,
                    attention_bias=True,
                    add_self_attention=add_self_attention,
                    cross_attention_norm=True,
                    upcast_attention=upcast_attention,
                )
            )
        self.resnets = nn.ModuleList(resnets)
        self.attentions = nn.ModuleList(attentions)
        if add_upsample:
            self.upsamplers = nn.ModuleList([KUpsample2D()])
        else:
            self.upsamplers = None
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
    ):
        res_hidden_states_tuple = res_hidden_states_tuple[-1]
        if res_hidden_states_tuple is not None:
            hidden_states = torch.cat([hidden_states, res_hidden_states_tuple], dim=1)
        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet), hidden_states, temb
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                    attention_mask,
                    cross_attention_kwargs,
                )[0]
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    emb=temb,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


AttnProcessor = Union[
    CrossAttnProcessor,
    XFormersCrossAttnProcessor,
    SlicedAttnProcessor,
    CrossAttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    LoRACrossAttnProcessor,
    LoRAXFormersCrossAttnProcessor,
]

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"

LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"


class UNet2DConditionLoadersMixin:
    def load_attn_procs(
        self,
        pretrained_model_name_or_path_or_dict: "Union[str, Dict[str, torch.Tensor]]",
        **kwargs,
    ):
        """
        Load pretrained attention processor layers into `UNet2DConditionModel`. Attention processor layers have to be
        defined in
        [cross_attention.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `torch.nn.Module` class.

        <Tip warning={true}>

            This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids should have an organization name, like `google/ddpm-celebahq-256`.
                    - A path to a *directory* containing model weights saved using [`~ModelMixin.save_config`], e.g.,
                      `./my_model_directory/`.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received files. Will attempt to resume the download if such a
                file exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `diffusers-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo (either remote in
                huggingface.co or downloaded locally), you can specify the folder name here.

            mirror (`str`, *optional*):
                Mirror source to accelerate downloads in China. If you are from China and have an accessibility
                problem, you can set this option to resolve it. Note that we do not guarantee the timeliness or safety.
                Please refer to the mirror site for more information.

        <Tip>

         It is required to be logged in (`huggingface-cli login`) when you want to use private or [gated
         models](https://huggingface.co/docs/hub/models-gated#gated-models).

        </Tip>

        <Tip>

        Activate the special ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use
        this method in a firewalled environment.

        </Tip>
        """
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}
        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if (
                is_safetensors_available()
                and weight_name is None
                or weight_name is not None
                and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except EnvironmentError:
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = torch.load(model_file, map_location="cpu")
        else:
            state_dict = pretrained_model_name_or_path_or_dict
        attn_processors = {}
        is_lora = all("lora" in k for k in state_dict.keys())
        if is_lora:
            lora_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(
                    key.split(".")[-3:]
                )
                lora_grouped_dict[attn_processor_key][sub_key] = value
            for key, value_dict in lora_grouped_dict.items():
                rank = value_dict["to_k_lora.down.weight"].shape[0]
                cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
                hidden_size = value_dict["to_k_lora.up.weight"].shape[0]
                attn_processors[key] = LoRACrossAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=rank,
                )
                attn_processors[key].load_state_dict(value_dict)
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA training."
            )
        attn_processors = {k: v for k, v in attn_processors.items()}
        self.set_attn_processor(attn_processors)

    def save_attn_procs(
        self,
        save_directory: "Union[str, os.PathLike]",
        is_main_process: "bool" = True,
        weight_name: "str" = None,
        save_function: "Callable" = None,
        safe_serialization: "bool" = False,
        **kwargs,
    ):
        """
        Save an attention processor to a directory, so that it can be re-loaded using the
        `[`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]` method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        weight_name = weight_name or deprecate(
            "weights_name",
            "0.18.0",
            "`weights_name` is deprecated, please use `weight_name` instead.",
            take_from=kwargs,
        )
        if os.path.isfile(save_directory):
            logger.error(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )
            return
        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(
                        weights, filename, metadata={"format": "pt"}
                    )

            else:
                save_function = torch.save
        os.makedirs(save_directory, exist_ok=True)
        model_to_save = AttnProcsLayers(self.attn_processors)
        state_dict = model_to_save.state_dict()
        if weight_name is None:
            if safe_serialization:
                weight_name = LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = LORA_WEIGHT_NAME
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(
            f"Model weights saved in {os.path.join(save_directory, weight_name)}"
        )


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: "torch.FloatTensor"


class UNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    """
    UNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a timestep
    and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2DCrossAttn"`):
            The mid block type. Choose from `UNetMidBlock2DCrossAttn` or `UNetMidBlock2DSimpleCrossAttn`, will skip the
            mid block layer if `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlock2D`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, or `"projection"`.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, default to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        timestep_post_act (`str, *optional*, default to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, default to `None`):
            The dimension of `cond_proj` layer in timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            using the "projection" `class_embed_type`. Required when using the "projection" `class_embed_type`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: "Optional[int]" = None,
        in_channels: "int" = 4,
        out_channels: "int" = 4,
        center_input_sample: "bool" = False,
        flip_sin_to_cos: "bool" = True,
        freq_shift: "int" = 0,
        down_block_types: "Tuple[str]" = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: "Optional[str]" = "UNetMidBlock2DCrossAttn",
        up_block_types: "Tuple[str]" = (
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        only_cross_attention: "Union[bool, Tuple[bool]]" = False,
        block_out_channels: "Tuple[int]" = (320, 640, 1280, 1280),
        layers_per_block: "int" = 2,
        downsample_padding: "int" = 1,
        mid_block_scale_factor: "float" = 1,
        act_fn: "str" = "silu",
        norm_num_groups: "Optional[int]" = 32,
        norm_eps: "float" = 1e-05,
        cross_attention_dim: "int" = 1280,
        attention_head_dim: "Union[int, Tuple[int]]" = 8,
        dual_cross_attention: "bool" = False,
        use_linear_projection: "bool" = False,
        class_embed_type: "Optional[str]" = None,
        num_class_embeds: "Optional[int]" = None,
        upcast_attention: "bool" = False,
        resnet_time_scale_shift: "str" = "default",
        time_embedding_type: "str" = "positional",
        timestep_post_act: "Optional[str]" = None,
        time_cond_proj_dim: "Optional[int]" = None,
        conv_in_kernel: "int" = 3,
        conv_out_kernel: "int" = 3,
        projection_class_embeddings_input_dim: "Optional[int]" = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(only_cross_attention, bool) and len(
            only_cross_attention
        ) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )
        if time_embedding_type == "fourier":
            time_embed_dim = block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Pleaes make sure to use one of `fourier` or `positional`."
            )
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        else:
            self.class_embedding = None
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )

    @property
    def attn_processors(self) -> Dict[str, AttnProcessor]:
        """
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}

        def fn_recursive_add_processors(
            name: "str",
            module: "torch.nn.Module",
            processors: "Dict[str, AttnProcessor]",
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(
        self, processor: "Union[AttnProcessor, Dict[str, AttnProcessor]]"
    ):
        """
        Parameters:
            `processor (`dict` of `AttnProcessor` or `AttnProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `CrossAttention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainablae attention processors.:

        """
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(
            name: "str", module: "torch.nn.Module", processor
        ):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_attention_slice(self, slice_size):
        """
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: "torch.nn.Module"):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)
        num_slicable_layers = len(sliceable_head_dims)
        if slice_size == "auto":
            slice_size = [(dim // 2) for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_slicable_layers * [1]
        slice_size = (
            num_slicable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )
        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )
        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(
            module: "torch.nn.Module", slice_size: "List[int]"
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: "torch.FloatTensor",
        timestep: "Union[torch.Tensor, float, int]",
        encoder_hidden_states: "torch.Tensor",
        class_labels: "Optional[torch.Tensor]" = None,
        timestep_cond: "Optional[torch.Tensor]" = None,
        attention_mask: "Optional[torch.Tensor]" = None,
        cross_attention_kwargs: "Optional[Dict[str, Any]]" = None,
        down_block_additional_residuals: "Optional[Tuple[torch.Tensor]]" = None,
        mid_block_additional_residual: "Optional[torch.Tensor]" = None,
        return_dict: "bool" = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)


def rename_key(key):
    regex = "\\w+[.]\\d+"
    pats = re.findall(regex, key)
    for pat in pats:
        key = key.replace(pat, "_".join(pat.split(".")))
    return key


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = torch.nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1
        )
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-06,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-06,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-06
        )
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(
            block_out_channels[-1], conv_out_channels, 3, padding=1
        )

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1
        )
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-06,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-06,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-06
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(
        self,
        n_e,
        vq_embed_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy
        self.embedding = nn.Embedding(self.n_e, self.vq_embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            None
        else:
            self.re_embed = n_e
        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used
        if self.re_embed > self.used.shape[0]:
            inds[inds >= self.used.shape[0]] = 0
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.vq_embed_dim)
        min_encoding_indices = torch.argmin(
            torch.cdist(z_flattened, self.embedding.weight), dim=1
        )
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None
        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0], -1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1, 1)
        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3]
            )
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0], -1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Decoded output sample of the model. Output of the last layer of the model.
    """

    sample: "torch.FloatTensor"


@dataclass
class VQEncoderOutput(BaseOutput):
    """
    Output of VQModel encoding method.

    Args:
        latents (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Encoded output sample of the model. Output of the last layer of the model.
    """

    latents: "torch.FloatTensor"


class VQModel(ModelMixin, ConfigMixin):
    """VQ-VAE model from the paper Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals and Koray
    Kavukcuoglu.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `3`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
        num_vq_embeddings (`int`, *optional*, defaults to `256`): Number of codebook vectors in the VQ-VAE.
        vq_embed_dim (`int`, *optional*): Hidden dim of codebook vectors in the VQ-VAE.
        scaling_factor (`float`, *optional*, defaults to `0.18215`):
            The component-wise standard deviation of the trained latent space computed using the first batch of the
            training set. This is used to scale the latent space to have unit variance when training the diffusion
            model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
            diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
            / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
            Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: "int" = 3,
        out_channels: "int" = 3,
        down_block_types: "Tuple[str]" = ("DownEncoderBlock2D",),
        up_block_types: "Tuple[str]" = ("UpDecoderBlock2D",),
        block_out_channels: "Tuple[int]" = (64,),
        layers_per_block: "int" = 1,
        act_fn: "str" = "silu",
        latent_channels: "int" = 3,
        sample_size: "int" = 32,
        num_vq_embeddings: "int" = 256,
        norm_num_groups: "int" = 32,
        vq_embed_dim: "Optional[int]" = None,
        scaling_factor: "float" = 0.18215,
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
        )
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_channels
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.quantize = VectorQuantizer(
            num_vq_embeddings,
            vq_embed_dim,
            beta=0.25,
            remap=None,
            sane_index_shape=False,
        )
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
        )

    def encode(
        self, x: "torch.FloatTensor", return_dict: "bool" = True
    ) -> VQEncoderOutput:
        h = self.encoder(x)
        h = self.quant_conv(h)
        if not return_dict:
            return (h,)
        return VQEncoderOutput(latents=h)

    def decode(
        self,
        h: "torch.FloatTensor",
        force_not_quantize: "bool" = False,
        return_dict: "bool" = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self, sample: "torch.FloatTensor", return_dict: "bool" = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Args:
            sample (`torch.FloatTensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        h = self.encode(x).latents
        dec = self.decode(h).sample
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)


class LDMBertAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: "int",
        num_heads: "int",
        head_dim: "int",
        dropout: "float" = 0.0,
        is_decoder: "bool" = False,
        bias: "bool" = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder
        self.k_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, self.inner_dim, bias=bias)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)

    def _shape(self, tensor: "torch.Tensor", seq_len: "int", bsz: "int"):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: "torch.Tensor",
        key_value_states: "Optional[torch.Tensor]" = None,
        past_key_value: "Optional[Tuple[torch.Tensor]]" = None,
        attention_mask: "Optional[torch.Tensor]" = None,
        layer_head_mask: "Optional[torch.Tensor]" = None,
        output_attentions: "bool" = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        if self.is_decoder:
            past_key_value = key_states, value_states
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}"
            )
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, tgt_len, src_len}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {self.num_heads,}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.inner_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, past_key_value


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


# ACT2CLS = {
#     "gelu": GELUActivation,
#     "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
#     "gelu_fast": FastGELUActivation,
#     "gelu_new": NewGELUActivation,
#     "gelu_python": (GELUActivation, {"use_gelu_python": True}),
#     "gelu_pytorch_tanh": PytorchGELUTanh,
#     "linear": LinearActivation,
#     "mish": MishActivation,
#     "quick_gelu": QuickGELUActivation,
#     "relu": nn.ReLU,
#     "relu6": nn.ReLU6,
#     "sigmoid": nn.Sigmoid,
#     "silu": SiLUActivation,
#     "swish": SiLUActivation,
#     "tanh": nn.Tanh,
# }
# ACT2FN = ClassInstantier(ACT2CLS)


class LDMBertEncoderLayer(nn.Module):
    def __init__(self, config: "LDMBertConfig"):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LDMBertAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            head_dim=config.head_dim,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: "torch.FloatTensor",
        attention_mask: "torch.FloatTensor",
        layer_head_mask: "torch.FloatTensor",
        output_attentions: "Optional[bool]" = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class PaintByExampleMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    hid_size,
                    num_heads,
                    hid_size,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(
        self,
        channels: "int" = 1,
        kernel_size: "int" = 3,
        sigma: "float" = 0.5,
        dim: "int" = 2,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *([1] * (kernel.dim() - 1)))
        self.register_buffer("weight", kernel)
        self.groups = channels
        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class StableUnCLIPImageNormalizer(ModelMixin, ConfigMixin):
    """
    This class is used to hold the mean and standard deviation of the CLIP embedder used in stable unCLIP.

    It is used to normalize the image embeddings before the noise is applied and un-normalize the noised image
    embeddings.
    """

    @register_to_config
    def __init__(self, embedding_dim: "int" = 768):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(1, embedding_dim))
        self.std = nn.Parameter(torch.ones(1, embedding_dim))

    def scale(self, embeds):
        embeds = (embeds - self.mean) * 1.0 / self.std
        return embeds

    def unscale(self, embeds):
        embeds = embeds * self.std + self.mean
        return embeds


class UnCLIPTextProjModel(ModelMixin, ConfigMixin):
    """
    Utility class for CLIP embeddings. Used to combine the image and text embeddings into a format usable by the
    decoder.

    For more details, see the original paper: https://arxiv.org/abs/2204.06125 section 2.1
    """

    @register_to_config
    def __init__(
        self,
        *,
        clip_extra_context_tokens: int = 4,
        clip_embeddings_dim: int = 768,
        time_embed_dim: int,
        cross_attention_dim,
    ):
        super().__init__()
        self.learned_classifier_free_guidance_embeddings = nn.Parameter(
            torch.zeros(clip_embeddings_dim)
        )
        self.embedding_proj = nn.Linear(clip_embeddings_dim, time_embed_dim)
        self.clip_image_embeddings_project_to_time_embeddings = nn.Linear(
            clip_embeddings_dim, time_embed_dim
        )
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.clip_extra_context_tokens_proj = nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.encoder_hidden_states_proj = nn.Linear(
            clip_embeddings_dim, cross_attention_dim
        )
        self.text_encoder_hidden_states_norm = nn.LayerNorm(cross_attention_dim)

    def forward(
        self,
        *,
        image_embeddings,
        prompt_embeds,
        text_encoder_hidden_states,
        do_classifier_free_guidance,
    ):
        if do_classifier_free_guidance:
            image_embeddings_batch_size = image_embeddings.shape[0]
            classifier_free_guidance_embeddings = (
                self.learned_classifier_free_guidance_embeddings.unsqueeze(0)
            )
            classifier_free_guidance_embeddings = (
                classifier_free_guidance_embeddings.expand(
                    image_embeddings_batch_size, -1
                )
            )
            image_embeddings = torch.cat(
                [classifier_free_guidance_embeddings, image_embeddings], dim=0
            )
        assert image_embeddings.shape[0] == prompt_embeds.shape[0]
        batch_size = prompt_embeds.shape[0]
        time_projected_prompt_embeds = self.embedding_proj(prompt_embeds)
        time_projected_image_embeddings = (
            self.clip_image_embeddings_project_to_time_embeddings(image_embeddings)
        )
        additive_clip_time_embeddings = (
            time_projected_image_embeddings + time_projected_prompt_embeds
        )
        clip_extra_context_tokens = self.clip_extra_context_tokens_proj(
            image_embeddings
        )
        clip_extra_context_tokens = clip_extra_context_tokens.reshape(
            batch_size, -1, self.clip_extra_context_tokens
        )
        text_encoder_hidden_states = self.encoder_hidden_states_proj(
            text_encoder_hidden_states
        )
        text_encoder_hidden_states = self.text_encoder_hidden_states_norm(
            text_encoder_hidden_states
        )
        text_encoder_hidden_states = text_encoder_hidden_states.permute(0, 2, 1)
        text_encoder_hidden_states = torch.cat(
            [clip_extra_context_tokens, text_encoder_hidden_states], dim=2
        )
        return text_encoder_hidden_states, additive_clip_time_embeddings


class UNetMidBlockFlatCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        resnets = [
            ResnetBlockFlat(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        attn_num_head_channels,
                        in_channels // attn_num_head_channels,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class UNetMidBlockFlatSimpleCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: "int",
        temb_channels: "int",
        dropout: "float" = 0.0,
        num_layers: "int" = 1,
        resnet_eps: "float" = 1e-06,
        resnet_time_scale_shift: "str" = "default",
        resnet_act_fn: "str" = "swish",
        resnet_groups: "int" = 32,
        resnet_pre_norm: "bool" = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
    ):
        super().__init__()
        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.num_heads = in_channels // self.attn_num_head_channels
        resnets = [
            ResnetBlockFlat(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                CrossAttention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=attn_num_head_channels,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    processor=CrossAttnAddedKVProcessor(),
                )
            )
            resnets.append(
                ResnetBlockFlat(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        attention_mask=None,
        cross_attention_kwargs=None,
    ):
        cross_attention_kwargs = (
            cross_attention_kwargs if cross_attention_kwargs is not None else {}
        )
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class UNetFlatConditionModel(ModelMixin, ConfigMixin):
    """
    UNetFlatConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a
    timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        flip_sin_to_cos (`bool`, *optional*, defaults to `False`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, *optional*, defaults to 0): The frequency shift to apply to the time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockFlat", "CrossAttnDownBlockFlat", "CrossAttnDownBlockFlat", "DownBlockFlat")`):
            The tuple of downsample blocks to use.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlockFlatCrossAttn"`):
            The mid block type. Choose from `UNetMidBlockFlatCrossAttn` or `UNetMidBlockFlatSimpleCrossAttn`, will skip
            the mid block layer if `None`.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockFlat", "CrossAttnUpBlockFlat", "CrossAttnUpBlockFlat", "CrossAttnUpBlockFlat",)`):
            The tuple of upsample blocks to use.
        only_cross_attention(`bool` or `Tuple[bool]`, *optional*, default to `False`):
            Whether to include self-attention in the basic transformer blocks, see
            [`~models.attention.BasicTransformerBlock`].
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        downsample_padding (`int`, *optional*, defaults to 1): The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, *optional*, defaults to 1.0): The scale factor to use for the mid block.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, it will skip the normalization and activation layers in post-processing
        norm_eps (`float`, *optional*, defaults to 1e-5): The epsilon to use for the normalization.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 8): The dimension of the attention heads.
        resnet_time_scale_shift (`str`, *optional*, defaults to `"default"`): Time scale shift config
            for resnet blocks, see [`~models.resnet.ResnetBlockFlat`]. Choose from `default` or `scale_shift`.
        class_embed_type (`str`, *optional*, defaults to None):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from `None`,
            `"timestep"`, `"identity"`, or `"projection"`.
        num_class_embeds (`int`, *optional*, defaults to None):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        time_embedding_type (`str`, *optional*, default to `positional`):
            The type of position embedding to use for timesteps. Choose from `positional` or `fourier`.
        timestep_post_act (`str, *optional*, default to `None`):
            The second activation function to use in timestep embedding. Choose from `silu`, `mish` and `gelu`.
        time_cond_proj_dim (`int`, *optional*, default to `None`):
            The dimension of `cond_proj` layer in timestep embedding.
        conv_in_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_in` layer.
        conv_out_kernel (`int`, *optional*, default to `3`): The kernel size of `conv_out` layer.
        projection_class_embeddings_input_dim (`int`, *optional*): The dimension of the `class_labels` input when
            using the "projection" `class_embed_type`. Required when using the "projection" `class_embed_type`.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: "Optional[int]" = None,
        in_channels: "int" = 4,
        out_channels: "int" = 4,
        center_input_sample: "bool" = False,
        flip_sin_to_cos: "bool" = True,
        freq_shift: "int" = 0,
        down_block_types: "Tuple[str]" = (
            "CrossAttnDownBlockFlat",
            "CrossAttnDownBlockFlat",
            "CrossAttnDownBlockFlat",
            "DownBlockFlat",
        ),
        mid_block_type: "Optional[str]" = "UNetMidBlockFlatCrossAttn",
        up_block_types: "Tuple[str]" = (
            "UpBlockFlat",
            "CrossAttnUpBlockFlat",
            "CrossAttnUpBlockFlat",
            "CrossAttnUpBlockFlat",
        ),
        only_cross_attention: "Union[bool, Tuple[bool]]" = False,
        block_out_channels: "Tuple[int]" = (320, 640, 1280, 1280),
        layers_per_block: "int" = 2,
        downsample_padding: "int" = 1,
        mid_block_scale_factor: "float" = 1,
        act_fn: "str" = "silu",
        norm_num_groups: "Optional[int]" = 32,
        norm_eps: "float" = 1e-05,
        cross_attention_dim: "int" = 1280,
        attention_head_dim: "Union[int, Tuple[int]]" = 8,
        dual_cross_attention: "bool" = False,
        use_linear_projection: "bool" = False,
        class_embed_type: "Optional[str]" = None,
        num_class_embeds: "Optional[int]" = None,
        upcast_attention: "bool" = False,
        resnet_time_scale_shift: "str" = "default",
        time_embedding_type: "str" = "positional",
        timestep_post_act: "Optional[str]" = None,
        time_cond_proj_dim: "Optional[int]" = None,
        conv_in_kernel: "int" = 3,
        conv_out_kernel: "int" = 3,
        projection_class_embeddings_input_dim: "Optional[int]" = None,
    ):
        super().__init__()
        self.sample_size = sample_size
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(only_cross_attention, bool) and len(
            only_cross_attention
        ) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )
        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(
            down_block_types
        ):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = LinearMultiDim(
            in_channels,
            block_out_channels[0],
            kernel_size=conv_in_kernel,
            padding=conv_in_padding,
        )
        if time_embedding_type == "fourier":
            time_embed_dim = block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = block_out_channels[0] * 4
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Pleaes make sure to use one of `fourier` or `positional`."
            )
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = TimestepEmbedding(
                projection_class_embeddings_input_dim, time_embed_dim
            )
        else:
            self.class_embedding = None
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)
        if mid_block_type == "UNetMidBlockFlatCrossAttn":
            self.mid_block = UNetMidBlockFlatCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        elif mid_block_type == "UNetMidBlockFlatSimpleCrossAttn":
            self.mid_block = UNetMidBlockFlatSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        self.num_upsamplers = 0
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0],
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
            self.conv_act = nn.SiLU()
        else:
            self.conv_norm_out = None
            self.conv_act = None
        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = LinearMultiDim(
            block_out_channels[0],
            out_channels,
            kernel_size=conv_out_kernel,
            padding=conv_out_padding,
        )

    @property
    def attn_processors(self) -> Dict[str, AttnProcessor]:
        """
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}

        def fn_recursive_add_processors(
            name: "str",
            module: "torch.nn.Module",
            processors: "Dict[str, AttnProcessor]",
        ):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)
        return processors

    def set_attn_processor(
        self, processor: "Union[AttnProcessor, Dict[str, AttnProcessor]]"
    ):
        """
        Parameters:
            `processor (`dict` of `AttnProcessor` or `AttnProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `CrossAttention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainablae attention processors.:

        """
        count = len(self.attn_processors.keys())
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(
            name: "str", module: "torch.nn.Module", processor
        ):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_attention_slice(self, slice_size):
        """
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: "torch.nn.Module"):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)
            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)
        num_slicable_layers = len(sliceable_head_dims)
        if slice_size == "auto":
            slice_size = [(dim // 2) for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_slicable_layers * [1]
        slice_size = (
            num_slicable_layers * [slice_size]
            if not isinstance(slice_size, list)
            else slice_size
        )
        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )
        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        def fn_recursive_set_attention_slice(
            module: "torch.nn.Module", slice_size: "List[int]"
        ):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())
            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(
            module,
            (CrossAttnDownBlockFlat, DownBlockFlat, CrossAttnUpBlockFlat, UpBlockFlat),
        ):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: "torch.FloatTensor",
        timestep: "Union[torch.Tensor, float, int]",
        encoder_hidden_states: "torch.Tensor",
        class_labels: "Optional[torch.Tensor]" = None,
        timestep_cond: "Optional[torch.Tensor]" = None,
        attention_mask: "Optional[torch.Tensor]" = None,
        cross_attention_kwargs: "Optional[Dict[str, Any]]" = None,
        down_block_additional_residuals: "Optional[Tuple[torch.Tensor]]" = None,
        mid_block_additional_residual: "Optional[torch.Tensor]" = None,
        return_dict: "bool" = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        default_overall_up_factor = 2 ** self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None
        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True
        if attention_mask is not None:
            attention_mask = (1 - attention_mask) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels should be provided when num_class_embeds > 0"
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels)
            emb = emb + class_emb
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if (
                hasattr(downsample_block, "has_cross_attention")
                and downsample_block.has_cross_attention
            ):
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = (
                    down_block_res_sample + down_block_additional_residual
                )
                new_down_block_res_samples += (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]
            if (
                hasattr(upsample_block, "has_cross_attention")
                and upsample_block.has_cross_attention
            ):
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        if not return_dict:
            return (sample,)
        return UNet2DConditionOutput(sample=sample)


class LearnedClassifierFreeSamplingEmbeddings(ModelMixin, ConfigMixin):
    """
    Utility class for storing learned text embeddings for classifier free sampling
    """

    @register_to_config
    def __init__(
        self,
        learnable: "bool",
        hidden_size: "Optional[int]" = None,
        length: "Optional[int]" = None,
    ):
        super().__init__()
        self.learnable = learnable
        if self.learnable:
            assert (
                hidden_size is not None
            ), "learnable=True requires `hidden_size` to be set"
            assert length is not None, "learnable=True requires `length` to be set"
            embeddings = torch.zeros(length, hidden_size)
        else:
            embeddings = None
        self.embeddings = torch.nn.Parameter(embeddings)
