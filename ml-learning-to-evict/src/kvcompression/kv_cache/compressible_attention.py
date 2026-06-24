#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

if TYPE_CHECKING:
    from kvcompression.hooks.compressor import LayerHeadCompressionConfig
from kvcompression.attention_utils import expand_kv_for_gqa
from kvcompression.kv_cache.compressible_kv_cache import CompressibleKVCache
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
)

logger = logging.getLogger(__name__)


# Compile flex_attention with dynamic=True for long context support
# This allows variable sequence lengths while keeping compression strategies uncompiled
flex_attention_compiled = torch.compile(flex_attention, dynamic=True)

# Also compile create_block_mask with dynamic=True to handle memory efficiently
# Without compilation, it creates huge dense tensors that cause OOM
create_block_mask_compiled = torch.compile(create_block_mask, dynamic=True)


def monkey_patched_setup_cache(
    self,
    batch_size: int,
    dtype: torch.dtype,
    max_seq_len: int,
    layer_head_configs: Optional[List] = None,
) -> None:
    """Setup key value caches for attention calculation with layer/head specific compression.

    Args:
        batch_size (int): batch size for the caches.
        dtype (torch.dtype): dtype for the caches.
        max_seq_len (int): maximum sequence length model will be run with.
        layer_head_configs (List): List of LayerHeadCompressionConfig objects.
    """
    # _layer_idx is injected by registrar
    layer_idx = self._layer_idx

    head_to_strategy_map = _build_head_strategy_mapping(
        layer_idx=layer_idx,
        num_kv_heads=self.num_kv_heads,
        layer_head_configs=layer_head_configs or [],
    )

    strategies = list(head_to_strategy_map.values())

    self.kv_cache = CompressibleKVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_kv_heads=self.num_kv_heads,
        num_q_heads=self.num_heads,
        head_dim=self.head_dim,
        dtype=dtype,
        compression_strategies=strategies,
    )

    self.cache_enabled = True


def _build_head_strategy_mapping(
    layer_idx: int,
    num_kv_heads: int,
    layer_head_configs: List[LayerHeadCompressionConfig],
) -> Dict[int, DummyCompressionStrategy]:
    """
    Build a mapping from head indices to compression strategies for a specific layer.

    Args:
        layer_idx: The layer index
        num_kv_heads: Number of KV heads in this layer
        layer_head_configs: List of LayerHeadCompressionConfig objects

    Returns:
        Dict mapping head_idx -> compression_strategy
    """
    head_to_strategy = {}

    # For each head, find the most specific matching configuration
    for head_idx in range(num_kv_heads):
        matching_configs = [
            config
            for config in layer_head_configs
            if config.matches(layer_idx, head_idx)  # Handles None properly
        ]

        if matching_configs:
            matching_configs.sort(key=lambda c: c.specificity(), reverse=True)
            chosen_config = matching_configs[0]
            head_to_strategy[head_idx] = chosen_config.strategy
            logger.debug(
                f"Assigned {chosen_config.strategy_name} to layer {layer_idx}, head {head_idx}"
            )
        else:
            # No matching config found - this should only happen if no wildcard configs exist
            # In most cases, there should be at least one config with layer=None, head=None
            logger.debug(
                f"No config found for layer {layer_idx}, head {head_idx}, using dummy strategy"
            )
            head_to_strategy[head_idx] = DummyCompressionStrategy()

    # Validate that all heads have a strategy assigned
    if len(head_to_strategy) != num_kv_heads:
        raise ValueError(
            f"Strategy mapping incomplete for layer {layer_idx}: got {len(head_to_strategy)} strategies for {num_kv_heads} heads"
        )

    return head_to_strategy


def monkey_patched_mha_forward(
    self,
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    *,
    mask: Optional[torch.Tensor] = None,
    input_pos: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Forward pass for multi-head attention with KV cache compression support.

    This monkey-patched forward function replaces the standard MHA forward to
    integrate with CompressibleKVCache. It handles prefill and generation phases,
    applying compression masks via flex_attention.

    Args:
        x: Input tensor of shape [B, S_x, embed_dim].
        y: Optional cross-attention input. If provided, keys/values are computed
           from y (prefill). If None, only queries come from x (generation).
        mask: Optional attention mask. True means attend, False means mask.
        input_pos: Optional position indices for rotary embeddings.

    Returns:
        Output tensor of shape [B, S_x, embed_dim].
    """
    b, s_x, _ = x.shape
    s_y = y.shape[1] if y is not None else 0

    q = self.q_proj(x)
    q_per_kv = self.num_heads // self.num_kv_heads
    q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

    if self.pos_embeddings is not None:
        q = self.pos_embeddings(q, input_pos=input_pos)
    q = q.transpose(1, 2)

    if self.q_norm is not None:
        q = self.q_norm(q)

    if y is not None:  # Prefill or cross-attention
        k = self.k_proj(y)
        v = self.v_proj(y)
        k = k.view(b, s_y, self.num_kv_heads, self.head_dim)
        v = v.view(b, s_y, self.num_kv_heads, self.head_dim)

        if self.pos_embeddings is not None:
            k = self.pos_embeddings(k, input_pos=input_pos if s_x == s_y else None)

        k = k.transpose(1, 2)  # [B, num_kv_heads, S_y, head_dim]
        v = v.transpose(1, 2)  # [B, num_kv_heads, S_y, head_dim]

        if self.k_norm is not None:
            k = self.k_norm(k)

        if self.kv_cache is not None and self.cache_enabled:
            try:
                # Store all queries so presses have access to complete query information
                q_for_cache = q  # [B, num_heads, S_x, head_dim]

                self.kv_cache.update(k, v, q_for_cache)
            except Exception as e:
                logger.warning(
                    f"Failed to store queries in cache: {e}. Skipping query storage."
                )
                self.kv_cache.update(k, v)

    k_from_cache, v_from_cache, mask_from_cache_active = (
        self.kv_cache.get_k_v_and_mask_for_attention()
    )

    k_for_attention, v_for_attention = expand_kv_for_gqa(
        k_from_cache,
        v_from_cache,
        num_q_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
    )

    # flex_attention mask: mask_mod(b_idx, h_idx, q_idx, kv_idx) -> True if attend

    _mask_from_cache_active_c = mask_from_cache_active
    _q_per_kv_c = q_per_kv
    _num_heads_c = self.num_heads
    _num_kv_heads_c = self.num_kv_heads

    def kv_cache_compression_mask_fn(b_idx, h_idx, q_idx, kv_idx):
        # Map query head index to KV head index for GQA
        kv_h_idx = h_idx // _q_per_kv_c if _num_heads_c != _num_kv_heads_c else h_idx
        return _mask_from_cache_active_c[b_idx, kv_h_idx, kv_idx]

    current_score_mod_fn = kv_cache_compression_mask_fn

    apply_dynamic_causal_mask = self.is_causal and (y is None)

    if apply_dynamic_causal_mask:
        _input_pos_c = input_pos

        def causal_mask_fn(b_idx, h_idx, q_idx, kv_idx):
            query_absolute_pos = q_idx
            if _input_pos_c is not None:
                query_absolute_pos = _input_pos_c[b_idx, q_idx]
            return query_absolute_pos >= kv_idx

        current_score_mod_fn = and_masks(causal_mask_fn, current_score_mod_fn)

    if mask is not None:
        # True means attend.
        _external_mask_c = mask
        if _external_mask_c.shape[0] == 1 and q.shape[0] > 1:
            _external_mask_c = _external_mask_c.expand(q.shape[0], -1, -1)

        if _external_mask_c.shape[1] == 1 and q.shape[2] > 1:
            _external_mask_c = _external_mask_c.expand(-1, q.shape[2], -1)
        _final_expanded_external_mask_c = _external_mask_c

        def external_mask_fn(b_idx, h_idx, q_idx, kv_idx):
            if _final_expanded_external_mask_c.ndim == 3:
                return _final_expanded_external_mask_c[b_idx, q_idx, kv_idx]
            else:
                return _final_expanded_external_mask_c[b_idx, h_idx, q_idx, kv_idx]

        current_score_mod_fn = and_masks(external_mask_fn, current_score_mod_fn)

    block_mask = create_block_mask_compiled(
        current_score_mod_fn,
        q.shape[0],  # B
        q.shape[1],  # num_heads
        q.shape[2],  # S_q
        k_for_attention.shape[2],  # S_kv (s_cache_active_len)
        device=q.device,
    )

    attn_output = flex_attention_compiled(
        query=q,  # [B, num_heads, S_x, head_dim]
        key=k_for_attention,  # [B, num_heads, s_cache_active_len, head_dim]
        value=v_for_attention,  # [B, num_heads, s_cache_active_len, head_dim]
        block_mask=block_mask,
    )

    # Apply dropout if training (flex_attention doesn't have dropout_p)
    if self.attn_dropout > 0 and self.training:
        attn_output = torch.nn.functional.dropout(attn_output, p=self.attn_dropout)

    attn_output = attn_output.transpose(1, 2).contiguous().view(b, s_x, -1)
    return self.output_proj(attn_output)
