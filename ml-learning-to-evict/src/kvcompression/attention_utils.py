#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import math
from typing import Optional

import torch


def materialize_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    query_seq_lengths: torch.Tensor,
    key_seq_lengths: torch.Tensor,
    attn_mask: torch.Tensor = None,
    is_causal: Optional[bool] = None,
) -> torch.Tensor:
    """
    Compute attention weights from query and key tensors.

    This is needed since nn.functional.scaled_dot_product_attention does not
    support returning the attention scores. Applies softmax normalization and
    handles padding/causal masking.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    Args:
        query: Query tensor [B, L, D] or [B, H, L, D].
        key: Key tensor [B, S, D] or [B, H, S, D].
        query_seq_lengths: Valid query lengths per batch item [B].
        key_seq_lengths: Valid key lengths per batch item [B].
        attn_mask: Optional attention mask. Boolean (True=attend) or additive float.
        is_causal: If True, apply causal mask. Cannot be used with attn_mask.

    Returns:
        Softmax-normalized attention weights with same leading dimensions as query,
        shape [..., L, S].
    """
    scale_factor = 1 / math.sqrt(query.size(-1))
    L, S = query.size(-2), key.size(-2)

    # Handle optional head dimension
    attn_bias_shape = list(query.shape[:-1]) + [key.shape[-2]]
    attn_bias = torch.zeros(attn_bias_shape, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None, "Cannot use causal mask and explicit mask together."
        causal_mask_values = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=query.device), diagonal=1
        )
        attn_bias.masked_fill_(causal_mask_values, float("-inf"))

    if attn_mask is not None:
        _effective_attn_mask = attn_mask

        # Handle common case: query (B,H,L,D), mask (B,L,S) -> expand mask to (B,1,L,S)
        if (
            query.ndim == 4
            and _effective_attn_mask.ndim == 3
            and _effective_attn_mask.size(0) == query.size(0)
            and _effective_attn_mask.size(1) == L
            and _effective_attn_mask.size(2) == S
        ):
            _effective_attn_mask = _effective_attn_mask[:, None, :, :]

        if _effective_attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(_effective_attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_bias + _effective_attn_mask

    if key_seq_lengths is not None:
        B = key.shape[0]  # Batch size
        max_S_key = key.size(-2)  # Padded key sequence length (S)

        assert key_seq_lengths.ndim == 1 and key_seq_lengths.size(0) == B, (
            f"key_seq_lengths must be of shape (B,), but got {key_seq_lengths.shape}"
        )

        # Create mask where True means key position is PADDED (and thus should be masked)
        key_indices = torch.arange(max_S_key, device=key.device)[None, :]
        padded_key_mask = key_indices >= key_seq_lengths[:, None]

        # Expand padded_key_mask to broadcast with attn_bias
        if (
            query.ndim == 4
        ):  # Query (B, H, Lq, D), Key (B, H, Lk, D) -> attn_bias (B,H,Lq,Lk)
            expanded_padded_key_mask = padded_key_mask[:, None, None, :]
        elif query.ndim == 3:  # Query (B, Lq, D), Key (B, Lk, D) -> attn_bias (B,Lq,Lk)
            expanded_padded_key_mask = padded_key_mask[:, None, :]
        else:
            raise ValueError(f"Unsupported query.ndim ({query.ndim}). Must be 3 or 4.")
        attn_bias.masked_fill_(expanded_padded_key_mask, float("-inf"))

    if query_seq_lengths is not None:
        B_query = query.shape[0]
        max_L_query = L

        assert query_seq_lengths.ndim == 1 and query_seq_lengths.size(0) == B_query, (
            f"query_seq_lengths must be of shape (B,), but got {query_seq_lengths.shape}"
        )

        query_indices = torch.arange(max_L_query, device=query.device)[None, :]
        padded_query_mask = query_indices >= query_seq_lengths[:, None]
        if query.ndim == 4:  # attn_bias (B,H,L,S)
            expanded_padded_query_mask = padded_query_mask[:, None, :, None]
        elif query.ndim == 3:  # attn_bias (B,L,S)
            expanded_padded_query_mask = padded_query_mask[:, :, None]
        else:
            raise ValueError(
                f"Unsupported query.ndim ({query.ndim}) for query padding."
            )
        attn_bias.masked_fill_(expanded_padded_query_mask, float("-inf"))

    attn_bias = attn_bias.to(query.dtype)

    if S == 0:  # No keys to attend to
        attn_weight = torch.full(
            attn_bias_shape, float("nan"), dtype=query.dtype, device=query.device
        )
    else:
        attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
        attn_weight = attn_weight + attn_bias
        attn_weight = attn_weight.softmax(dim=-1)

        # If any row in attn_weight became all NaNs (e.g. softmax over all -inf),
        # convert that row to zeros. This matches SDPA's behavior.
        nan_query_rows_mask = torch.isnan(attn_weight[..., 0])
        if torch.any(nan_query_rows_mask):
            expanded_nan_mask = nan_query_rows_mask.unsqueeze(-1).expand_as(attn_weight)
            attn_weight = attn_weight.masked_fill(expanded_nan_mask, 0.0)

    return attn_weight


def get_attention_mask(
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generates an attention mask suitable for prefill or other scenarios.
    The mask is boolean.
    It ensures causality for the first q_seq_len keys (if kv_seq_len >= q_seq_len)
    and masks out any keys beyond q_seq_len if kv_seq_len > q_seq_len
    (treating them as padding from the perspective of current queries).

    A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``
    """
    causal_mask_for_active_keys = torch.tril(
        torch.ones(
            (batch_size, q_seq_len, kv_seq_len),
            device=device,
            dtype=torch.bool,
        ),
    )
    return causal_mask_for_active_keys


def expand_kv_for_gqa(k_cache, v_cache, num_q_heads, num_kv_heads):
    """
    Expand key and value tensors for Grouped Query Attention.

    Replicates each KV head to match the number of query heads that share it.

    Args:
        k_cache: Key tensor of shape [B, num_kv_heads, S, head_dim].
        v_cache: Value tensor of shape [B, num_kv_heads, S, head_dim].
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads.

    Returns:
        Tuple of (k_expanded, v_expanded), each of shape [B, num_q_heads, S, head_dim].
        If num_q_heads == num_kv_heads, returns inputs unchanged.
    """
    if num_q_heads > num_kv_heads:
        q_per_kv = num_q_heads // num_kv_heads
        batch_size, _, seq_len, head_dim = k_cache.shape

        k_expanded = (
            k_cache.unsqueeze(2)
            .expand(batch_size, num_kv_heads, q_per_kv, seq_len, head_dim)
            .flatten(1, 2)
        )
        v_expanded = (
            v_cache.unsqueeze(2)
            .expand(batch_size, num_kv_heads, q_per_kv, seq_len, v_cache.shape[-1])
            .flatten(1, 2)
        )
        return k_expanded, v_expanded
    else:
        return k_cache, v_cache


def expand_k_for_gqa(k_cache, num_q_heads, num_kv_heads):
    """
    Expand key tensors for Grouped Query Attention.

    Args:
        k_cache: Key tensor of shape [B, num_kv_heads, S, head_dim].
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads.

    Returns:
        Expanded key tensor of shape [B, num_q_heads, S, head_dim].
    """
    if num_q_heads > num_kv_heads:
        q_per_kv = num_q_heads // num_kv_heads
        batch_size, _, seq_len, head_dim = k_cache.shape

        k_expanded = (
            k_cache.unsqueeze(2)
            .expand(batch_size, num_kv_heads, q_per_kv, seq_len, head_dim)
            .flatten(1, 2)
        )
        return k_expanded
    else:
        return k_cache
