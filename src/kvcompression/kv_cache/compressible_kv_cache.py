#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
)

logger = logging.getLogger(__name__)


class CompressibleKVCache(nn.Module):
    """
    A KVCache storing past keys, values, and associated queries.
    Assumes total unique tokens added will not exceed max_seq_len.
    Passes only the active portion of the cache to strategies.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        num_q_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        compression_strategies: Union[
            DummyCompressionStrategy, List[DummyCompressionStrategy]
        ],
    ) -> None:
        """
        Initialize the compressible KV cache.

        Args:
            batch_size: Number of sequences in the batch.
            max_seq_len: Maximum sequence length the cache can hold.
            num_kv_heads: Number of KV attention heads.
            num_q_heads: Number of query attention heads (for GQA support).
            head_dim: Dimension of each attention head.
            dtype: Data type for cache tensors.
            compression_strategies: One compression strategy per KV head.
        """
        super().__init__()

        if batch_size > 1:
            logger.error(
                "Currently a batch size greater than one is not guaranteed to work, because the different lengths/padding are not properly handled."
            )

        # NOTE: Arrives one-per head even if multihead
        if num_kv_heads != len(compression_strategies):
            raise ValueError(
                f"Provide one CompressionStrategy per KV head ({num_kv_heads}). "
                f"Got {len(compression_strategies)} strategies."
            )

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.compression_strategies = compression_strategies

        kv_cache_shape = (batch_size, num_kv_heads, max_seq_len, self.head_dim)
        q_cache_shape = (batch_size, self.num_q_heads, max_seq_len, self.head_dim)

        self.register_buffer(
            "k_cache", torch.zeros(kv_cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(kv_cache_shape, dtype=dtype), persistent=False
        )

        # Store all queries for compression strategies that need them
        self.register_buffer(
            "q_cache", torch.zeros(q_cache_shape, dtype=dtype), persistent=False
        )

        self.register_buffer(
            "_current_seq_len", torch.tensor(0, dtype=torch.long), persistent=False
        )

        initial_mask_shape = (batch_size, num_kv_heads, max_seq_len)

        # True: allowed, False: masked
        # Initially all masked
        self.register_buffer(
            "attention_mask",
            torch.zeros(initial_mask_shape, dtype=torch.bool),
            persistent=False,
        )

    def reset(self) -> None:
        """Reset the cache to its initial empty state."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.q_cache.zero_()
        self.attention_mask.fill_(False)  # Reset to False (masked) rather than zero
        self._current_seq_len.zero_()

    def size(self) -> int:
        """Return the current sequence length stored in the cache."""
        return self._current_seq_len

    @property
    def num_active_entries(self) -> int:
        """Return the total number of active (unmasked) entries across all heads."""
        return torch.sum(self.attention_mask).item()

    def update(
        self,
        k_val: torch.Tensor,
        v_val: torch.Tensor,
        q_val: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add new key, value, and optionally query tensors to the cache.

        Args:
            k_val: Key tensor of shape [B, num_kv_heads, S_new, head_dim].
            v_val: Value tensor of shape [B, num_kv_heads, S_new, head_dim].
            q_val: Optional query tensor of shape [B, num_q_heads, S_new, head_dim].

        Returns:
            Tuple of (k_cache, v_cache) tensors.
        """
        s_new = k_val.shape[2]
        if s_new == 0:
            return self.k_cache, self.v_cache

        current_len = self._current_seq_len
        if current_len + s_new > self.max_seq_len:
            raise ValueError(
                f"Cannot add {s_new} tokens. Cache current size {current_len} "
                f"and max_seq_len {self.max_seq_len} would be exceeded."
            )

        write_slice = slice(current_len, current_len + s_new)
        self.k_cache[:, :, write_slice, :] = k_val
        self.v_cache[:, :, write_slice, :] = v_val

        if q_val is not None:
            # Store all queries - q_val should have shape [B, num_q_heads, S_new, D]
            if q_val.shape[1] == self.num_q_heads and q_val.shape[2] == s_new:
                self.q_cache[:, :, write_slice, :] = q_val
            else:
                raise ValueError(
                    f"Query shape {q_val.shape} doesn't match expected shape "
                    f"[{k_val.shape[0]}, {self.num_q_heads}, {s_new}, {self.head_dim}], skipping query storage"
                )

        self.attention_mask[:, :, write_slice] = True
        self._current_seq_len.add_(s_new)

        return self.k_cache, self.v_cache

    def compress(
        self, target_size_per_head: int, **strategy_kwargs: Any
    ) -> torch.Tensor:
        """
        Apply compression strategies to reduce active cache entries.

        Args:
            target_size_per_head: Target number of entries to keep per KV head.
            **strategy_kwargs: Additional arguments passed to compression strategies.

        Returns:
            Updated attention mask tensor of shape [B, num_kv_heads, max_seq_len].
        """
        current_actual_len = self._current_seq_len

        if not (0 <= target_size_per_head <= self.max_seq_len):
            raise ValueError(
                f"target_size_per_head ({target_size_per_head}) "
                f"must be between 0 and max_seq_len ({self.max_seq_len})."
            )

        actual_target_size_for_strategy = min(target_size_per_head, current_actual_len)

        if self.num_kv_heads == 0 or current_actual_len == 0:
            self.attention_mask.zero_()
            return self.attention_mask

        if actual_target_size_for_strategy == 0:
            self.attention_mask.zero_()
            return self.attention_mask

        active_slice = slice(0, current_actual_len)
        k_cache_active = self.k_cache[:, :, active_slice, :]
        v_cache_active = self.v_cache[:, :, active_slice, :]
        q_cache_active = self.q_cache[
            :, :, active_slice, :
        ]  # All queries [B, num_q_heads, S, D]

        active_mask_parts: Optional[torch.Tensor] = None

        head_masks_list: List[torch.Tensor] = []
        for i in range(self.num_kv_heads):
            strategy = self.compression_strategies[i]

            k_head_active = k_cache_active[:, [i], :, :]
            v_head_active = v_cache_active[:, [i], :, :]

            # For GQA, provide all queries associated with this KV head
            # Each KV head corresponds to multiple query heads
            if self.num_q_heads >= self.num_kv_heads:
                q_per_kv = self.num_q_heads // self.num_kv_heads
                q_start_idx = i * q_per_kv
                q_end_idx = min(
                    q_start_idx + q_per_kv, self.num_q_heads
                )  # Ensure we don't exceed bounds
                q_head_active = q_cache_active[
                    :, q_start_idx:q_end_idx, :, :
                ]  # [B, q_per_kv, S, D]
            else:
                # Handle case where num_q_heads < num_kv_heads (unusual but possible)
                q_head_active = q_cache_active[
                    :, [i % self.num_q_heads], :, :
                ]  # [B, 1, S, D]

            mask_for_head_active = strategy.get_mask(
                k_cache_active=k_head_active,
                v_cache_active=v_head_active,
                q_assoc_cache_active=q_head_active,
                target_compressed_size=actual_target_size_for_strategy,
                **strategy_kwargs,
            )
            expected_shape = (self.batch_size, 1, current_actual_len)
            if mask_for_head_active.shape != expected_shape:
                raise ValueError(
                    f"Strategy for head {i} returned mask shape {mask_for_head_active.shape}, expected {expected_shape}"
                )
            head_masks_list.append(mask_for_head_active)

        active_mask_parts = torch.cat(head_masks_list, dim=1)

        if active_mask_parts.dtype != torch.bool:
            raise ValueError(
                f"Strategy returned mask with dtype {active_mask_parts.dtype}, expected torch.bool."
            )

        # Apply the computed mask for the active region
        self.attention_mask[:, :, :current_actual_len] = active_mask_parts

        return self.attention_mask

    def get_k_v_and_mask_for_attention(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns K, V caches and the attention mask, sliced to the current
        actual sequence length if it's less than max_seq_len.
        """
        k_out = self.k_cache
        v_out = self.v_cache
        mask_out = self.attention_mask
        return k_out, v_out, mask_out

    def get_queries_for_kv_head(self, kv_head_idx: int) -> torch.Tensor:
        """
        Get all queries associated with a specific KV head for GQA.

        Args:
            kv_head_idx: Index of the KV head

        Returns:
            Query tensor [B, q_per_kv, current_seq_len, D] for the specified KV head
        """
        if kv_head_idx >= self.num_kv_heads:
            raise ValueError(
                f"KV head index {kv_head_idx} >= num_kv_heads {self.num_kv_heads}"
            )

        q_per_kv = self.num_q_heads // self.num_kv_heads
        q_start_idx = kv_head_idx * q_per_kv
        q_end_idx = q_start_idx + q_per_kv

        current_len = self._current_seq_len
        return self.q_cache[:, q_start_idx:q_end_idx, :current_len, :]

    def get_all_queries_active(self) -> torch.Tensor:
        """
        Get all active queries.

        Returns:
            Query tensor [B, num_q_heads, current_seq_len, D]
        """
        current_len = self._current_seq_len
        return self.q_cache[:, :, :current_len, :]
