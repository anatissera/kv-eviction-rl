#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import abc
from typing import Any, Optional

import torch

from kvcompression.attention_utils import (
    expand_k_for_gqa,
    materialize_attention_weights,
)
from kvcompression.presses.base_press import BasePress


class CompressionStrategy(abc.ABC):
    """
    Protocol for a compression strategy.
    Decides which entries in the currently active KV cache should be kept.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def get_mask(
        self,
        k_cache_active: torch.Tensor,
        v_cache_active: torch.Tensor,
        q_assoc_cache_active: torch.Tensor,
        target_compressed_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generates a boolean mask for the provided active cache entries.

        Args:
            k_cache_active: The portion of the key cache that currently holds active tokens.
                            Shape: [B, H_kv_or_1, current_actual_seq_len, D_kv].
            v_cache_active: The portion of the value cache with active tokens.
                            Shape: [B, H_kv_or_1, current_actual_seq_len, D_kv].
            q_assoc_cache_active: The portion of the associated query cache with active tokens.
                                    Shape: [B, H_kv_or_1, current_actual_seq_len, D_q].
            target_compressed_size: The desired number of active tokens per head, selected
                                    from the `current_actual_seq_len` tokens provided.
            **kwargs: Additional contextual information.

        Returns:
            torch.Tensor: A boolean mask of shape corresponding to the input active caches.
                            with shape: `[B, H_kv, current_actual_seq_len]`.
                            `True` indicates the token at that position (within the active slice)
                            should be kept.
        """
        raise NotImplementedError()


class DummyCompressionStrategy(CompressionStrategy):
    """A no-op compression strategy that keeps all entries."""

    def get_mask(
        self,
        k_cache_active: torch.Tensor,
        v_cache_active: torch.Tensor,
        q_assoc_cache_active: torch.Tensor,
        target_compressed_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        # All True, do not mask anything out.
        return torch.ones(
            k_cache_active.shape[:-1], dtype=torch.bool, device=k_cache_active.device
        )


class RandomCompressionStrategy(CompressionStrategy):
    """A random compression strategy that keeps entries with 50% probability."""

    def get_mask(
        self,
        k_cache_active: torch.Tensor,
        v_cache_active: torch.Tensor,
        q_assoc_cache_active: torch.Tensor,
        target_compressed_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        return (
            torch.rand(
                k_cache_active.shape[:-1],
                device=k_cache_active.device,
            )
            > 0.5
        )


class PressCompressionStrategy(CompressionStrategy):
    """Compression strategy that delegates to a BasePress for scoring."""

    def __init__(self, press: BasePress):
        self.press = press

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(press={repr(self.press)})"

    def get_mask(
        self,
        k_cache_active: torch.Tensor,
        v_cache_active: torch.Tensor,
        target_compressed_size: int,
        q_assoc_cache_active: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # Re-materialize attention weights if the press requires them but they're not provided
        if self.press.requires_attention_weights() and attention_weights is None:
            if q_assoc_cache_active is not None:
                # For GQA, we need to expand queries to match attention head count
                batch_size, num_kv_heads, seq_len, head_dim = k_cache_active.shape
                _, num_q_heads, _, _ = q_assoc_cache_active.shape
                queries_for_attn = q_assoc_cache_active

                k_expanded = expand_k_for_gqa(
                    k_cache_active, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
                )

                # NOTE: assumes batch_size = 1
                seq_lengths = torch.full(
                    (batch_size,), seq_len, device=k_cache_active.device
                )

                # NOTE: This is an approximation that assumes only a causal mask.
                # External masks and previous compressions are ignored.
                # Ignoring previous evictions allows model reloads from CPU.
                attention_weights = materialize_attention_weights(
                    query=queries_for_attn,
                    key=k_expanded,
                    key_seq_lengths=seq_lengths,
                    query_seq_lengths=seq_lengths,
                    is_causal=True,
                )
            else:
                raise ValueError(
                    f"Press {self.press.__class__.__name__} requires attention weights "
                    "but neither attention_weights nor queries were provided"
                )

        cache_sorting = self.press(
            hidden_states=hidden_states,
            keys=k_cache_active,
            values=v_cache_active,
            queries=q_assoc_cache_active,
            attention_weights=attention_weights,
        )

        expected_shape = k_cache_active.shape[:3]  # [B, H, S]
        if cache_sorting.shape != expected_shape:
            raise ValueError(
                f"Press returned sorting indices with shape {cache_sorting.shape}, "
                f"expected {expected_shape}"
            )

        # Ensure target_compressed_size doesn't exceed sequence length
        seq_len = k_cache_active.shape[2]
        actual_target_size = min(target_compressed_size, seq_len)

        if actual_target_size <= 0 or seq_len == 0:
            # Return all-False mask if no tokens should be kept
            return torch.zeros(
                expected_shape,
                dtype=torch.bool,
                device=k_cache_active.device,
            )

        prefix_indices = cache_sorting[..., :actual_target_size]

        if prefix_indices.numel() > 0:  # Only check if there are indices to check
            if torch.any(prefix_indices >= seq_len) or torch.any(prefix_indices < 0):
                raise ValueError(
                    f"Press returned invalid indices. Max: {torch.max(prefix_indices)}, "
                    f"Min: {torch.min(prefix_indices)}, Sequence length: {seq_len}"
                )

        mask = torch.zeros(
            expected_shape,
            dtype=torch.bool,
            device=k_cache_active.device,
        )

        if prefix_indices.numel() > 0:  # Only scatter if there are indices
            mask.scatter_(
                index=prefix_indices,
                dim=-1,
                src=torch.ones_like(prefix_indices, dtype=torch.bool),
            )

        return mask
