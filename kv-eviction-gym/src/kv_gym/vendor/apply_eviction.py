# Copied from internal-signals-context-compression/src/eviction/apply_eviction.py
# Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
# Used in Phase 2 (correctness reward) to trim the HuggingFace DynamicCache
# to the tokens the policy chose to keep, then re-run generation.
"""
Apply per-layer keep-indices to a HuggingFace ``DynamicCache``.

Different layers can keep different KV positions — the convention used by
H2O, SnapKV, PyramidKV, …. For each layer we slice K and V along the
sequence-length dim (dim=2 in the standard ``[batch, n_kv_heads, seq_len,
head_dim]`` layout) using that layer's own keep-indices row.

This works with stock HuggingFace because RoPE is baked into the K values
at the time they enter the cache (rotation applied per K's original
position). Slicing leaves those rotations untouched. The caller still
needs to pass the new token's true original ``position_ids`` to the next
forward pass.

Requires ``transformers >= 4.40``; the project pins it.
"""

from torch import Tensor
from transformers.cache_utils import DynamicCache


def apply_eviction(cache: DynamicCache, keep_indices_per_layer: Tensor) -> DynamicCache:
    """Trim each layer of ``cache`` to its own kept positions, in place.

    Args:
        cache: ``DynamicCache`` from a HuggingFace forward pass.
        keep_indices_per_layer: Long tensor of shape ``[L, K_kept]``.
            ``keep_indices_per_layer[layer_idx]`` lists the KV positions to keep
            in layer layer_idx, sorted ascending.
    """
    for layer_idx, layer in enumerate(cache.layers):
        idx = keep_indices_per_layer[layer_idx].to(layer.keys.device)
        layer.keys = layer.keys.index_select(dim=2, index=idx)
        layer.values = layer.values.index_select(dim=2, index=idx)
    return cache


def cache_seq_len(cache: DynamicCache) -> int:
    return cache.layers[0].keys.shape[2]


def kv_cache_size_mb(cache: DynamicCache) -> float:
    total_bytes = 0
    for layer in cache.layers:
        total_bytes += layer.keys.numel() * layer.keys.element_size()
        total_bytes += layer.values.numel() * layer.values.element_size()
    return total_bytes / (1024 ** 2)
