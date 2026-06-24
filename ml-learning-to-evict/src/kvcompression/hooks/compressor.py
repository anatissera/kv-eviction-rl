#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union

from torch import nn

from kvcompression.hooks.registrar import monkeypatch_multihead_attention
from kvcompression.kv_cache.compressible_attention import (
    monkey_patched_mha_forward,
    monkey_patched_setup_cache,
)
from kvcompression.kv_cache.compressible_kv_cache import CompressibleKVCache
from kvcompression.kv_cache.compression_strategy_protocol import (
    CompressionStrategy,
    DummyCompressionStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class LayerHeadCompressionConfig:
    """Configuration for layer/head specific compression strategies using tuple format."""

    layer: Optional[int]  # None means all layers
    head: Optional[int]  # None means all heads within the target layer(s)
    strategy: DummyCompressionStrategy
    strategy_name: str = "unnamed"

    def matches(self, layer_idx: int, head_idx: int) -> bool:
        """Check if this configuration matches the given layer and head."""
        layer_match = self.layer is None or self.layer == layer_idx
        head_match = self.head is None or self.head == head_idx
        return layer_match and head_match

    def specificity(self) -> int:
        """Return specificity score for conflict resolution. Higher = more specific."""
        return (0 if self.layer is None else 2) + (0 if self.head is None else 1)

    def __lt__(self, other: "LayerHeadCompressionConfig") -> bool:
        """Enable sorting: None < 0 < 1 < ..., first by layer then by head."""
        if not isinstance(other, LayerHeadCompressionConfig):
            return NotImplemented

        # Convert None to -1 for comparison (so None sorts before 0)
        self_layer = -1 if self.layer is None else self.layer
        other_layer = -1 if other.layer is None else other.layer
        self_head = -1 if self.head is None else self.head
        other_head = -1 if other.head is None else other.head

        # Sort by layer first, then by head
        return (self_layer, self_head) < (other_layer, other_head)


class KVCompressor:
    """
    Context manager for applying KV cache compression to transformer models.

    Monkey-patches the model's attention layers to use CompressibleKVCache,
    enabling dynamic compression during generation. Supports per-layer and
    per-head compression strategy configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        compression_strategies: List[
            Union[LayerHeadCompressionConfig, CompressionStrategy]
        ],
    ):
        """
        Initialize the KV compressor.

        Args:
            model: The transformer model to apply compression to.
            compression_strategies: List of compression configurations. Can be either
                LayerHeadCompressionConfig objects for fine-grained control, or
                CompressionStrategy objects which apply globally to all layers/heads.
        """
        self._model = model

        # Normalize input to LayerHeadCompressionConfig format
        compression_strategies = [
            (
                LayerHeadCompressionConfig(
                    layer=None, head=None, strategy=config, strategy_name=repr(config)
                )
                if isinstance(config, CompressionStrategy)
                else config
            )
            for config in compression_strategies
        ]
        self.layer_head_configs = compression_strategies
        self._validate_config_consistency(compression_strategies)

        # Create the monkey-patched setup_cache function with layer_head_configs bound
        def bound_setup_cache(self_attn, batch_size, dtype, max_seq_len, **kwargs):
            return monkey_patched_setup_cache(
                self_attn,
                batch_size=batch_size,
                dtype=dtype,
                max_seq_len=max_seq_len,
                layer_head_configs=self.layer_head_configs,
                **kwargs,
            )

        self._context_manager = monkeypatch_multihead_attention(
            model=self._model,
            forward_post_hook=self.mha_attention_post_hook,
            monkeypatch_mh_attention_forward=monkey_patched_mha_forward,
            monkeypatch_mh_attention_setup_cache=bound_setup_cache,
        )

    def _validate_config_consistency(
        self, configs: List[LayerHeadCompressionConfig]
    ) -> None:
        """Validate that there are no conflicting configurations for the same layer/head."""
        if not isinstance(configs, Sequence):
            raise ValueError(
                f"Config expected to be a sequence of `LayerHeadCompressionConfig` found instead: {configs}"
            )

        if not configs:
            raise ValueError("No compression configurations provided")

        # Validate config types
        for config in configs:
            if not isinstance(config, LayerHeadCompressionConfig):
                raise ValueError(
                    f"Config expected to be a sequence of `LayerHeadCompressionConfig` found instead: {configs}"
                )

        # Check for conflicts
        self._check_exact_location_conflicts(configs)
        self._check_wildcard_conflicts(configs)
        self._check_wildcard_vs_specific_conflicts(configs)

    def _check_exact_location_conflicts(
        self, configs: List[LayerHeadCompressionConfig]
    ) -> None:
        """Check for conflicts between configs with exact (layer, head) specifications."""
        location_configs = {}

        for config in configs:
            if config.layer is not None and config.head is not None:
                key = (config.layer, config.head)
                if key in location_configs:
                    raise ValueError(
                        f"Conflicting compression strategies for layer={config.layer}, head={config.head}. "
                        f"Found both '{location_configs[key].strategy_name}' and '{config.strategy_name}'"
                    )
                location_configs[key] = config

    def _check_wildcard_conflicts(
        self, configs: List[LayerHeadCompressionConfig]
    ) -> None:
        """Check for conflicts between different wildcard configurations."""
        wildcard_types = {
            "global": [],  # layer=None, head=None
            "layer_specific": [],  # layer=X, head=None
            "head_specific": [],  # layer=None, head=Y
        }

        # Categorize wildcard configs
        for config in configs:
            if config.layer is None and config.head is None:
                wildcard_types["global"].append(config)
            elif config.layer is None:
                wildcard_types["head_specific"].append(config)
            elif config.head is None:
                wildcard_types["layer_specific"].append(config)

        # Check for conflicts between wildcard types
        if wildcard_types["global"] and wildcard_types["layer_specific"]:
            global_config = wildcard_types["global"][0]
            layer_config = wildcard_types["layer_specific"][0]
            raise ValueError(
                f"Logical conflict: global wildcard '{global_config.strategy_name}' (layer=None, head=None) "
                f"conflicts with layer-specific wildcard '{layer_config.strategy_name}' "
                f"(layer={layer_config.layer}, head=None)"
            )

        if wildcard_types["global"] and wildcard_types["head_specific"]:
            global_config = wildcard_types["global"][0]
            head_config = wildcard_types["head_specific"][0]
            raise ValueError(
                f"Logical conflict: global wildcard '{global_config.strategy_name}' (layer=None, head=None) "
                f"conflicts with head-specific wildcard '{head_config.strategy_name}' "
                f"(layer=None, head={head_config.head})"
            )

    def _check_wildcard_vs_specific_conflicts(
        self, configs: List[LayerHeadCompressionConfig]
    ) -> None:
        """Check for conflicts between wildcard and specific configurations."""
        # Group configs by type
        specific_configs = [
            (c.layer, c.head, c)
            for c in configs
            if c.layer is not None and c.head is not None
        ]
        wildcard_configs = [c for c in configs if c.layer is None or c.head is None]

        # Check each specific config against wildcards
        for layer, head, config in specific_configs:
            for wildcard_config in wildcard_configs:
                if self._configs_conflict(config, wildcard_config):
                    raise ValueError(
                        f"Logical conflict for layer={layer}, head={head}: "
                        f"specific assignment '{config.strategy_name}' conflicts with "
                        f"wildcard assignment '{wildcard_config.strategy_name}' "
                        f"(layer={wildcard_config.layer}, head={wildcard_config.head})"
                    )

    def _configs_conflict(
        self,
        specific_config: LayerHeadCompressionConfig,
        wildcard_config: LayerHeadCompressionConfig,
    ) -> bool:
        """Check if a specific config conflicts with a wildcard config."""
        # Wildcard config matches specific config if:
        # 1. layer=X, head=Y conflicts with layer=X, head=None
        # 2. layer=X, head=Y conflicts with layer=None, head=Y
        # 3. layer=X, head=Y conflicts with layer=None, head=None

        if (
            wildcard_config.layer == specific_config.layer
            and wildcard_config.head is None
        ):
            return True  # Case 1

        if (
            wildcard_config.layer is None
            and wildcard_config.head == specific_config.head
        ):
            return True  # Case 2

        if wildcard_config.layer is None and wildcard_config.head is None:
            return True  # Case 3

        return False

    def __enter__(self):
        self._context_manager.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for layer in self._model.layers:
            layer.attn.cache_enabled = False
            layer.attn.kv_cache = None
        return self._context_manager.__exit__(exc_type, exc_val, exc_tb)

    def compress_caches(
        self,
        target_size_per_head: int,
        **strategy_kwargs: Any,
    ):
        for layer_idx, layer in enumerate(self._model.layers):
            if hasattr(layer.attn, "kv_cache") and isinstance(
                layer.attn.kv_cache, CompressibleKVCache
            ):
                layer.attn.kv_cache.compress(
                    target_size_per_head=target_size_per_head, **strategy_kwargs
                )

    def mha_attention_post_hook(self, *args, **kwargs):
        pass
