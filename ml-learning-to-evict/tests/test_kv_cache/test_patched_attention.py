#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import types
from typing import Any, Dict, List

import pytest
import torch
import torch.nn as nn
from torchtune.modules import KVCache, MultiHeadAttention

from kvcompression.kv_cache.compressible_attention import monkey_patched_mha_forward
from kvcompression.kv_cache.compressible_kv_cache import CompressibleKVCache
from kvcompression.kv_cache.compressible_mha_layer import CompressibleMultiHeadAttention
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
)

# Base head configurations (num_heads, num_kv_heads, head_dim)
_head_configs = [
    {"num_heads": 4, "num_kv_heads": 4, "head_dim": 32},  # MHA
    {"num_heads": 4, "num_kv_heads": 2, "head_dim": 32},  # GQA
    {"num_heads": 4, "num_kv_heads": 1, "head_dim": 32},  # MQA
]

# Scenario configurations
_scenario_configs_simplified: List[pytest.param] = []
for head_conf in _head_configs:
    for batch_size in [1, 2]:
        for seq_len_prefill in [1, 5]:
            for seq_len_decode in [1, 3, 5]:
                for is_causal_mha in [True, False]:
                    for max_len_cache_val in [16, 32]:
                        if seq_len_prefill + seq_len_decode <= max_len_cache_val:
                            cfg = {
                                "batch_size": batch_size,
                                "seq_len_prefill": seq_len_prefill,
                                "seq_len_decode": seq_len_decode,
                                "is_causal": is_causal_mha,
                                "max_seq_len": max_len_cache_val,
                                "dtype": torch.float32,
                                **head_conf,
                            }
                            cfg["embed_dim"] = cfg["num_heads"] * cfg["head_dim"]

                            param_id = (
                                f"H{cfg['num_heads']}_KV{cfg['num_kv_heads']}"
                                f"_B{batch_size}_Pref{seq_len_prefill}_Dec{seq_len_decode}"
                                f"_Causal{is_causal_mha}_Max{max_len_cache_val}"
                            )
                            _scenario_configs_simplified.append(
                                pytest.param(cfg, id=param_id)
                            )


@pytest.fixture(params=_scenario_configs_simplified)
def config(request):
    return request.param


def create_mha_instance(
    config: Dict[str, Any],
    use_standard_kv_cache: bool,
    device: torch.device,
    mha_class=MultiHeadAttention,
):
    q_proj = nn.Linear(
        config["embed_dim"],
        config["num_heads"] * config["head_dim"],
        device=device,
        dtype=config["dtype"],
    )
    k_proj = nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"] * config["head_dim"],
        device=device,
        dtype=config["dtype"],
    )
    v_proj = nn.Linear(
        config["embed_dim"],
        config["num_kv_heads"] * config["head_dim"],
        device=device,
        dtype=config["dtype"],
    )
    output_proj = nn.Linear(
        config["num_heads"] * config["head_dim"],
        config["embed_dim"],
        device=device,
        dtype=config["dtype"],
    )

    kv_cache_instance = None
    if use_standard_kv_cache:
        # Create and setup the KVCache
        kv_cache_instance = KVCache(
            batch_size=config["batch_size"],
            max_seq_len=config["max_seq_len"],
            num_kv_heads=config["num_kv_heads"],
            head_dim=config["head_dim"],
            dtype=config["dtype"],
        )
        kv_cache_instance.to(device)
    else:
        kv_cache_instance = None

    mha = mha_class(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        output_proj=output_proj,
        kv_cache=kv_cache_instance,
        max_seq_len=config["max_seq_len"],
        is_causal=config["is_causal"],
        attn_dropout=0.0,
    )
    mha.to(device=device, dtype=config["dtype"])
    mha.eval()

    if use_standard_kv_cache and kv_cache_instance:
        mha.cache_enabled = True

    return mha


def test_mha_patch_equivalence(config: Dict[str, Any]):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Original MHA and run (prefill)
    mha_original = create_mha_instance(
        config, use_standard_kv_cache=True, device=device
    )

    # Initialize MHA for patching (share weights)
    mha_patched = create_mha_instance(
        config, use_standard_kv_cache=False, device=device
    )
    mha_patched.q_proj.load_state_dict(mha_original.q_proj.state_dict())
    mha_patched.k_proj.load_state_dict(mha_original.k_proj.state_dict())
    mha_patched.v_proj.load_state_dict(mha_original.v_proj.state_dict())
    mha_patched.output_proj.load_state_dict(mha_original.output_proj.state_dict())

    # Substitute KV Cache with CompressibleKVCache
    comp_strategy = DummyCompressionStrategy()
    mock_comp_cache = CompressibleKVCache(
        batch_size=config["batch_size"],
        max_seq_len=config["max_seq_len"],
        num_kv_heads=config["num_kv_heads"],
        num_q_heads=config["num_heads"],
        head_dim=config["head_dim"],
        dtype=config["dtype"],
        compression_strategies=[comp_strategy] * config["num_kv_heads"],
    )
    mock_comp_cache.to(device)
    mha_patched.kv_cache = mock_comp_cache
    mha_patched.cache_enabled = True

    mha_patched.forward = types.MethodType(monkey_patched_mha_forward, mha_patched)

    # Prefill Step
    x_prefill = torch.randn(
        config["batch_size"],
        config["seq_len_prefill"],
        config["embed_dim"],
        device=device,
        dtype=config["dtype"],
    )

    # IMPORTANT!
    # Original forward with KV caching enabled and no mask is BROKEN!
    # We need to always pass the mask, otherwise it will silently attend to empty cache entries up to max len.
    attention_mask_for_original = torch.zeros(
        config["batch_size"],
        config["seq_len_prefill"],
        config["max_seq_len"],
        dtype=torch.bool,
        device=device,
    )
    attention_mask_for_original[:, :, : config["seq_len_prefill"]] = True

    input_pos_prefill = (
        torch.arange(config["seq_len_prefill"], device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(config["batch_size"], 1)
    )

    for input_pos in [None, input_pos_prefill]:
        # Do not put in cache things twice
        mha_original.kv_cache.reset()
        mha_patched.kv_cache.reset()

        # Run Original MHA (prefill)
        output_original_prefill = mha_original(
            x_prefill,
            y=x_prefill,
            mask=attention_mask_for_original,
            input_pos=input_pos,
        )
        # Run Patched MHA (prefill)
        output_patched_prefill = mha_patched(
            x_prefill, y=x_prefill, mask=None, input_pos=input_pos
        )
        # Assert prefill outputs
        assert torch.allclose(
            output_original_prefill, output_patched_prefill, atol=1e-5, rtol=1e-4
        ), f"Prefill outputs differ for config: {config}"

    # Decode Step
    if config["seq_len_decode"] > 0:
        x_decode = torch.randn(
            config["batch_size"],
            config["seq_len_decode"],
            config["embed_dim"],
            device=device,
            dtype=config["dtype"],
        )

        decode_start_pos = config["seq_len_prefill"]
        input_pos_decode_values = torch.arange(
            decode_start_pos,
            decode_start_pos + config["seq_len_decode"],
            device=device,
            dtype=torch.long,
        )
        input_pos_decode = input_pos_decode_values.unsqueeze(0).repeat(
            config["batch_size"], 1
        )

        # Mask for original MHA decode. True means attend.
        decode_mask_for_original = torch.zeros(
            config["batch_size"],
            config["seq_len_decode"],
            config["max_seq_len"],
            dtype=torch.bool,
            device=device,
        )
        # Allow attention to the first `seq_len_prefill` tokens in the cache.
        # This is the state of the cache after the prefill step.
        valid_cache_len_after_prefill = config["seq_len_prefill"]
        decode_mask_for_original[:, :, :valid_cache_len_after_prefill] = True

        output_original_decode = mha_original(
            x_decode, y=None, mask=decode_mask_for_original, input_pos=input_pos_decode
        )

        # Patched MHA decode
        # Mask=None: Patched forward attends to all valid tokens in its cache.
        # `input_pos` is passed for RoPE application to new Q, K_new, V_new and for cache update.
        output_patched_decode = mha_patched(
            x_decode, y=None, mask=None, input_pos=input_pos_decode
        )
        # Assert decode outputs
        assert torch.allclose(
            output_original_decode, output_patched_decode, atol=1e-5, rtol=1e-4
        ), "Decode outputs differ."

        # Patched MHA decode with explicit mask
        output_patched_decode_explicit_mask = mha_patched(
            x_decode, y=None, mask=decode_mask_for_original, input_pos=input_pos_decode
        )
        # Assert decode outputs
        assert torch.allclose(
            output_original_decode,
            output_patched_decode_explicit_mask,
            atol=1e-5,
            rtol=1e-4,
        ), "Decode outputs differ."

        print(
            f"Test passed for config: {config['num_heads']}H, {config['num_kv_heads']}KVH"
        )


def test_compressible_mha_equivalence(config: Dict[str, Any]):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Original MHA and run (prefill)
    mha_original = create_mha_instance(
        config, use_standard_kv_cache=True, device=device
    )

    # Initialize MHA for patching (share weights)
    mha_compressible = create_mha_instance(
        config,
        use_standard_kv_cache=False,
        device=device,
        mha_class=CompressibleMultiHeadAttention,
    )
    mha_compressible.load_state_dict(mha_original.state_dict())

    # Substitute KV Cache with CompressibleKVCache
    comp_strategy = DummyCompressionStrategy()
    mock_comp_cache = CompressibleKVCache(
        batch_size=config["batch_size"],
        max_seq_len=config["max_seq_len"],
        num_q_heads=config["num_heads"],
        num_kv_heads=config["num_kv_heads"],
        head_dim=config["head_dim"],
        dtype=config["dtype"],
        compression_strategies=[comp_strategy] * config["num_kv_heads"],
    )
    mock_comp_cache.to(device)
    mha_compressible.kv_cache = mock_comp_cache
    mha_compressible.cache_enabled = True

    # Prefill Step
    x_prefill = torch.randn(
        config["batch_size"],
        config["seq_len_prefill"],
        config["embed_dim"],
        device=device,
        dtype=config["dtype"],
    )

    # IMPORTANT!
    # Original forward with KV caching enabled and no mask is BROKEN!
    # We need to always pass the mask, otherwise it will silently attend to empty cache entries up to max len.
    attention_mask_for_original = torch.zeros(
        config["batch_size"],
        config["seq_len_prefill"],
        config["max_seq_len"],
        dtype=torch.bool,
        device=device,
    )
    attention_mask_for_original[:, :, : config["seq_len_prefill"]] = True

    input_pos_prefill = (
        torch.arange(config["seq_len_prefill"], device=device, dtype=torch.long)
        .unsqueeze(0)
        .repeat(config["batch_size"], 1)
    )

    for input_pos in [None, input_pos_prefill]:
        # Do not put in cache things twice
        mha_original.kv_cache.reset()
        mha_compressible.kv_cache.reset()

        # Run Original MHA (prefill)
        output_original_prefill = mha_original(
            x_prefill,
            y=x_prefill,
            mask=attention_mask_for_original,
            input_pos=input_pos,
        )
        # Run Patched MHA (prefill)
        output_patched_prefill = mha_compressible(
            x_prefill, y=x_prefill, mask=None, input_pos=input_pos
        )
        # Assert prefill outputs
        assert torch.allclose(
            output_original_prefill, output_patched_prefill, atol=1e-5, rtol=1e-4
        ), f"Prefill outputs differ for config: {config}"

    # Decode Step
    if config["seq_len_decode"] > 0:
        x_decode = torch.randn(
            config["batch_size"],
            config["seq_len_decode"],
            config["embed_dim"],
            device=device,
            dtype=config["dtype"],
        )

        decode_start_pos = config["seq_len_prefill"]
        input_pos_decode_values = torch.arange(
            decode_start_pos,
            decode_start_pos + config["seq_len_decode"],
            device=device,
            dtype=torch.long,
        )
        input_pos_decode = input_pos_decode_values.unsqueeze(0).repeat(
            config["batch_size"], 1
        )

        # Mask for original MHA decode. True means attend.
        decode_mask_for_original = torch.zeros(
            config["batch_size"],
            config["seq_len_decode"],
            config["max_seq_len"],
            dtype=torch.bool,
            device=device,
        )
        # Allow attention to the first `seq_len_prefill` tokens in the cache.
        # This is the state of the cache after the prefill step.
        valid_cache_len_after_prefill = config["seq_len_prefill"]
        decode_mask_for_original[:, :, :valid_cache_len_after_prefill] = True

        output_original_decode = mha_original(
            x_decode, y=None, mask=decode_mask_for_original, input_pos=input_pos_decode
        )

        # Patched MHA decode
        # Mask=None: Patched forward attends to all valid tokens in its cache.
        # `input_pos` is passed for RoPE application to new Q, K_new, V_new and for cache update.
        output_patched_decode = mha_compressible(
            x_decode, y=None, mask=None, input_pos=input_pos_decode
        )
        # Assert decode outputs
        assert torch.allclose(
            output_original_decode, output_patched_decode, atol=1e-5, rtol=1e-4
        ), "Decode outputs differ."

        # Patched MHA decode with explicit mask
        output_patched_decode_explicit_mask = mha_compressible(
            x_decode, y=None, mask=decode_mask_for_original, input_pos=input_pos_decode
        )
        # Assert decode outputs
        assert torch.allclose(
            output_original_decode,
            output_patched_decode_explicit_mask,
            atol=1e-5,
            rtol=1e-4,
        ), "Decode outputs differ."

        print(
            f"Test passed for config: {config['num_heads']}H, {config['num_kv_heads']}KVH"
        )
