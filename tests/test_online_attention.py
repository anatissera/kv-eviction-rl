#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from kvcompression.attention_utils import (
    expand_kv_for_gqa,
    get_attention_mask,
    materialize_attention_weights,
)
from kvcompression.hooks.compressor import KVCompressor
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
)
from tests.test_utils import base_test_model, fixed_init_model


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("num_kv_heads,num_q_heads", [(2, 4), (2, 2)])
def test_attention_rematerialization_equivalence(
    model_base_config_fixture: dict,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    num_q_heads: int,
):
    """Test that attention weights re-materialized from stored queries match direct computation."""
    # Setup model and run forward pass
    model, kv_cache, stored_q_active, k_expanded, v_expanded = (
        _setup_model_and_run_forward(
            model_base_config_fixture, batch_size, seq_len, num_kv_heads, num_q_heads
        )
    )

    device = next(model.parameters()).device
    seq_lengths = torch.full((batch_size,), seq_len, device=device)

    # Re-materialize attention weights
    rematerialized_weights = materialize_attention_weights(
        query=stored_q_active,
        key=k_expanded,
        key_seq_lengths=seq_lengths,
        query_seq_lengths=seq_lengths,
        is_causal=True,
    )

    # Compute outputs for comparison
    output_custom = torch.matmul(rematerialized_weights, v_expanded)
    output_sdpa = torch.nn.functional.scaled_dot_product_attention(
        stored_q_active, k_expanded, v_expanded, is_causal=True
    )

    # Verify shapes and correctness
    _verify_attention_outputs(
        rematerialized_weights,
        output_custom,
        output_sdpa,
        batch_size,
        num_q_heads,
        seq_len,
        k_expanded.shape[-1],
    )


def _setup_model_and_run_forward(
    model_base_config_fixture, batch_size, seq_len, num_kv_heads, num_q_heads
):
    """Helper to setup model and run forward pass, returning key components for testing."""
    model_config = model_base_config_fixture.copy()
    model_config.update(
        {
            "num_kv_heads": num_kv_heads,
            "num_heads": num_q_heads,
            "max_seq_len": max(64, seq_len + 10),
        }
    )

    model = base_test_model(model_config)
    model_dtype = torch.float32
    model = model.to(dtype=model_dtype)
    fixed_init_model(model, min_val=-0.02, max_val=0.02, dtype=model_dtype)

    device = next(model.parameters()).device
    prompt_tokens = torch.randint(
        5,
        model_config["vocab_size"] // 2,
        (batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    # Setup and run forward pass
    dummy_strategy = DummyCompressionStrategy()
    kv_compressor = KVCompressor(model=model, compression_strategies=[dummy_strategy])

    with kv_compressor:
        model.setup_caches(
            batch_size=batch_size,
            dtype=model_dtype,
            decoder_max_seq_len=model_config["max_seq_len"],
        )

        input_pos = torch.arange(seq_len, device=device).unsqueeze(0)

        attention_mask = get_attention_mask(
            batch_size=batch_size,
            q_seq_len=seq_len,
            kv_seq_len=model_config["max_seq_len"],
            device=device,
        )

        torch.manual_seed(42)
        model.eval()
        with torch.no_grad():
            _ = model(prompt_tokens, input_pos=input_pos, mask=attention_mask)

        # Extract components
        first_attn_layer = next(
            module
            for module in model.modules()
            if hasattr(module, "kv_cache") and module.kv_cache is not None
        )
        kv_cache = first_attn_layer.kv_cache
        k_cache, v_cache, _ = kv_cache.get_k_v_and_mask_for_attention()
        stored_queries = kv_cache.get_all_queries_active()

        # Slice and expand for GQA
        k_active = k_cache[:, :, :seq_len, :]
        stored_q_active = stored_queries[:, :, :seq_len, :]
        k_expanded, v_expanded = expand_kv_for_gqa(
            k_active, v_cache[:, :, :seq_len, :], num_q_heads, num_kv_heads
        )

        return model, kv_cache, stored_q_active, k_expanded, v_expanded


def _verify_attention_outputs(
    rematerialized_weights,
    output_custom,
    output_sdpa,
    batch_size,
    num_q_heads,
    seq_len,
    head_dim,
):
    """Helper to verify attention outputs and weights."""
    # Verify shapes
    expected_attn_shape = (batch_size, num_q_heads, seq_len, seq_len)
    expected_output_shape = (batch_size, num_q_heads, seq_len, head_dim)

    assert rematerialized_weights.shape == expected_attn_shape, (
        f"Attention weights shape mismatch: {rematerialized_weights.shape} vs {expected_attn_shape}"
    )
    assert output_custom.shape == expected_output_shape
    assert output_sdpa.shape == expected_output_shape

    # Compare outputs
    torch.testing.assert_close(
        output_custom,
        output_sdpa,
        atol=1e-4,
        rtol=1e-3,
        equal_nan=True,
        msg="Rematerialized attention output differs from SDPA reference",
    )

    # Verify attention weights sum to 1
    attn_sums = rematerialized_weights.sum(dim=-1)
    expected_sums = torch.ones_like(attn_sums)
    torch.testing.assert_close(
        attn_sums,
        expected_sums,
        atol=1e-5,
        rtol=1e-4,
        msg="Attention weights don't sum to 1",
    )


def test_gqa_expansion_correctness():
    """Test that GQA K,V expansion works correctly."""
    batch_size, seq_len, head_dim = 2, 16, 32
    num_kv_heads, num_q_heads = 2, 8

    k_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    v_cache = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)

    k_expanded, v_expanded = expand_kv_for_gqa(
        k_cache, v_cache, num_q_heads, num_kv_heads
    )

    # Verify shapes
    assert k_expanded.shape == (batch_size, num_q_heads, seq_len, head_dim)
    assert v_expanded.shape == (batch_size, num_q_heads, seq_len, head_dim)

    # Verify replication pattern
    q_per_kv = num_q_heads // num_kv_heads
    for kv_head_idx in range(num_kv_heads):
        for replica in range(q_per_kv):
            expanded_head_idx = kv_head_idx * q_per_kv + replica
            torch.testing.assert_close(
                k_expanded[:, expanded_head_idx, :, :],
                k_cache[:, kv_head_idx, :, :],
            )


def test_edge_cases():
    """Test edge cases for attention rematerialization."""
    device = torch.device("cpu")

    # Test with minimal sequence length
    batch_size, seq_len, head_dim = 1, 1, 32
    queries = torch.randn(batch_size, 2, seq_len, head_dim, device=device)
    keys = torch.randn(batch_size, 2, seq_len, head_dim, device=device)
    seq_lengths = torch.tensor([seq_len], device=device)

    attention_weights = materialize_attention_weights(
        query=queries,
        key=keys,
        query_seq_lengths=seq_lengths,
        key_seq_lengths=seq_lengths,
        is_causal=True,
    )

    # Should be [1, 2, 1, 1] and contain only 1.0
    assert attention_weights.shape == (1, 2, 1, 1)
    torch.testing.assert_close(attention_weights, torch.ones_like(attention_weights))
