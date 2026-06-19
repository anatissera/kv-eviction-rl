#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import warnings

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flex_attention_vs_sdpa_numerical_equivalence(dtype: torch.dtype):
    """Test numerical equivalence between FlexAttention and SDPA.

    This test verifies that FlexAttention and SDPA produce equivalent results.
    In float32, they should be nearly identical. In bfloat16, there are known
    numerical differences due to different implementation details.
    """
    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create identical inputs
    torch.manual_seed(42)
    q = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
    )

    # Create causal mask for both methods
    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    )

    # Method 1: Standard SDPA with causal mask
    with torch.no_grad():
        sdpa_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
            is_causal=False,  # We're providing explicit mask
        )

    # Method 2: FlexAttention with equivalent mask
    def causal_mask_fn(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx  # Causal: can attend to current and previous tokens

    block_mask = create_block_mask(
        causal_mask_fn, batch_size, num_heads, seq_len, seq_len, device=device
    )

    with torch.no_grad():
        flex_output = flex_attention(q, k, v, block_mask=block_mask)

    # Calculate differences
    diff = (flex_output - sdpa_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if dtype == torch.float32:
        # Float32 should have very high precision
        assert torch.allclose(sdpa_output, flex_output, atol=1e-5, rtol=1e-5), (
            f"FlexAttention and SDPA should be nearly identical in float32. "
            f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )
    elif dtype == torch.bfloat16:
        # BFloat16 has known numerical differences
        warnings.warn(
            f"FlexAttention and SDPA have inherent numerical differences in bfloat16. "
            f"Max difference: {max_diff:.2e}, Mean difference: {mean_diff:.2e}. "
            f"This is expected behavior due to different implementation details.",
            UserWarning,
        )

        # Verify the differences are within reasonable bounds for bfloat16
        assert max_diff < 0.1, (
            f"Numerical differences too large even for bfloat16. Max diff: {max_diff:.2e}"
        )

        # Test should still pass, but with appropriate tolerance
        assert torch.allclose(sdpa_output, flex_output, atol=2e-2, rtol=2e-2), (
            f"FlexAttention and SDPA outputs should be reasonably close in bfloat16. "
            f"Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}"
        )


def test_flex_attention_vs_sdpa_small_example():
    """Test with smaller dimensions similar to the failing test case."""
    batch_size = 1
    num_heads = 4
    seq_len = 10
    head_dim = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dtype in [torch.float32, torch.bfloat16]:
        torch.manual_seed(123)
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        k = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )
        v = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device
        )

        # SDPA
        with torch.no_grad():
            sdpa_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # FlexAttention
        def causal_mask_fn(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask_fn, batch_size, num_heads, seq_len, seq_len, device=device
        )

        with torch.no_grad():
            flex_output = flex_attention(q, k, v, block_mask=block_mask)

        diff = (flex_output - sdpa_output).abs()
        max_diff = diff.max().item()

        if dtype == torch.float32:
            assert max_diff < 1e-5, f"Float32 max diff too large: {max_diff:.2e}"
        else:  # bfloat16
            # BFloat16 has known numerical differences
            warnings.warn(
                f"FlexAttention and SDPA have inherent numerical differences in bfloat16. "
                f"Max difference: {max_diff:.2e}. "
                f"This is expected behavior due to different implementation details.",
                UserWarning,
            )
            # Just verify it's not completely wrong
            assert max_diff < 0.1, f"BFloat16 max diff too large: {max_diff:.2e}"
