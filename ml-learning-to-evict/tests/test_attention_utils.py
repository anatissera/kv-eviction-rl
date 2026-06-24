#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from kvcompression.attention_utils import materialize_attention_weights


def _create_sdpa_equivalent_additive_bias(
    query_shape: Tuple[int, ...],
    key_shape: Tuple[int, ...],
    is_causal_custom: Optional[bool],
    explicit_attn_mask_custom: Optional[torch.Tensor],
    query_seq_lengths_custom: Optional[torch.Tensor],
    key_seq_lengths_custom: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    L, S = query_shape[-2], key_shape[-2]
    needs_bias = False

    if len(query_shape) == 4:
        target_bias_shape = (query_shape[0], query_shape[1], L, S)
    elif len(query_shape) == 3:
        target_bias_shape = (query_shape[0], L, S)
    else:
        raise ValueError("Query shape not supported for bias construction")

    expected_bias = torch.zeros(target_bias_shape, dtype=dtype, device=device)

    if is_causal_custom and L > 0 and S > 0:
        causal_mask_values = torch.triu(
            torch.ones(L, S, dtype=torch.bool, device=device), diagonal=1
        )
        expected_bias.masked_fill_(causal_mask_values, float("-inf"))
        needs_bias = True

    if explicit_attn_mask_custom is not None:
        needs_bias = True
        _eff_mask = explicit_attn_mask_custom
        if query_shape[0] > 0 and L > 0 and S > 0:
            if (
                len(query_shape) == 4
                and _eff_mask.ndim == 3
                and _eff_mask.size(0) == query_shape[0]
                and _eff_mask.size(1) == L
                and _eff_mask.size(2) == S
            ):
                _eff_mask = _eff_mask[:, None, :, :]

            if _eff_mask.dtype == torch.bool:
                expected_bias.masked_fill_(_eff_mask.logical_not(), float("-inf"))
            else:
                expected_bias = expected_bias + _eff_mask
        elif (
            _eff_mask.dtype != torch.bool
        ):  # if float, add even if L/S=0, bias might have other values
            expected_bias = expected_bias + _eff_mask

    if key_seq_lengths_custom is not None and S > 0:
        needs_bias = True
        B_key = key_shape[0]
        key_indices = torch.arange(S, device=device)[None, :]
        padded_key_mask = key_indices >= key_seq_lengths_custom[:, None]
        expanded_mask_shape = (
            (B_key, 1, 1, S) if len(query_shape) == 4 else (B_key, 1, S)
        )
        expected_bias.masked_fill_(
            padded_key_mask.view(expanded_mask_shape), float("-inf")
        )

    if query_seq_lengths_custom is not None and L > 0:
        needs_bias = True
        B_query = query_shape[0]
        query_indices = torch.arange(L, device=device)[None, :]
        padded_query_mask = query_indices >= query_seq_lengths_custom[:, None]
        expanded_mask_shape = (
            (B_query, 1, L, 1) if len(query_shape) == 4 else (B_query, L, 1)
        )
        expected_bias.masked_fill_(
            padded_query_mask.view(expanded_mask_shape), float("-inf")
        )

    return expected_bias.to(dtype) if needs_bias else None


SHAPE_PARAMS = [
    # (B, H_or_None, L, S, Dk)
    (2, None, 10, 10, 32),
    (1, None, 8, 12, 16),
    (2, 4, 10, 10, 32),
    (1, 2, 8, 6, 64),
    (1, None, 5, 5, 8),
    (1, None, 0, 5, 8),
    (1, None, 5, 0, 8),
    (1, None, 0, 0, 8),  # Zero-length seqs
    (1, 2, 0, 5, 8),
    (1, 2, 5, 0, 8),
    (1, None, 1, 1, 1),  # Minimal non-zero
]

MASK_PARAMS = [
    # (is_causal, use_query_padding, use_key_padding, explicit_mask_type)
    (False, False, False, None),
    (True, False, False, None),
    (False, True, False, None),
    (False, False, True, None),
    (True, True, True, None),  # All three masks
    (False, True, True, "bool_keep_true"),  # Padding + explicit bool
    (False, True, True, "float_additive"),  # Padding + explicit float
]

DTYPES = [torch.float32, torch.float64]


@pytest.mark.parametrize("B, H, L_padded, S_padded, Dk", SHAPE_PARAMS)
@pytest.mark.parametrize(
    "use_causal, use_q_pad, use_k_pad, explicit_mask_type", MASK_PARAMS
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_materialize_attention_weights_against_sdpa(
    B,
    H,
    L_padded,
    S_padded,
    Dk,
    use_causal,
    use_q_pad,
    use_k_pad,
    explicit_mask_type,
    dtype,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_causal and explicit_mask_type:
        pytest.skip("Cannot use causal and explicit mask together.")
    if Dk == 0 and (L_padded > 0 or S_padded > 0):
        pytest.skip(
            "SDPA behavior for Dk=0 with non-zero L/S can be specific; skipping."
        )

    q_shape = (B, H, L_padded, Dk) if H else (B, L_padded, Dk)
    k_shape = (B, H, S_padded, Dk) if H else (B, S_padded, Dk)
    v_shape = k_shape  # Assuming V has same S_padded and Dk as K for simplicity

    query = torch.randn(q_shape, device=device, dtype=dtype)
    key = torch.randn(k_shape, device=device, dtype=dtype)
    value = torch.randn(v_shape, device=device, dtype=dtype)

    q_lens = (
        torch.randint(0, L_padded + 1, (B,), device=device)
        if use_q_pad and L_padded > 0
        else None
    )
    if use_q_pad and L_padded == 0:
        q_lens = torch.zeros(B, dtype=torch.long, device=device)

    k_lens = (
        torch.randint(0, S_padded + 1, (B,), device=device)
        if use_k_pad and S_padded > 0
        else None
    )
    if use_k_pad and S_padded == 0:
        k_lens = torch.zeros(B, dtype=torch.long, device=device)

    expl_mask = None
    if explicit_mask_type and L_padded > 0 and S_padded > 0:
        mask_shape = (B, L_padded, S_padded)
        if explicit_mask_type == "bool_keep_true":
            expl_mask = torch.rand(mask_shape, device=device) > 0.3
        elif explicit_mask_type == "float_additive":
            expl_mask = torch.randn(mask_shape, device=device, dtype=dtype)
            expl_mask[torch.rand(mask_shape, device=device) < 0.2] = -float("inf")
    elif explicit_mask_type:  # L or S is 0, cannot create mask
        explicit_mask_type = None  # Disable this scenario

    # Custom implementation
    weights_custom = materialize_attention_weights(
        query,
        key,
        attn_mask=expl_mask,
        is_causal=use_causal,
        query_seq_lengths=q_lens,
        key_seq_lengths=k_lens,
    )
    output_custom = torch.matmul(weights_custom, value)

    # SDPA equivalent
    sdpa_bias = _create_sdpa_equivalent_additive_bias(
        q_shape, k_shape, use_causal, expl_mask, q_lens, k_lens, device, dtype
    )
    # If only causal is true and no other masks generated a bias, SDPA's own causal flag is used.
    sdpa_is_causal_arg = use_causal and sdpa_bias is None

    output_sdpa = F.scaled_dot_product_attention(
        query, key, value, attn_mask=sdpa_bias, is_causal=sdpa_is_causal_arg
    )

    # Comparison
    atol = 1e-6 if dtype == torch.float64 else 1e-4
    rtol = 1e-5 if dtype == torch.float64 else 1e-3
    assert torch.allclose(
        output_custom, output_sdpa, atol=atol, rtol=rtol, equal_nan=True
    ), (
        f"Config: B={B},H={H},L={L_padded},S={S_padded},Dk={Dk}, "
        f"causal={use_causal}, q_pad={use_q_pad}, k_pad={use_k_pad}, "
        f"mask_type={explicit_mask_type}, dtype={dtype}"
    )
