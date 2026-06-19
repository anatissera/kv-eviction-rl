#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from typing import Any

import pytest
import torch

from kvcompression.kv_cache import CompressibleKVCache, DummyCompressionStrategy


class MockStrategy(DummyCompressionStrategy):
    """A mock strategy that keeps the first N tokens or last N tokens."""

    def __init__(self, multihead: bool, keep_first: bool = True):
        self._multihead = multihead
        self.keep_first = keep_first
        self.last_k_cache_active_shape = None
        self.last_target_compressed_size = None

    def is_multihead(self) -> bool:
        return self._multihead

    def get_mask(
        self,
        k_cache_active: torch.Tensor,
        v_cache_active: torch.Tensor,
        target_compressed_size: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.last_k_cache_active_shape = k_cache_active.shape
        self.last_target_compressed_size = target_compressed_size

        bsz, num_h, s_active, _ = k_cache_active.shape
        mask = torch.zeros(
            (bsz, num_h, s_active), dtype=torch.bool, device=k_cache_active.device
        )

        if target_compressed_size == 0 or s_active == 0:
            return mask

        actual_keep_count = min(target_compressed_size, s_active)

        if self.keep_first:
            mask[:, :, :actual_keep_count] = True
        else:
            mask[:, :, -actual_keep_count:] = True
        return mask


@pytest.fixture
def cache_params():
    return {
        "batch_size": 2,
        "max_seq_len": 8,
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 4,
        "dtype": torch.float32,
    }


@pytest.fixture
def mock_strategy_single_head_keep_first():
    return MockStrategy(multihead=False, keep_first=True)


@pytest.fixture
def mock_strategy_multi_head_keep_last():
    return MockStrategy(multihead=True, keep_first=False)


def _create_dummy_data(bsz, num_h, s_new, h_dim, dtype, value_offset=0):
    k = torch.randn(bsz, num_h, s_new, h_dim, dtype=dtype) + value_offset
    v = torch.randn(bsz, num_h, s_new, h_dim, dtype=dtype) + value_offset
    return k, v


def test_initialization(cache_params, mock_strategy_single_head_keep_first):
    cache = CompressibleKVCache(
        **cache_params,
        compression_strategies=[mock_strategy_single_head_keep_first]
        * cache_params["num_kv_heads"],
    )
    assert cache.size() == 0
    assert cache.k_cache.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
        cache_params["head_dim"],
    )
    assert cache.v_cache.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
        cache_params["head_dim"],
    )

    assert cache.attention_mask.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
    )
    assert cache.attention_mask.dtype == torch.bool
    assert not torch.any(cache.attention_mask).item()


def test_reset(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    (
        k,
        v,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        2,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(
        k,
        v,
    )
    cache.compress(target_size_per_head=1)

    assert cache.size() == 2
    assert cache.num_active_entries > 0

    cache.reset()
    assert cache.size() == 0
    assert cache.num_active_entries == 0
    assert torch.all(cache.k_cache == 0).item()
    assert torch.all(cache.v_cache == 0).item()

    assert not torch.any(cache.attention_mask).item()


def test_update_simple(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    s_new = 3
    (
        k,
        v,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_new,
        cache_params["head_dim"],
        cache_params["dtype"],
    )

    cache.update(
        k,
        v,
    )
    assert cache.size() == s_new
    assert torch.equal(cache.k_cache[:, :, :s_new, :], k)
    assert torch.equal(cache.v_cache[:, :, :s_new, :], v)
    assert torch.all(cache.k_cache[:, :, s_new:, :] == 0).item()


def test_update_multiple(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    s_new1 = 2
    (
        k1,
        v1,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_new1,
        cache_params["head_dim"],
        cache_params["dtype"],
        value_offset=1,
    )
    cache.update(
        k1,
        v1,
    )
    assert cache.size() == s_new1

    s_new2 = 3
    (
        k2,
        v2,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_new2,
        cache_params["head_dim"],
        cache_params["dtype"],
        value_offset=2,
    )
    cache.update(
        k2,
        v2,
    )
    assert cache.size() == s_new1 + s_new2

    assert torch.equal(cache.k_cache[:, :, :s_new1, :], k1)
    assert torch.equal(cache.k_cache[:, :, s_new1 : s_new1 + s_new2, :], k2)


def test_update_overflow(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    s_fill = cache_params["max_seq_len"] - 1
    k_fill, v_fill = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_fill,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(k_fill, v_fill)

    s_overflow = 2
    (
        k_over,
        v_over,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_overflow,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    with pytest.raises(ValueError, match="would be exceeded"):
        cache.update(k_over, v_over)


def test_update_s_new_zero(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    (
        k0,
        v0,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        0,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(
        k0,
        v0,
    )
    assert cache.size() == 0


def test_compress_empty_cache(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    mask = cache.compress(target_size_per_head=2)
    assert not torch.any(mask).item()
    assert cache.num_active_entries == 0

    if cache_params["num_kv_heads"] > 0:
        assert (
            mock_strategy_single_head_keep_first.last_k_cache_active_shape is None
            or mock_strategy_single_head_keep_first.last_k_cache_active_shape[2] == 0
        )


def test_compress_target_zero(cache_params, mock_strategy_single_head_keep_first):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    (
        k,
        v,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        3,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(
        k,
        v,
    )
    mask = cache.compress(target_size_per_head=0)
    assert not torch.any(mask).item()
    assert cache.num_active_entries == 0


def test_compress_logic_single_head_strategy(
    cache_params, mock_strategy_single_head_keep_first
):
    num_heads = cache_params["num_kv_heads"]
    strategies = [mock_strategy_single_head_keep_first] * num_heads
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)

    s_active = 5
    (
        k,
        v,
    ) = _create_dummy_data(
        cache_params["batch_size"],
        num_heads,
        s_active,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(
        k,
        v,
    )
    assert cache.size() == s_active

    target_keep = 2
    mask = cache.compress(target_size_per_head=target_keep)

    assert mock_strategy_single_head_keep_first.last_k_cache_active_shape == (
        cache_params["batch_size"],
        1,
        s_active,
        cache_params["head_dim"],
    )
    assert (
        mock_strategy_single_head_keep_first.last_target_compressed_size == target_keep
    )

    assert mask.shape == (
        cache_params["batch_size"],
        num_heads,
        cache_params["max_seq_len"],
    )
    expected_active_part = torch.zeros(
        (cache_params["batch_size"], num_heads, cache_params["max_seq_len"]),
        dtype=torch.bool,
    )
    expected_active_part[:, :, :target_keep] = True

    assert torch.equal(mask, expected_active_part)
    assert not torch.any(mask[:, :, s_active:]).item()
    assert (
        cache.num_active_entries == cache_params["batch_size"] * num_heads * target_keep
    )


def test_compress_invalid_target_size(
    cache_params, mock_strategy_single_head_keep_first
):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)
    with pytest.raises(ValueError, match="must be between 0 and max_seq_len"):
        cache.compress(target_size_per_head=-1)
    with pytest.raises(ValueError, match="must be between 0 and max_seq_len"):
        cache.compress(target_size_per_head=cache_params["max_seq_len"] + 1)


def test_num_kv_heads_zero(cache_params):
    params_no_heads = cache_params.copy()
    params_no_heads["num_kv_heads"] = 0
    cache = CompressibleKVCache(**params_no_heads, compression_strategies=[])

    assert cache.k_cache.shape == (
        params_no_heads["batch_size"],
        0,
        params_no_heads["max_seq_len"],
        params_no_heads["head_dim"],
    )
    assert cache.attention_mask.shape == (
        params_no_heads["batch_size"],
        0,
        params_no_heads["max_seq_len"],
    )

    s_new = 3

    (
        k,
        v,
    ) = _create_dummy_data(
        params_no_heads["batch_size"],
        0,
        s_new,
        params_no_heads["head_dim"],
        params_no_heads["dtype"],
    )
    cache.update(
        k,
        v,
    )
    assert cache.size() == s_new

    mask = cache.compress(target_size_per_head=1)
    assert mask.shape == (
        params_no_heads["batch_size"],
        0,
        params_no_heads["max_seq_len"],
    )
    assert cache.num_active_entries == 0


def test_get_k_v_q_and_mask_sliced_output(
    cache_params, mock_strategy_single_head_keep_first
):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)

    s_active = 3
    assert s_active < cache_params["max_seq_len"]

    k_in, v_in = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_active,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(k_in, v_in)

    target_keep = 2
    internal_full_mask = cache.compress(target_size_per_head=target_keep)

    assert internal_full_mask.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
    )
    expected_active_part_internal = torch.zeros(
        (
            cache_params["batch_size"],
            cache_params["num_kv_heads"],
            cache_params["max_seq_len"],
        ),
        dtype=torch.bool,
    )
    expected_active_part_internal[:, :, :target_keep] = True
    assert torch.equal(internal_full_mask, expected_active_part_internal)
    assert not torch.any(internal_full_mask[:, :, s_active:]).item()

    k_out, v_out, mask_out = cache.get_k_v_and_mask_for_attention()

    assert k_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
        cache_params["head_dim"],
    )
    assert v_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
        cache_params["head_dim"],
    )

    assert mask_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        cache_params["max_seq_len"],
    )

    assert torch.equal(k_out[..., :s_active, :], k_in)
    assert torch.equal(v_out[..., :s_active, :], v_in)

    assert torch.equal(mask_out, expected_active_part_internal)


def test_get_k_v_q_and_mask_full_length_output(
    cache_params, mock_strategy_single_head_keep_first
):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)

    s_active = cache_params["max_seq_len"]

    k_in, v_in = _create_dummy_data(
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_active,
        cache_params["head_dim"],
        cache_params["dtype"],
    )
    cache.update(k_in, v_in)

    target_keep = s_active // 2
    internal_full_mask = cache.compress(target_size_per_head=target_keep)

    k_out, v_out, mask_out = cache.get_k_v_and_mask_for_attention()

    assert k_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_active,
        cache_params["head_dim"],
    )
    assert mask_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        s_active,
    )
    assert torch.equal(mask_out, internal_full_mask)


def test_get_k_v_q_and_mask_empty_cache_output(
    cache_params, mock_strategy_single_head_keep_first
):
    strategies = [mock_strategy_single_head_keep_first] * cache_params["num_kv_heads"]
    cache = CompressibleKVCache(**cache_params, compression_strategies=strategies)

    internal_full_mask = cache.compress(target_size_per_head=1)
    assert not torch.any(internal_full_mask).item()

    k_out, v_out, mask_out = cache.get_k_v_and_mask_for_attention()

    maxlen = cache_params["max_seq_len"]
    assert k_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        maxlen,
        cache_params["head_dim"],
    )
    assert v_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        maxlen,
        cache_params["head_dim"],
    )

    assert mask_out.shape == (
        cache_params["batch_size"],
        cache_params["num_kv_heads"],
        maxlen,
    )
    assert mask_out.sum() == 0


def test_query_group_extraction_gqa():
    """Test that CompressibleKVCache correctly extracts query groups for GQA."""
    cache = CompressibleKVCache(
        batch_size=1,
        max_seq_len=10,
        num_kv_heads=2,
        num_q_heads=6,
        head_dim=8,
        dtype=torch.float32,
        compression_strategies=[MockStrategy(False)] * 2,
    )

    # Simulate full query tensor from model (as received from q_proj)
    q_full = torch.randn(1, 6, 5, 8)  # [B, num_q_heads, S, D]
    k_val = torch.randn(1, 2, 5, 8)  # [B, num_kv_heads, S, D]
    v_val = torch.randn(1, 2, 5, 8)

    cache.update(k_val, v_val, q_full)

    # Test get_queries_for_kv_head method - should extract 3 queries per KV head
    queries_head_0 = cache.get_queries_for_kv_head(0)
    queries_head_1 = cache.get_queries_for_kv_head(1)

    assert queries_head_0.shape == (1, 3, 5, 8), "First KV head should get queries 0-2"
    assert queries_head_1.shape == (1, 3, 5, 8), "Second KV head should get queries 3-5"

    # Verify extracted queries match original
    assert torch.equal(queries_head_0, q_full[:, :3, :5, :])
    assert torch.equal(queries_head_1, q_full[:, 3:6, :5, :])


def test_query_storage_without_queries():
    """Test cache behavior when queries are not provided during update."""
    cache = CompressibleKVCache(
        batch_size=1,
        max_seq_len=10,
        num_kv_heads=2,
        num_q_heads=4,
        head_dim=8,
        dtype=torch.float32,
        compression_strategies=[MockStrategy(False)] * 2,
    )

    k_val = torch.randn(1, 2, 3, 8)
    v_val = torch.randn(1, 2, 3, 8)

    # Update without queries (q_val=None)
    cache.update(k_val, v_val, q_val=None)

    # Query cache should remain zeros
    assert torch.all(cache.q_cache == 0), (
        "Query cache should remain empty when no queries provided"
    )

    # get_queries_for_kv_head should return zeros but correct shape
    queries_head_0 = cache.get_queries_for_kv_head(0)
    assert queries_head_0.shape == (1, 2, 3, 8)
    assert torch.all(queries_head_0 == 0)


def test_query_storage_shape_mismatch():
    """Test cache behavior with mismatched query shapes."""
    cache = CompressibleKVCache(
        batch_size=1,
        max_seq_len=10,
        num_kv_heads=2,
        num_q_heads=4,
        head_dim=8,
        dtype=torch.float32,
        compression_strategies=[MockStrategy(False)] * 2,
    )

    k_val = torch.randn(1, 2, 3, 8)
    v_val = torch.randn(1, 2, 3, 8)

    # Wrong number of query heads
    q_wrong_heads = torch.randn(1, 6, 3, 8)  # Should be 4 heads, not 6
    with pytest.raises(ValueError, match="Query shape .* doesn't match expected shape"):
        cache.update(k_val, v_val, q_wrong_heads)

    # Wrong sequence length
    q_wrong_seq = torch.randn(1, 4, 5, 8)  # Should be 3 seq len, not 5
    with pytest.raises(ValueError, match="Query shape .* doesn't match expected shape"):
        cache.update(k_val, v_val, q_wrong_seq)


class QueryCapturingStrategy(DummyCompressionStrategy):
    """Strategy that captures the queries it receives during compression."""

    def __init__(self):
        self.received_queries = None
        self.received_k_shape = None
        self.received_v_shape = None

    def get_mask(
        self,
        k_cache_active,
        v_cache_active,
        q_assoc_cache_active=None,
        target_compressed_size=1,
        **kwargs,
    ):
        self.received_queries = q_assoc_cache_active
        self.received_k_shape = k_cache_active.shape
        self.received_v_shape = v_cache_active.shape

        # Return simple mask (keep first token)
        bsz, num_h, s_active, _ = k_cache_active.shape
        mask = torch.zeros((bsz, num_h, s_active), dtype=torch.bool)
        if s_active > 0:
            mask[:, :, 0] = True
        return mask


def test_compression_strategy_receives_correct_query_groups():
    """Test that compression strategies receive properly grouped queries."""
    strategy_0 = QueryCapturingStrategy()
    strategy_1 = QueryCapturingStrategy()

    cache = CompressibleKVCache(
        batch_size=1,
        max_seq_len=10,
        num_kv_heads=2,
        num_q_heads=6,
        head_dim=8,
        dtype=torch.float32,
        compression_strategies=[strategy_0, strategy_1],
    )

    # Full query tensor (6 queries total, 3 per KV head)
    q_full = torch.randn(1, 6, 5, 8)
    k_val = torch.randn(1, 2, 5, 8)
    v_val = torch.randn(1, 2, 5, 8)

    cache.update(k_val, v_val, q_full)
    cache.compress(target_size_per_head=2)

    # Strategy 0 (KV head 0) should receive queries 0-2
    assert strategy_0.received_queries.shape == (1, 3, 5, 8)
    assert torch.equal(strategy_0.received_queries, q_full[:, :3, :5, :])

    # Strategy 1 (KV head 1) should receive queries 3-5
    assert strategy_1.received_queries.shape == (1, 3, 5, 8)
    assert torch.equal(strategy_1.received_queries, q_full[:, 3:6, :5, :])

    # Both strategies should receive single KV head data
    assert strategy_0.received_k_shape == (1, 1, 5, 8)
    assert strategy_1.received_k_shape == (1, 1, 5, 8)


def test_get_queries_for_kv_head_edge_cases():
    """Test edge cases for get_queries_for_kv_head method."""
    cache = CompressibleKVCache(
        batch_size=1,
        max_seq_len=10,
        num_kv_heads=2,
        num_q_heads=4,
        head_dim=8,
        dtype=torch.float32,
        compression_strategies=[MockStrategy(False)] * 2,
    )

    # Test invalid KV head index
    with pytest.raises(ValueError, match="KV head index .* >= num_kv_heads"):
        cache.get_queries_for_kv_head(2)

    # Test with empty cache
    queries = cache.get_queries_for_kv_head(0)
    assert queries.shape == (1, 2, 0, 8)  # Empty sequence dimension

    # Test after partial update
    q_full = torch.randn(1, 4, 3, 8)
    k_val = torch.randn(1, 2, 3, 8)
    v_val = torch.randn(1, 2, 3, 8)

    cache.update(k_val, v_val, q_full)

    queries_0 = cache.get_queries_for_kv_head(0)
    queries_1 = cache.get_queries_for_kv_head(1)

    assert queries_0.shape == (1, 2, 3, 8)
    assert queries_1.shape == (1, 2, 3, 8)
    assert torch.equal(queries_0, q_full[:, :2, :3, :])
    assert torch.equal(queries_1, q_full[:, 2:4, :3, :])
