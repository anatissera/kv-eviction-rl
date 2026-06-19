#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch
from torchtune import generation

from kvcompression.hooks.compressor import KVCompressor
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
    RandomCompressionStrategy,
)
from kvcompression.utils.generation import generate_with_compression
from tests.test_utils import base_test_model, fixed_init_model


@pytest.fixture
def create_model(model_base_config_fixture: dict):
    """
    Provides a factory function to create new, identically initialized models.
    Models are created on CPU by default.
    """

    def _create(device: str = "cpu"):
        model = base_test_model(model_base_config_fixture, device=device)
        torch.manual_seed(0)
        fixed_init_model(
            model, min_val=-0.01, max_val=0.01, dtype=torch.float32, nonlinear=True
        )
        model = model.to(torch.float32)
        model.eval()
        return model

    return _create


@pytest.mark.parametrize("prompt_length", [10, 20])
@pytest.mark.parametrize("max_generated_tokens", [15, 25])
def test_matches_torchtune_without_compression(
    create_model, prompt_length: int, max_generated_tokens: int
):
    """
    Verifies `generate_with_compression(kv_compressor=None)`
    is identical to `torchtune.generate`, including stop token handling.
    """
    model1 = create_model()
    model2 = create_model()
    device = next(model1.parameters()).device
    prompt = torch.randint(low=3, high=100, size=(1, prompt_length), device=device)
    eos_id = getattr(model1, "eos_id", 2)

    model1.setup_caches(batch_size=1, dtype=torch.float32)
    torch.manual_seed(42)
    our_tokens = generate_with_compression(
        model=model1,
        prompt=prompt,
        kv_compressor=None,
        target_cache_size=0,
        max_generated_tokens=max_generated_tokens,
        stop_tokens=[eos_id],
    )

    model2.setup_caches(batch_size=1, dtype=torch.float32)
    torch.manual_seed(42)
    torchtune_result, _ = generation.generate(
        model=model2,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=0.0,
        pad_id=eos_id,
        stop_tokens=[eos_id],
    )

    assert torch.equal(our_tokens, torchtune_result), (
        f"Generation without compression should match TorchTune exactly.\n"
        f"Our result:     {our_tokens}\n"
        f"TorchTune result: {torchtune_result}"
    )


@pytest.mark.requires_cuda
@pytest.mark.parametrize("prompt_length", [30, 40])
@pytest.mark.parametrize("max_generated_tokens", [15, 20])
def test_differs_from_torchtune_with_real_compression(
    create_model, prompt_length: int, max_generated_tokens: int
):
    """
    Verifies that a real compression strategy produces results
    DIFFERENT from the standard `torchtune.generate`.
    """
    model1 = create_model(device="cuda")
    model2 = create_model(device="cuda")
    device = next(model1.parameters()).device
    prompt = torch.randint(low=3, high=100, size=(1, prompt_length), device=device)
    target_cache_size = 2
    eos_id = getattr(model1, "eos_id", 2)

    real_strategy = RandomCompressionStrategy()
    kv_compressor = KVCompressor(model=model1, compression_strategies=[real_strategy])

    torch.manual_seed(42)
    with kv_compressor:
        # setup_caches is called *after* the model is patched.
        with torch.device("cuda"):
            model1.setup_caches(batch_size=1, dtype=torch.float32)
        our_compressed_tokens = generate_with_compression(
            model=model1,
            prompt=prompt,
            kv_compressor=kv_compressor,
            target_cache_size=target_cache_size,
            max_generated_tokens=max_generated_tokens,
            stop_tokens=[eos_id],
        )

    with torch.device("cuda"):
        model2.setup_caches(batch_size=1, dtype=torch.float32)
    torch.manual_seed(42)
    torchtune_result, _ = generation.generate(
        model=model2,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=0.0,
        pad_id=eos_id,
        stop_tokens=[eos_id],
    )

    assert not torch.equal(our_compressed_tokens, torchtune_result), (
        f"Generation with a real compressor should NOT match TorchTune.\n"
        f"Our compressed result: {our_compressed_tokens}\n"
        f"TorchTune result:      {torchtune_result}"
    )


@pytest.mark.requires_cuda
@pytest.mark.parametrize("prompt_length", [10, 20])
@pytest.mark.parametrize("max_generated_tokens", [15, 25])
def test_matches_torchtune_with_noop_compression(
    create_model, prompt_length: int, max_generated_tokens: int
):
    """
    Verifies that a no-op compression strategy produces
    results IDENTICAL to `torchtune.generate`.
    """
    model1 = create_model(device="cuda")
    model2 = create_model(device="cuda")
    device = next(model1.parameters()).device
    prompt = torch.randint(low=3, high=100, size=(1, prompt_length), device=device)
    target_cache_size = prompt_length // 2
    eos_id = getattr(model1, "eos_id", 2)

    noop_strategy = DummyCompressionStrategy()
    kv_compressor = KVCompressor(model=model1, compression_strategies=[noop_strategy])

    torch.manual_seed(42)
    with kv_compressor:
        # setup_caches is called *after* the model is patched.
        with torch.device("cuda"):
            model1.setup_caches(batch_size=1, dtype=torch.float32)
        our_noop_tokens = generate_with_compression(
            model=model1,
            prompt=prompt,
            kv_compressor=kv_compressor,
            target_cache_size=target_cache_size,
            max_generated_tokens=max_generated_tokens,
            stop_tokens=[eos_id],
        )

    with torch.device("cuda"):
        model2.setup_caches(batch_size=1, dtype=torch.float32)
    torch.manual_seed(42)
    torchtune_result, _ = generation.generate(
        model=model2,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=0.0,
        pad_id=eos_id,
        stop_tokens=[eos_id],
    )

    assert torch.equal(our_noop_tokens, torchtune_result), (
        f"Generation with a no-op compressor should match TorchTune.\n"
        f"Our no-op result: {our_noop_tokens}\n"
        f"TorchTune result: {torchtune_result}"
    )


def test_batch_size_validation(create_model):
    """Tests that an error is raised for batch sizes greater than 1."""
    model = create_model()
    device = next(model.parameters()).device
    prompt = torch.randint(low=3, high=100, size=(2, 10), device=device)  # batch_size=2

    with pytest.raises(
        ValueError, match="Currently generation supports only batch size 1"
    ):
        generate_with_compression(
            model=model,
            prompt=prompt,
            kv_compressor=None,
            target_cache_size=0,
            max_generated_tokens=5,
        )
