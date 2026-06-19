#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch
from torchtune import training

from kvcompression.attention_utils import get_attention_mask
from kvcompression.hooks.compressor import KVCompressor
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
)
from tests.test_utils import base_test_model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_patched_model_forward(model_base_config_fixture: dict, dtype: torch.dtype):
    if dtype is torch.bfloat16:
        pytest.skip(
            "Currently patched and non patched attention are not equivalent in bfloat16, due to attention backend numerical differences."
        )

    with training.set_default_dtype(dtype):
        model_base_config = model_base_config_fixture
        model = base_test_model(model_base_config)

        # Use the same model and sample from your notebook
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create test inputs
        batch_size = 1
        seq_len = 10  # Small for quick testing
        tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Create input_pos and proper causal mask
        input_pos = torch.arange(seq_len, device=device).unsqueeze(0)

        # Create proper attention mask using your utility function
        attention_mask = get_attention_mask(
            batch_size=batch_size,
            q_seq_len=seq_len,
            kv_seq_len=model.max_seq_len,
            device=device,
        )

        # Baseline path
        with device:
            model.setup_caches(
                batch_size=batch_size,
                dtype=model.tok_embeddings.weight.dtype,
                decoder_max_seq_len=model.max_seq_len,
            )

            with torch.no_grad():
                baseline_output = model(
                    tokens, input_pos=input_pos, mask=attention_mask
                )

            print(f"Baseline output shape: {baseline_output.shape}")

            # Compressed path (no actual compression)
            model.reset_caches()  # Clear cache

            dummy_strategy = DummyCompressionStrategy()
            with KVCompressor(model, [dummy_strategy]):
                model.setup_caches(
                    batch_size=batch_size,
                    dtype=model.tok_embeddings.weight.dtype,
                    decoder_max_seq_len=model.max_seq_len,
                )

                with torch.no_grad():
                    # For the compressed path, we might not need the mask since it handles masking internally,
                    # but let's test both ways
                    compressed_output = model(
                        tokens, input_pos=input_pos, mask=attention_mask
                    )

        print(f"Compressed output shape: {compressed_output.shape}")

        assert torch.allclose(baseline_output, compressed_output)
