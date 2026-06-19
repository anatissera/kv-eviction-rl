#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import pytest
import torch
from torchtune import generation

from kvcompression.attention_utils import get_attention_mask
from kvcompression.hooks.compressor import KVCompressor
from kvcompression.kv_cache.compression_strategy_protocol import (
    DummyCompressionStrategy,
    PressCompressionStrategy,
    RandomCompressionStrategy,
)
from kvcompression.presses.random_press import RandomPress
from tests.test_utils import base_test_model, fixed_init_model


@pytest.mark.requires_cuda
class TestKVCompressorEquivalence:
    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("use_kv_cache", [True])
    @pytest.mark.parametrize("prompt_seq_len", [8, 16])
    def test_output_equivalence_with_noop_strategy(
        self,
        model_base_config_fixture: dict,
        batch_size: int,
        use_kv_cache: bool,
        prompt_seq_len: int,
    ):
        model_base_config = model_base_config_fixture
        original_model = base_test_model(model_base_config, device="cuda")
        patched_model = base_test_model(model_base_config, device="cuda")

        assert use_kv_cache, "This test is designed for use_kv_cache=True"
        assert prompt_seq_len <= model_base_config["max_seq_len"], (
            f"prompt_seq_len ({prompt_seq_len}) > model.max_seq_len ({model_base_config['max_seq_len']})"
        )

        # Use device context to ensure caches are created on CUDA
        with torch.device("cuda"):
            original_model.setup_caches(
                batch_size=batch_size,
                dtype=original_model.tok_embeddings.weight.dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )

        current_device = original_model.tok_embeddings.weight.device
        prompt_tokens = torch.randint(
            5,
            model_base_config["vocab_size"] // 2,
            (batch_size, prompt_seq_len),
            dtype=torch.long,
            device=current_device,
        )

        input_pos_tensor = torch.arange(
            0, prompt_seq_len, device=current_device, dtype=torch.long
        ).unsqueeze(0)

        # Get output from the original model
        attention_mask = get_attention_mask(
            batch_size=batch_size,
            q_seq_len=prompt_seq_len,
            kv_seq_len=model_base_config["max_seq_len"],
            device=current_device,
        )

        torch.manual_seed(123)
        original_model.eval()
        with torch.no_grad():
            output_original = original_model(
                prompt_tokens,
                input_pos=input_pos_tensor,
                mask=attention_mask,
            )
        assert output_original is not None

        # Get output from the model within the KVCompressor context

        torch.manual_seed(123)
        original_model.eval()

        no_op_strategy = DummyCompressionStrategy()
        kv_compressor = KVCompressor(
            model=patched_model,
            compression_strategies=[no_op_strategy],
        )
        with kv_compressor:
            with torch.device("cuda"):
                patched_model.setup_caches(
                    batch_size=batch_size,
                    dtype=original_model.tok_embeddings.weight.dtype,
                    decoder_max_seq_len=model_base_config["max_seq_len"],
                )
            with torch.no_grad():
                output_patched = patched_model(
                    prompt_tokens,
                    input_pos=input_pos_tensor,
                    mask=attention_mask,
                )
        assert output_patched is not None

        torch.testing.assert_close(
            output_original,
            output_patched,
            msg=(
                f"Patched output differs from original. "
                f"Params: batch_size={batch_size}, use_kv_cache={use_kv_cache}, seq_len={prompt_seq_len}"
            ),
        )

        # Get output after KVCompressor context (model should be restored)
        with torch.device("cuda"):
            patched_model.setup_caches(
                batch_size=batch_size,
                dtype=patched_model.tok_embeddings.weight.dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )

        attention_mask_after_context = get_attention_mask(
            batch_size=batch_size,
            q_seq_len=prompt_seq_len,
            kv_seq_len=model_base_config["max_seq_len"],
            device=current_device,
        )

        torch.manual_seed(123)
        patched_model.eval()
        with torch.no_grad():
            output_after_context = patched_model(
                prompt_tokens,
                input_pos=input_pos_tensor,
                mask=attention_mask_after_context,
            )
        assert output_after_context is not None

        torch.testing.assert_close(
            output_original,
            output_after_context,
            msg=(
                f"Restored output differs from original. "
                f"Params: batch_size={batch_size}, use_kv_cache={use_kv_cache}, seq_len={prompt_seq_len}"
            ),
        )

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("prompt_seq_len", [8, 12])
    @pytest.mark.parametrize("max_generated_tokens", [5, 7])
    def test_generate_equivalence_with_noop_strategy(
        self,
        model_base_config_fixture: dict,
        batch_size: int,
        prompt_seq_len: int,
        max_generated_tokens: int,
    ):
        model_base_config = model_base_config_fixture.copy()

        # Ensure total length does not exceed model's max_seq_len
        if prompt_seq_len + max_generated_tokens > model_base_config["max_seq_len"]:
            adjusted_max_gen = model_base_config["max_seq_len"] - prompt_seq_len
            if adjusted_max_gen <= 0:
                pytest.skip(
                    f"Prompt len {prompt_seq_len} + min 1 generated token "
                    f"exceeds max_seq_len {model_base_config['max_seq_len']}"
                )
            max_generated_tokens = adjusted_max_gen
            if max_generated_tokens == 0:
                pytest.skip(
                    f"Prompt len {prompt_seq_len} fills max_seq_len {model_base_config['max_seq_len']}, cannot generate more."
                )

        # Create models on CUDA
        original_model = base_test_model(model_base_config, device="cuda")
        model_dtype = torch.float32
        original_model = original_model.to(dtype=model_dtype)
        fixed_init_model(original_model, min_val=-0.02, max_val=0.02, dtype=model_dtype)

        patched_model = base_test_model(model_base_config, device="cuda")
        patched_model = patched_model.to(dtype=model_dtype)
        fixed_init_model(patched_model, min_val=-0.02, max_val=0.02, dtype=model_dtype)

        # Setup caches using device context
        with torch.device("cuda"):
            original_model.setup_caches(
                batch_size=batch_size,
                dtype=model_dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )

        current_device = next(original_model.parameters()).device

        pad_id = getattr(original_model, "eos_id", 2)

        prompt_tokens = torch.randint(
            low=pad_id + 1
            if pad_id is not None and pad_id + 1 < model_base_config["vocab_size"]
            else 0,
            high=model_base_config["vocab_size"] // 2,
            size=(batch_size, prompt_seq_len),
            dtype=torch.long,
            device=current_device,
        )

        # Generate from the original model
        torch.manual_seed(123)
        original_model.eval()
        with torch.no_grad():
            generated_tokens_original, _ = generation.generate(
                model=original_model,
                prompt=prompt_tokens,
                max_generated_tokens=max_generated_tokens,
                pad_id=pad_id,
                temperature=0.0,
                top_k=1,
                stop_tokens=pad_id,
            )
        assert generated_tokens_original is not None

        # Generate from the patched model (within KVCompressor context)
        no_op_strategy = RandomCompressionStrategy()
        kv_compressor = KVCompressor(
            model=patched_model,
            compression_strategies=[no_op_strategy],
        )

        torch.manual_seed(123)
        patched_model.eval()
        with kv_compressor:
            with torch.device("cuda"):
                patched_model.setup_caches(
                    batch_size=batch_size,
                    dtype=model_dtype,
                    decoder_max_seq_len=model_base_config["max_seq_len"],
                )
            with torch.no_grad():
                generated_tokens_patched, _ = generation.generate(
                    model=patched_model,
                    prompt=prompt_tokens,
                    max_generated_tokens=max_generated_tokens,
                    pad_id=pad_id,
                    temperature=0.0,
                    top_k=1,
                    stop_tokens=pad_id,
                )

        assert generated_tokens_patched is not None

        assert torch.allclose(generated_tokens_original, generated_tokens_patched), (
            f"Patched model generation differs from original. \n"
            f"{generated_tokens_original.shape=}\n"
            f"{generated_tokens_patched.shape=}\n"
            f"{generated_tokens_original=}\n"
            f"{generated_tokens_patched=}\n"
            f"Params: batch_size={batch_size}, prompt_len={prompt_seq_len}, gen_len={max_generated_tokens}"
        )

        # Setup caches again after context
        torch.manual_seed(123)
        patched_model.eval()
        with torch.device("cuda"):
            patched_model.setup_caches(
                batch_size=batch_size,
                dtype=model_dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )
        with torch.no_grad():
            generated_tokens_after_context, _ = generation.generate(
                model=patched_model,
                prompt=prompt_tokens,
                max_generated_tokens=max_generated_tokens,
                pad_id=pad_id,
                temperature=0.0,
                top_k=1,
                stop_tokens=pad_id,
            )
        assert generated_tokens_after_context is not None

        assert torch.allclose(
            generated_tokens_original, generated_tokens_after_context
        ), (
            f"Restored model generation differs from original. "
            f"Params: batch_size={batch_size}, prompt_len={prompt_seq_len}, gen_len={max_generated_tokens}"
        )

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("use_kv_cache", [True])
    @pytest.mark.parametrize("prompt_seq_len", [50, 60])
    @pytest.mark.parametrize("queries_num", [10, 5])
    @pytest.mark.parametrize(
        "compression_perc",
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.7,
        ],
    )
    def test_output_no_equivalence_with_no_noop_strategy(
        self,
        model_base_config_fixture: dict,
        batch_size: int,
        use_kv_cache: bool,
        prompt_seq_len: int,
        queries_num: int,
        compression_perc: float,
    ):
        model_base_config = model_base_config_fixture

        assert use_kv_cache, "This test is designed for use_kv_cache=True"
        assert prompt_seq_len <= model_base_config["max_seq_len"], (
            f"prompt_seq_len ({prompt_seq_len}) > model.max_seq_len ({model_base_config['max_seq_len']})"
        )

        # Single forward prediction
        original_model = base_test_model(model_base_config, device="cuda")
        with torch.device("cuda"):
            original_model.setup_caches(
                batch_size=batch_size,
                dtype=original_model.tok_embeddings.weight.dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )

        current_device = "cuda"
        prompt_tokens = torch.randint(
            5,
            model_base_config["vocab_size"] // 2,
            (batch_size, prompt_seq_len),
            dtype=torch.long,
            device=current_device,
        )
        input_pos_tensor = torch.arange(
            0, prompt_seq_len, device=current_device, dtype=torch.long
        ).unsqueeze(0)
        attention_mask = get_attention_mask(
            batch_size=batch_size,
            q_seq_len=prompt_seq_len,
            kv_seq_len=model_base_config["max_seq_len"],
            device=current_device,
        )

        torch.manual_seed(123)
        original_model.eval()
        with torch.no_grad():
            output_original_singleforward = original_model(
                prompt_tokens,
                input_pos=input_pos_tensor,
                mask=attention_mask,
            )

        # Multiple forward prediction
        original_model = base_test_model(model_base_config, device="cuda")
        with torch.device("cuda"):
            original_model.setup_caches(
                batch_size=batch_size,
                dtype=original_model.tok_embeddings.weight.dtype,
                decoder_max_seq_len=model_base_config["max_seq_len"],
            )

        torch.manual_seed(123)
        original_model.eval()
        with torch.no_grad():
            output_original_multi1 = original_model(
                prompt_tokens[:, :-queries_num],
                input_pos=input_pos_tensor[:, :-queries_num],
                mask=attention_mask[:, :-queries_num],
            )
            output_original_multi2 = original_model(
                prompt_tokens[:, -queries_num:],
                input_pos=input_pos_tensor[:, -queries_num:],
                mask=attention_mask[:, -queries_num:],
            )
            output_original_multi = torch.cat(
                [output_original_multi1, output_original_multi2], dim=1
            )

        torch.testing.assert_close(
            output_original_singleforward,
            output_original_multi,
            msg=(
                f"Decomposed forward output differs from single forward. "
                f"Params: batch_size={batch_size}, use_kv_cache={use_kv_cache}, seq_len={prompt_seq_len}"
            ),
        )

        # Multiple forward prediction within no-op KV Compressor
        torch.manual_seed(123)

        patched_model = base_test_model(model_base_config, device="cuda")
        patched_model.eval()
        op_strategy = PressCompressionStrategy(
            press=RandomPress(),
        )
        kv_compressor = KVCompressor(
            model=patched_model,
            compression_strategies=[op_strategy],
        )
        with kv_compressor:
            with torch.device("cuda"):
                patched_model.setup_caches(
                    batch_size=batch_size,
                    dtype=original_model.tok_embeddings.weight.dtype,  # Match model's dtype
                    decoder_max_seq_len=model_base_config["max_seq_len"],
                )
            with torch.no_grad():
                output_patched_multi1 = patched_model(
                    prompt_tokens[:, :-queries_num],
                    input_pos=input_pos_tensor[:, :-queries_num],
                    mask=attention_mask[:, :-queries_num],
                )
                output_patched_multi2 = patched_model(
                    prompt_tokens[:, -queries_num:],
                    input_pos=input_pos_tensor[:, -queries_num:],
                    mask=attention_mask[:, -queries_num:],
                )
                output_patched_multi = torch.cat(
                    [output_patched_multi1, output_patched_multi2], dim=1
                )

        torch.testing.assert_close(
            output_original_singleforward,
            output_patched_multi,
            msg=(
                f"Patched output differs from original. "
                f"Params: batch_size={batch_size}, use_kv_cache={use_kv_cache}, seq_len={prompt_seq_len}"
            ),
        )

        # Multiple forward prediction while compressing KV Compressor
        torch.manual_seed(123)

        compressed_model = base_test_model(model_base_config, device="cuda")
        compressed_model.eval()
        op_strategy = PressCompressionStrategy(
            press=RandomPress(),
        )
        kv_compressor = KVCompressor(
            model=compressed_model,
            compression_strategies=[op_strategy],
        )
        with kv_compressor:
            with torch.device("cuda"):
                compressed_model.setup_caches(
                    batch_size=batch_size,
                    dtype=original_model.tok_embeddings.weight.dtype,  # Match model's dtype
                    decoder_max_seq_len=model_base_config["max_seq_len"],
                )
            with torch.no_grad():
                output_compressed_multi1 = compressed_model(
                    prompt_tokens[:, :-queries_num],
                    input_pos=input_pos_tensor[:, :-queries_num],
                    mask=attention_mask[:, :-queries_num],
                )
                kv_compressor.compress_caches(
                    int(prompt_tokens.shape[1] * compression_perc)
                )
                output_compressed_multi2 = compressed_model(
                    prompt_tokens[:, -queries_num:],
                    input_pos=input_pos_tensor[:, -queries_num:],
                    mask=attention_mask[:, -queries_num:],
                )
                output_compressed_multi = torch.cat(
                    [output_compressed_multi1, output_compressed_multi2], dim=1
                )

        # If equal: either the compression is not working, or we deleted entries that did not affect the forward
        assert not torch.allclose(
            output_original_singleforward, output_compressed_multi
        ), (
            f"Compressed output does not differ from original. "
            f"Params: batch_size={batch_size}, use_kv_cache={use_kv_cache}, seq_len={prompt_seq_len}"
        )
