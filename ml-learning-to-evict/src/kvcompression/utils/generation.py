#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Generation utilities for KV cache compression inference."""

from typing import List, Optional

import torch


def _sample_like_torchtune(
    logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample exactly like TorchTune's sample function.

    For temperature > 0, this uses the Gumbel-softmax trick to sample from the
    model's output distribution.
    For temperature = 0.0, this is a deterministic operation equivalent to
    greedy decoding (argmax).
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def generate_with_compression(
    model,
    prompt: torch.Tensor,
    kv_compressor,
    target_cache_size: int,
    max_generated_tokens: int,
    pad_id: int = 0,
    stop_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Generate tokens with optional KV cache compression.

    Uses a unified generation procedure that exactly matches TorchTune's generation pattern.
    When kv_compressor is None: Should be identical to torchtune.generate(temperature=0)
    When kv_compressor is provided: Uses the same logic but with compressed cache

    Args:
        model: The transformer model with KV caching enabled
        prompt: Input prompt tokens [batch_size, prompt_length]
        kv_compressor: KV cache compressor (None for no compression)
        target_cache_size: Target cache size after compression
        max_generated_tokens: Maximum number of tokens to generate
        pad_id: Padding token ID
        stop_tokens: List of token IDs that stop generation

    Returns:
        generated_tokens: Full sequence [batch_size, prompt_length + generated_length]
    """
    if prompt.shape[0] != 1:
        raise ValueError("Currently generation supports only batch size 1")

    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    batch_size, prompt_length = prompt.shape
    device = prompt.device

    max_seq_len = model.decoder_max_cache_seq_len
    total_response_length = prompt_length + max_generated_tokens

    if total_response_length > max_seq_len:
        raise ValueError(
            f"Sequence length ({total_response_length}) exceeds "
            f"model's max sequence length ({max_seq_len})."
        )

    generated_tokens = prompt.clone()

    masks = torch.tril(
        torch.ones(
            total_response_length,
            max_seq_len,
            dtype=torch.bool,
            device=device,
        )
    ).unsqueeze(0)  # [1, total_response_length, max_seq_len]

    input_pos = torch.arange(0, total_response_length, device=device).unsqueeze(
        0
    )  # [1, total_response_length]

    stop_tokens_tensor = (
        torch.tensor(stop_tokens, device=device, dtype=prompt.dtype)
        if stop_tokens is not None
        else None
    )
    stop_token_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # Prefill
    curr_masks = masks[:, :prompt_length]
    curr_input_pos = input_pos[:, :prompt_length]
    logits = model(prompt, input_pos=curr_input_pos, mask=curr_masks)

    # Compress cache after prefill, before autoregressive generation
    if kv_compressor is not None:
        kv_compressor.compress_caches(target_cache_size)

    first_token = _sample_like_torchtune(logits[:, -1, :], temperature=0.0)
    generated_tokens = torch.cat([generated_tokens, first_token], dim=1)

    if stop_tokens_tensor is not None:
        stop_token_reached |= torch.isin(first_token.squeeze(-1), stop_tokens_tensor)
        if torch.all(stop_token_reached):
            return generated_tokens

    # Autoregressive generation
    curr_pos = prompt_length

    for _ in range(max_generated_tokens - 1):
        curr_masks = masks[:, curr_pos, None, :]
        curr_input_pos = input_pos[:, curr_pos]
        last_token = generated_tokens[:, [-1]]
        logits = model(last_token, input_pos=curr_input_pos, mask=curr_masks)
        next_token = _sample_like_torchtune(logits[:, -1, :], temperature=0.0)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
        curr_pos += 1

        if stop_tokens_tensor is not None:
            stop_token_reached |= torch.isin(next_token.squeeze(-1), stop_tokens_tensor)
            if torch.all(stop_token_reached):
                break

        if generated_tokens.shape[1] >= max_seq_len:
            break

    return generated_tokens
