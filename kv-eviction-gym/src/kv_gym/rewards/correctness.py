"""
GSM8K correctness scoring (terminal reward).

This module is now thin — the generate + score logic lives in
SharedKVVecEnv._terminal_reward() so it has direct access to the model,
tokenizer, and input_ids without re-running the prefill.

This file keeps the standalone helper for eval scripts and unit tests.
"""

import torch
from kv_gym.vendor.answer_extraction_gsm8k import flexible_extract
from kv_gym.vendor.prompts import format_gsm8k


def score_evicted(
    model,
    tokenizer,
    input_ids:    torch.Tensor,   # [1, prompt_len]
    resident:     torch.Tensor,   # [n_layers, n_kv_heads, prompt_len] bool
    budget:       int,
    gold_answer:  str,
    device:       torch.device,
    max_new_tokens: int = 512,
) -> float:
    """Aggregate per-head eviction decisions and score generation correctness.

    Keeps the top-budget tokens by mean keep-score across all heads,
    runs generate with an attention mask that zeros out the rest.

    Returns flexible_extract score in {0.0, 1.0}.
    """
    T = input_ids.shape[1]

    token_scores = resident.float().mean(dim=(0, 1))    # [T]
    n_keep = min(budget, T)
    _, topk = token_scores.topk(n_keep)

    attn_mask = torch.zeros(1, T, dtype=torch.long, device=device)
    attn_mask[0, topk] = 1

    # Explicit position_ids so surviving tokens keep their original RoPE rotations.
    # Without this, HF derives positions via cumsum(attn_mask)-1, shifting all
    # positions after each masked gap.
    position_ids = torch.arange(T, device=device).unsqueeze(0)  # [1, T]

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attn_mask,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(out[0, T:], skip_special_tokens=True)
    return flexible_extract(text, [gold_answer])
