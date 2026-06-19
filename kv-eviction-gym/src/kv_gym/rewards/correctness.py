"""
Phase 2 reward: GSM8K answer correctness.

After the episode ends (cache evicted to budget), we:
1. Apply the per-head eviction decisions to the HuggingFace DynamicCache.
2. Run greedy generation from the evicted cache.
3. Score the output against the gold answer.

This reward requires a second LLM call per episode (expensive) but gives
a direct signal about whether the kept tokens preserve reasoning ability.
Use it after Phase 1 pre-training has converged.

Note: in the sequential gym we evict tokens per (layer, head) independently.
For the correctness reward we need a single evicted cache. We use the
intersection strategy: a token is kept in layer L only if a majority of
heads in that layer voted to keep it. This is a simple heuristic — a
learnable aggregation can replace it later.
"""

import torch
from torch import Tensor

from kv_gym.vendor.answer_extraction_gsm8k import score as gsm8k_score
from kv_gym.vendor.prompts import format_gsm8k


def gsm8k_correctness(
    model,
    tokenizer,
    example:  dict,
    device:   torch.device,
    resident: Tensor,   # [n_layers, n_heads, prompt_len]  bool
    budget:   int,
) -> float:
    """Run evicted-cache generation and score against the gold answer.

    Args:
        model:     HuggingFace CausalLM (same model used at episode reset).
        tokenizer: Matching tokenizer.
        example:   Original GSM8K example dict (for prompt and gold answer).
        device:    Model device.
        resident:  Final resident mask from the episode.
        budget:    Number of tokens kept per head (for verification).

    Returns:
        flexible_extract score ∈ {0.0, 1.0}.
    """
    from transformers import DynamicCache

    prompt_text, max_new_tokens = format_gsm8k(example)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    # Build evicted cache: re-run prefill, then slice to kept positions.
    model.eval()
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
        cache = prefill_out.past_key_values

        # Per-layer keep indices: majority vote across heads.
        # resident: [L, H, T]  → majority: [L, T]
        keep_mask = _majority_vote(resident)  # [L, T]  bool
        keep_indices = _mask_to_indices(keep_mask, budget)  # [L, budget]

        # Trim the cache in place.
        for layer_idx in range(len(cache)):
            idx = keep_indices[layer_idx].to(device)
            k, v = cache[layer_idx]
            cache.key_cache[layer_idx]   = k.index_select(dim=2, index=idx)
            cache.value_cache[layer_idx] = v.index_select(dim=2, index=idx)

        # Generate from the evicted cache.
        position_ids = torch.arange(
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[1] + 1,
            device=device,
        ).unsqueeze(0)

        gen_out = model.generate(
            input_ids=inputs["input_ids"][:, -1:],
            attention_mask=inputs["attention_mask"],
            past_key_values=cache,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prediction = tokenizer.decode(gen_out[0], skip_special_tokens=True)
    result = gsm8k_score(prediction, example["gold_answers"])
    return result["flexible_extract"]


def _majority_vote(resident: Tensor) -> Tensor:
    """Return True for positions where > half the heads voted to keep."""
    # resident: [L, H, T]
    return resident.float().mean(dim=1) > 0.5   # [L, T]


def _mask_to_indices(mask: Tensor, budget: int) -> Tensor:
    """Convert per-layer boolean mask to per-layer keep indices [L, budget].

    If a layer has fewer than `budget` True entries, pad with 0 (harmless
    since masked-out positions don't affect generation output meaningfully).
    """
    L, T = mask.shape
    indices = torch.zeros(L, budget, dtype=torch.long)
    for l in range(L):
        idx = mask[l].nonzero(as_tuple=False).squeeze(1)
        k = min(len(idx), budget)
        indices[l, :k] = idx[:k]
    return indices
