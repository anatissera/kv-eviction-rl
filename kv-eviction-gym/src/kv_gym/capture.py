"""
Prefill capture: run one LLM forward pass on a GSM8K prompt and extract
K and V tensors for all (layer, kv_head) pairs.

No hooks, no output_attentions, no reference decode.
Compatible with any attention backend (eager, sdpa, flash_attention_2).

The stored K/V already have RoPE applied (they come directly from
past_key_values after the prefill). They are kept on CPU to avoid
holding GPU memory across the episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from kv_gym.vendor.prompts import format_gsm8k


@dataclass
class AllHeadCapture:
    K:          Tensor  # [n_layers, n_kv_heads, prompt_len, head_dim]  (RoPE applied)
    V:          Tensor  # [n_layers, n_kv_heads, prompt_len, head_dim]
    input_ids:  Tensor  # [1, prompt_len]  — for terminal generate
    prompt_len: int
    gold_answer: str


def capture(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
) -> AllHeadCapture:
    """Prefill the model on one GSM8K example and return K/V for all heads.

    Args:
        model:     HuggingFace CausalLM (any attn_implementation).
        tokenizer: Matching tokenizer.
        example:   GSM8K dict with "prompt_text" and "gold_answers".
        device:    Model device.

    Returns:
        AllHeadCapture with K, V on CPU.
    """
    prompt_text, _ = format_gsm8k(example)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    n_layers   = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads",
                         model.config.num_attention_heads)

    model.eval()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)

    past_kv = out.past_key_values

    def _get_kv(layer_idx: int):
        if hasattr(past_kv, "layers"):
            # transformers >= 4.47 DynamicCache
            return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
        elif hasattr(past_kv, "key_cache"):
            # transformers 4.40-4.46
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
        else:
            # legacy tuple-of-tuples
            return past_kv[layer_idx][0], past_kv[layer_idx][1]

    K_list, V_list = [], []
    for l in range(n_layers):
        k, v = _get_kv(l)
        K_list.append(k.squeeze(0).cpu())  # [n_kv_heads, T, D]
        V_list.append(v.squeeze(0).cpu())

    return AllHeadCapture(
        K=torch.stack(K_list),            # [L, n_kv_heads, T, D]
        V=torch.stack(V_list),
        input_ids=inputs["input_ids"].cpu(),
        prompt_len=prompt_len,
        gold_answer=example["gold_answers"][0],
    )
