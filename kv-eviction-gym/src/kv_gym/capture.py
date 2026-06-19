"""
Run one LLM prefill+decode on a GSM8K example and capture all per-head tensors.

One call to `capture()` yields Q, K, V, and future-attention weights for
every (layer, head) pair simultaneously. The SharedKVVecEnv calls this
once per episode reset to feed all 56 environments.

We use attn_implementation="eager" because SDPA and FlashAttention do not
expose attention weights via output_attentions=True.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from kv_gym.vendor.prompts import format_gsm8k


@dataclass
class AllHeadCapture:
    Q:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    K:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    V:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    future_attn:  Tensor  # [n_layers, n_heads, prompt_len]  — attention mass from
                          # generated tokens back onto each prompt token
    prompt_len:   int
    gold_answer:  str


def capture(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
    max_new_tokens: int = 64,
) -> AllHeadCapture:
    """Prefill + greedy decode with output_attentions=True.

    Args:
        model:          HuggingFace CausalLM, loaded with attn_implementation="eager".
        tokenizer:      Matching tokenizer.
        example:        Dict with "prompt_text" and "gold_answers" keys (GSM8K schema).
        device:         Device the model lives on.
        max_new_tokens: Number of generation tokens to collect attention from.
                        More tokens → better future_attn signal, but slower.

    Returns:
        AllHeadCapture with tensors on `device`.
    """
    prompt_text, _ = format_gsm8k(example)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    # out.attentions is a tuple over generation steps.
    # Each step is a tuple over layers.
    # Each layer is [batch, n_heads, query_len, key_len].
    #
    # Prefill step (index 0): query_len == prompt_len, key_len == prompt_len.
    # Generation steps (index 1+): query_len == 1, key_len grows by 1 per step.
    #
    # We extract Q, K, V from the prefill step via hooks, and future_attn from
    # the generation steps (how much each new token attends to each prompt token).

    prefill_attns = out.attentions[0]   # tuple[n_layers] of [1, H, T, T]
    gen_attns     = out.attentions[1:]  # tuple[n_new] of tuple[n_layers] of [1, H, 1, T+step]

    n_layers = len(prefill_attns)

    # --- Extract Q, K, V via a second forward pass with hooks ---
    # We re-run only the prefill (no generation) to get Q/K/V tensors.
    # This is slightly redundant but keeps the capture logic simple and
    # avoids patching model internals.

    captured: dict[int, dict[str, Tensor]] = {}

    def make_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            # Qwen2Attention.forward receives (hidden_states, attention_mask, ...)
            # and internally computes Q, K, V. We intercept just after the
            # projection by registering on the attention module itself.
            # args[0] is hidden_states; we use the module's q/k/v_proj directly.
            with torch.no_grad():
                h = args[0]                        # [1, T, hidden]
                q = module.q_proj(h)               # [1, T, n_heads * head_dim]
                k = module.k_proj(h)               # [1, T, n_kv_heads * head_dim]
                v = module.v_proj(h)               # [1, T, n_kv_heads * head_dim]

                bsz, seq, _ = q.shape
                H   = module.num_heads
                KVH = module.num_key_value_heads
                D   = module.head_dim

                q = q.view(bsz, seq, H,   D).squeeze(0).permute(1, 0, 2)   # [H, T, D]
                k = k.view(bsz, seq, KVH, D).squeeze(0).permute(1, 0, 2)   # [KVH, T, D]
                v = v.view(bsz, seq, KVH, D).squeeze(0).permute(1, 0, 2)   # [KVH, T, D]

                # For GQA (grouped query attention), repeat K/V to match H.
                if KVH != H:
                    repeats = H // KVH
                    k = k.repeat_interleave(repeats, dim=0)
                    v = v.repeat_interleave(repeats, dim=0)

                captured[layer_idx] = {"Q": q.cpu(), "K": k.cpu(), "V": v.cpu()}
        return hook

    hooks = []
    for i, layer in enumerate(model.model.layers):
        h = layer.self_attn.register_forward_hook(make_hook(i), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    Q_all = torch.stack([captured[i]["Q"] for i in range(n_layers)])  # [L, H, T, D]
    K_all = torch.stack([captured[i]["K"] for i in range(n_layers)])  # [L, H, T, D]
    V_all = torch.stack([captured[i]["V"] for i in range(n_layers)])  # [L, H, T, D]

    # --- Build future_attn: how much do generated tokens attend to each prompt token? ---
    # For each generation step s and layer l, gen_attns[s][l] is [1, H, 1, T+s+1].
    # We take the attention over the prompt portion (first prompt_len positions)
    # and sum across all generation steps → proxy for token importance.

    n_heads = Q_all.shape[1]
    future_attn = torch.zeros(n_layers, n_heads, prompt_len)

    for step_attns in gen_attns:
        for layer_idx, layer_attn in enumerate(step_attns):
            # layer_attn: [1, H, 1, key_len]  — key_len = prompt_len + step
            attn = layer_attn[0, :, 0, :prompt_len]  # [H, prompt_len]
            future_attn[layer_idx] += attn.cpu()

    # Normalize so values sum to 1 per head (makes AUC reward scale-invariant).
    row_sum = future_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    future_attn = future_attn / row_sum

    return AllHeadCapture(
        Q=Q_all,
        K=K_all,
        V=V_all,
        future_attn=future_attn,
        prompt_len=prompt_len,
        gold_answer=example["gold_answers"][0],
    )
