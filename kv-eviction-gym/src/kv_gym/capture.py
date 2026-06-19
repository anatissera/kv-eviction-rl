"""
Run one LLM prefill+decode on a GSM8K example and capture all per-head tensors.

Memory-efficient strategy (adapted from internal-signals-context-compression/
src/scoring/hook_capture.py):

  For each attention layer, register a pre-hook that injects
  output_attentions=True into THAT LAYER's kwargs only, and a post-hook
  that immediately reads output[1] (the [1,H,Q,K] attention matrix),
  reduces it to [H, K], and sets output[1] = None so the full matrix is
  freed before the next layer runs.

  Peak memory = model weights + KV cache + ONE layer's attention matrix,
  regardless of the number of layers.

Two passes:

  Pass 1 (prefill, use_cache=True):
    - Q: captured from q_proj(hidden_states) before RoPE.
      NOTE: this is a mild approximation for the env attn recompute feature.
      RoPE affects absolute magnitudes but not relative orderings, so the
      policy can still learn from these scores.
    - K, V: read from past_key_values after the forward pass.
      These have RoPE applied (correct for cache use).
    - attn_score (initial feature): per-head column sums of the softmax
      attention from the prefill, captured via the post-hook.

  Pass 2 (greedy decode, max_new_tokens steps):
    - future_attn: accumulated per decode step.
      Each step the attention hook fires for each layer with a [1,H,1,T+s]
      matrix. We take [:,:,0,:prompt_len], immediately add [H,prompt_len]
      to the accumulator, then free.
      This uses the actual softmax weights with RoPE — correct.

Source: /Users/alexanderbodner/Documents/Udesa/5to/tesis/internal-signals-context-compression
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kv_gym.vendor.prompts import format_gsm8k


@dataclass
class AllHeadCapture:
    Q:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]  (no RoPE — approx)
    K:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]  (with RoPE — correct)
    V:            Tensor  # [n_layers, n_heads, prompt_len, head_dim]
    attn_score:   Tensor  # [n_layers, n_heads, prompt_len]  — prefill column-sum attention
    future_attn:  Tensor  # [n_layers, n_heads, prompt_len]  — decode attention, normalized
    prompt_len:   int
    gold_answer:  str


# ---------------------------------------------------------------------------
# Pre-hook: inject output_attentions=True for one layer at a time
# ---------------------------------------------------------------------------

def _inject_output_attentions(
    module: torch.nn.Module,
    args: tuple,
    kwargs: dict,
) -> tuple[tuple, dict]:
    kwargs["output_attentions"] = True
    return args, kwargs


# ---------------------------------------------------------------------------
# Main capture function
# ---------------------------------------------------------------------------

def capture(
    model,
    tokenizer,
    example: dict,
    device: torch.device,
    max_new_tokens: int = 64,
) -> AllHeadCapture:
    """Capture Q/K/V, prefill attn scores, and future_attn for all heads.

    Args:
        model:          HuggingFace CausalLM with attn_implementation="eager".
        tokenizer:      Matching tokenizer.
        example:        GSM8K dict with "prompt_text" and "gold_answers".
        device:         Model device.
        max_new_tokens: How many decode steps to accumulate future_attn from.

    Returns:
        AllHeadCapture with all tensors on CPU.
    """
    prompt_text, _ = format_gsm8k(example)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    n_layers    = model.config.num_hidden_layers
    n_heads     = model.config.num_attention_heads
    n_kv_heads  = getattr(model.config, "num_key_value_heads", n_heads)
    head_dim    = model.config.hidden_size // n_heads

    # ------------------------------------------------------------------ #
    # Pass 1: prefill                                                     #
    # Captures Q (no RoPE), K/V (from past_key_values, with RoPE),       #
    # and per-head prefill attention column sums.                         #
    # ------------------------------------------------------------------ #

    captured_q: dict[int, Tensor] = {}    # layer_idx → [H, T, D]  (no RoPE)
    prefill_attn: dict[int, Tensor] = {}  # layer_idx → [H, T]  (col-sum softmax)

    handles: list[Any] = []

    def make_q_hook(layer_idx: int):
        """Post-hook: capture Q from q_proj, and attention col-sums."""
        def hook(module, args, kwargs, output):
            with torch.no_grad():
                h = args[0] if args else kwargs.get("hidden_states")
                q = module.q_proj(h)           # [1, T, H*D]
                bsz, seq, _ = q.shape
                # Use config-level head counts (Qwen2Attention has no num_heads attr)
                q = q.view(bsz, seq, n_heads, head_dim).squeeze(0).permute(1, 0, 2)  # [H, T, D]
                captured_q[layer_idx] = q.cpu()

                # Capture per-head attention column sums from output[1]
                # output[1]: [1, H, T, T]  (injected by pre-hook)
                attn_weights = output[1]
                if attn_weights is not None:
                    # sum over query dim → [H, T]  (how much each key is attended to)
                    col_sum = attn_weights.squeeze(0).sum(dim=1)  # [H, T]
                    prefill_attn[layer_idx] = col_sum.cpu()

            # Free the full attention matrix immediately
            return (output[0], None) + output[2:]
        return hook

    for i, layer in enumerate(model.model.layers):
        h_pre = layer.self_attn.register_forward_pre_hook(
            _inject_output_attentions, with_kwargs=True
        )
        h_post = layer.self_attn.register_forward_hook(make_q_hook(i), with_kwargs=True)
        handles += [h_pre, h_post]

    model.eval()
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)

    for h in handles:
        h.remove()
    handles.clear()

    # Read K and V from past_key_values (RoPE already applied)
    # past_key_values[layer_idx] is a tuple (k, v) each [1, n_kv_heads, T, D]
    past_kv = prefill_out.past_key_values
    # Transformers >= 4.47 uses DynamicCache with .layers[i].keys / .values
    # Transformers 4.40-4.46 used .key_cache / .value_cache lists
    # Older versions returned tuple-of-tuples
    def _get_kv(layer_idx):
        if hasattr(past_kv, "layers"):
            # New API: DynamicCache with CacheLayer objects
            return past_kv.layers[layer_idx].keys, past_kv.layers[layer_idx].values
        elif hasattr(past_kv, "key_cache"):
            return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
        else:
            return past_kv[layer_idx][0], past_kv[layer_idx][1]

    K_list, V_list = [], []
    for layer_idx in range(n_layers):
        k, v = _get_kv(layer_idx)
        k = k.squeeze(0)  # [n_kv_heads, T, D]
        v = v.squeeze(0)  # [n_kv_heads, T, D]
        if n_kv_heads != n_heads:
            repeats = n_heads // n_kv_heads
            k = k.repeat_interleave(repeats, dim=0)  # [H, T, D]
            v = v.repeat_interleave(repeats, dim=0)
        K_list.append(k.cpu())
        V_list.append(v.cpu())

    Q_all    = torch.stack([captured_q[i]    for i in range(n_layers)])  # [L, H, T, D]
    K_all    = torch.stack(K_list)                                        # [L, H, T, D]
    V_all    = torch.stack(V_list)
    attn_all = torch.stack([prefill_attn.get(i, torch.zeros(n_heads, prompt_len))
                             for i in range(n_layers)])                   # [L, H, T]

    # ------------------------------------------------------------------ #
    # Pass 2: greedy decode → accumulate future_attn per step             #
    # Each step's attention is [1, H, 1, cache_len]. We take              #
    # [:, :, 0, :prompt_len] → [H, prompt_len] and accumulate.           #
    # Only one layer's [H, 1, T] lives in memory at a time.              #
    # ------------------------------------------------------------------ #

    future_attn = torch.zeros(n_layers, n_heads, prompt_len)

    def make_future_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            attn_weights = output[1]  # [1, H, 1, cache_len]  or None
            if attn_weights is not None and attn_weights.shape[2] == 1:
                # Decode step (query_len == 1). Take attention over prompt portion.
                step_attn = attn_weights[0, :, 0, :prompt_len]  # [H, prompt_len]
                future_attn[layer_idx].add_(step_attn.cpu())
            # Free immediately
            return (output[0], None) + output[2:]
        return hook

    for i, layer in enumerate(model.model.layers):
        h_pre = layer.self_attn.register_forward_pre_hook(
            _inject_output_attentions, with_kwargs=True
        )
        h_post = layer.self_attn.register_forward_hook(make_future_hook(i), with_kwargs=True)
        handles += [h_pre, h_post]

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            past_key_values=None,  # fresh decode from prompt (no reuse of pass 1 cache)
        )

    for h in handles:
        h.remove()

    # Normalize future_attn: each head sums to 1 (makes AUC reward scale-invariant)
    row_sum = future_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    future_attn = future_attn / row_sum

    return AllHeadCapture(
        Q=Q_all,
        K=K_all,
        V=V_all,
        attn_score=attn_all,
        future_attn=future_attn,
        prompt_len=prompt_len,
        gold_answer=example["gold_answers"][0],
    )
