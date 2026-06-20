"""
Per-layer KV-cache eviction followed by greedy generation.

Adapted from src/eviction/apply_eviction.py and
src/inference/generate_with_eviction.py in the internal-signals repo.

The approach
------------
model.generate() only accepts ONE global attention_mask applied identically
to all layers.  To apply independent per-layer eviction decisions we instead
manipulate the DynamicCache directly:

1. Run a prefill forward pass to get the full KV cache.
2. For each layer l, keep only the positions where resident_mask[l, t] is True
   by slicing layer.keys and layer.values with index_select (dim=2 = seq dim).
3. Run a manual greedy decode loop, passing explicit position_ids and
   cache_position so the new token's RoPE rotation uses its true original
   position — not the (smaller) trimmed cache length that HuggingFace would
   otherwise infer.

Why cache slicing is exact
--------------------------
RoPE is baked into the K values when they first enter the cache (rotation
applied per position at prefill time).  Slicing leaves those rotations
untouched.  The new token's Q uses its true position via the explicit
position_ids we pass.  The attention logit Q[T] · K[i] is therefore correct
for every kept position i, regardless of where i now sits in the trimmed cache.

Requires transformers >= 5.0 (uses cache.layers[l].keys/values API).
"""

from __future__ import annotations

import torch
from torch import Tensor


def apply_per_layer_eviction(cache, resident_mask: Tensor) -> None:
    """Trim each layer of the DynamicCache to its own kept positions, in place.

    Args:
        cache:         DynamicCache from a HuggingFace prefill forward pass.
        resident_mask: Bool tensor [L, T]. resident_mask[l, t] = True means
                       keep token t in layer l's cache.

    Handles transformers 5.x (cache.layers) and 4.x (cache.key_cache).
    """
    L = resident_mask.shape[0]

    if hasattr(cache, "layers"):
        # transformers >= 5.0
        for l in range(L):
            idx = resident_mask[l].nonzero(as_tuple=True)[0].to(cache.layers[l].keys.device)
            cache.layers[l].keys   = cache.layers[l].keys.index_select(dim=2, index=idx)
            cache.layers[l].values = cache.layers[l].values.index_select(dim=2, index=idx)
    elif hasattr(cache, "key_cache"):
        # transformers 4.x DynamicCache
        for l in range(L):
            idx = resident_mask[l].nonzero(as_tuple=True)[0].to(cache.key_cache[l].device)
            cache.key_cache[l]   = cache.key_cache[l].index_select(dim=2, index=idx)
            cache.value_cache[l] = cache.value_cache[l].index_select(dim=2, index=idx)
    else:
        raise RuntimeError(
            "Unsupported past_key_values format: expected DynamicCache with "
            "'layers' (transformers>=5) or 'key_cache' (transformers 4.x)."
        )


@torch.no_grad()
def generate_with_per_layer_eviction(
    model,
    tokenizer,
    input_ids: Tensor,          # [1, T]
    resident_mask: Tensor,      # [L, T] bool
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Prefill → per-layer eviction → greedy decode.  Returns decoded new tokens.

    Args:
        model:          Frozen causal LM.
        tokenizer:      Matching tokenizer (for EOS id).
        input_ids:      [1, T] prompt token ids.
        resident_mask:  [L, T] bool — True = keep this position in this layer.
        max_new_tokens: Maximum generation steps.
        device:         Device for tensor ops.

    Returns:
        Decoded string of the generated tokens (prompt excluded).
    """
    input_ids  = input_ids.to(device)
    prompt_len = input_ids.shape[1]

    # --- Prefill ---
    prefill_out = model(input_ids=input_ids, use_cache=True)
    past_kv     = prefill_out.past_key_values

    # --- Per-layer cache eviction ---
    apply_per_layer_eviction(past_kv, resident_mask)

    # --- Greedy decode ---
    # The first predicted token comes from the prefill logits (position prompt_len-1
    # → next token is at prompt_len in the original sequence).
    next_token    = int(prefill_out.logits[0, -1].argmax().item())
    eos_id        = tokenizer.eos_token_id
    generated: list[int] = []
    true_position = prompt_len   # original sequence position of the next token

    for _ in range(max_new_tokens):
        # Check EOS before appending so we never feed EOS back as input_ids
        # and never include it in the decoded text.
        if eos_id is not None and next_token == eos_id:
            break
        generated.append(next_token)

        # position_ids: the new token's TRUE position in the original sequence.
        # Without this, HuggingFace infers position from past_kv.get_seq_length()
        # which is the EVICTED cache size (budget), not the true position — wrong RoPE.
        pos = torch.tensor([[true_position]], device=device)

        step_out = model(
            input_ids=torch.tensor([[next_token]], device=device),
            past_key_values=past_kv,
            position_ids=pos,
            cache_position=pos.squeeze(0),  # 1D: needed by transformers 5.x causal mask
            use_cache=True,
        )

        next_token    = int(step_out.logits[0, -1].argmax().item())
        past_kv       = step_out.past_key_values
        true_position += 1

    return tokenizer.decode(generated, skip_special_tokens=True)
