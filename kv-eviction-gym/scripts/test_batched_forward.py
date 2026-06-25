"""
Verify that batched forward passes (batch>1) are numerically equivalent
to sequential single-batch forward passes.

Tests:
  1. Prefill: batch=2 with identical inputs == 2x batch=1
  2. Decode step: batched next-token matches sequential
  3. Eviction via attention_mask: masked batch == sequential after _evict_slot

Run with:
  uv run python scripts/test_batched_forward.py
"""

import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from kv_gym.env import _evict_slot, _get_layer_kv

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATOL = 1e-2   # fp32 on GPU — allow small numerical error

def load_model():
    print(f"Loading {MODEL} on {DEVICE}...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa"
    ).to(DEVICE).eval()
    return tok, model

def prefill(model, input_ids):
    """Run prefill, return (logits, past_kv)."""
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    return out.logits, out.past_key_values

def decode_step(model, next_tok, past_kv, position):
    pos = torch.tensor([[position]], device=DEVICE)
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([[next_tok]], device=DEVICE),
            past_key_values=past_kv,
            position_ids=pos,
            cache_position=pos.squeeze(0),
            use_cache=True,
        )
    return out.logits, out.past_key_values

def _n_layers(cache):
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache.key_cache)

def _get_kv(cache, l):
    """Return (K, V) tensors [1, H, S, D] for layer l."""
    if hasattr(cache, "layers"):
        return cache.layers[l].keys, cache.layers[l].values
    return cache.key_cache[l], cache.value_cache[l]

def _set_kv(cache, l, K, V):
    if hasattr(cache, "layers"):
        cache.layers[l].keys   = K
        cache.layers[l].values = V
    else:
        cache.key_cache[l]   = K
        cache.value_cache[l] = V

def clone_cache(cache):
    new = DynamicCache()
    # Force same internal structure by copying layer-by-layer
    for l in range(_n_layers(cache)):
        K, V = _get_kv(cache, l)
        K2 = K.clone(); V2 = V.clone()
        if hasattr(cache, "layers"):
            import copy
            new.layers.append(copy.copy(cache.layers[l]))
            new.layers[l].keys   = K2
            new.layers[l].values = V2
        else:
            new.key_cache.append(K2)
            new.value_cache.append(V2)
    return new

def cache_to_batched(cache_a, cache_b):
    """Stack two single-sequence caches into one batch=2 cache."""
    batched = DynamicCache()
    for l in range(_n_layers(cache_a)):
        Ka, Va = _get_kv(cache_a, l)
        Kb, Vb = _get_kv(cache_b, l)
        K_bat = torch.cat([Ka, Kb], dim=0)
        V_bat = torch.cat([Va, Vb], dim=0)
        if hasattr(cache_a, "layers"):
            import copy
            batched.layers.append(copy.copy(cache_a.layers[l]))
            batched.layers[l].keys   = K_bat
            batched.layers[l].values = V_bat
        else:
            batched.key_cache.append(K_bat)
            batched.value_cache.append(V_bat)
    return batched

def check(name, a, b, atol=ATOL):
    a = a.float().cpu()
    b = b.float().cpu()
    max_diff = (a - b).abs().max().item()
    ok = max_diff < atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}  max_diff={max_diff:.2e}")
    return ok

def main():
    tok, model = load_model()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {n_layers} layers, device={DEVICE}\n")

    prompt_a = "What is 2 + 2?"
    prompt_b = "What is 3 + 3?"

    ids_a = tok(prompt_a, return_tensors="pt").input_ids.to(DEVICE)
    ids_b = tok(prompt_b, return_tensors="pt").input_ids.to(DEVICE)

    # Pad to same length for batching
    L = max(ids_a.shape[1], ids_b.shape[1])
    def pad(ids):
        if ids.shape[1] < L:
            pad_tok = tok.pad_token_id or tok.eos_token_id
            ids = torch.cat([
                torch.full((1, L - ids.shape[1]), pad_tok, device=DEVICE), ids
            ], dim=1)
        return ids
    ids_a_pad = pad(ids_a)
    ids_b_pad = pad(ids_b)

    print("=" * 60)
    print("TEST 1: Prefill — batch=2 vs 2x batch=1")
    print("=" * 60)

    logits_a1, kv_a1 = prefill(model, ids_a_pad)
    logits_b1, kv_b1 = prefill(model, ids_b_pad)

    ids_batch = torch.cat([ids_a_pad, ids_b_pad], dim=0)
    # attention_mask: both sequences are same length, all 1s
    attn_mask = torch.ones(2, L, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        out_batch = model(input_ids=ids_batch, attention_mask=attn_mask, use_cache=True)
    logits_batch = out_batch.logits
    kv_batch = out_batch.past_key_values

    all_pass = True
    all_pass &= check("logits[0] batch==seq_a", logits_batch[0], logits_a1[0])
    all_pass &= check("logits[1] batch==seq_b", logits_batch[1], logits_b1[0])

    # Check KV cache layer 0
    K_bat, _ = _get_kv(kv_batch, 0)   # [2, H, S, D]
    ka_batch = K_bat[0]
    kb_batch = K_bat[1]
    Ka_seq, _ = _get_kv(kv_a1, 0)
    Kb_seq, _ = _get_kv(kv_b1, 0)
    ka_seq = Ka_seq[0]
    kb_seq = Kb_seq[0]
    all_pass &= check("KV cache layer0 seq_a", ka_batch, ka_seq)
    all_pass &= check("KV cache layer0 seq_b", kb_batch, kb_seq)

    print()
    print("=" * 60)
    print("TEST 2: Decode step — batched vs sequential")
    print("=" * 60)

    next_a = int(logits_a1[0, -1].argmax())
    next_b = int(logits_b1[0, -1].argmax())
    pos = L  # next position

    # Sequential
    logits_a2_seq, kv_a2_seq = decode_step(model, next_a, clone_cache(kv_a1), pos)
    logits_b2_seq, kv_b2_seq = decode_step(model, next_b, clone_cache(kv_b1), pos)

    # Batched
    kv_bat = cache_to_batched(kv_a1, kv_b1)
    next_batch = torch.tensor([[next_a], [next_b]], device=DEVICE)
    pos_batch  = torch.tensor([[pos], [pos]], device=DEVICE)
    attn_mask2 = torch.ones(2, L + 1, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        out2 = model(
            input_ids=next_batch,
            past_key_values=kv_bat,
            position_ids=pos_batch,
            attention_mask=attn_mask2,
            use_cache=True,
        )
    logits_a2_bat = out2.logits[0:1]
    logits_b2_bat = out2.logits[1:2]

    all_pass &= check("decode logits[0]", logits_a2_bat, logits_a2_seq)
    all_pass &= check("decode logits[1]", logits_b2_bat, logits_b2_seq)

    print()
    print("=" * 60)
    print("TEST 3: Eviction via attention_mask vs _evict_slot")
    print("=" * 60)
    # Evict slot 2 from seq_a, layer 0, then do a decode step.
    # Compare: (a) _evict_slot on sequential cache, (b) attention_mask=0 at pos 2 on batched cache.

    EVICT_SLOT = 2

    # Sequential: physically remove slot 2 from all layers
    kv_a_evicted = clone_cache(kv_a1)
    for l in range(n_layers):
        _evict_slot(kv_a_evicted, l, EVICT_SLOT)

    logits_a_evict_seq, _ = decode_step(model, next_a, kv_a_evicted, pos)

    # Batched: mask out slot 2 for seq_a only
    # attention_mask shape: [2, L] for the KV cache + [2, 1] for the new token = [2, L+1]
    kv_bat2 = cache_to_batched(kv_a1, kv_b1)
    attn_mask_evict = torch.ones(2, L + 1, device=DEVICE, dtype=torch.long)
    attn_mask_evict[0, EVICT_SLOT] = 0   # evict slot 2 for seq_a only

    with torch.no_grad():
        out3 = model(
            input_ids=next_batch,
            past_key_values=kv_bat2,
            position_ids=pos_batch,
            attention_mask=attn_mask_evict,
            use_cache=True,
        )
    logits_a_evict_bat = out3.logits[0:1]

    all_pass &= check("evict: mask==physical (seq_a)", logits_a_evict_bat, logits_a_evict_seq)

    print()
    print("=" * 60)
    print("TEST 4: Per-layer batched eviction — different slot per (batch, layer)")
    print("=" * 60)
    # This is the real use case: each layer independently evicts a different slot,
    # and we want to do it for a batch of B episodes simultaneously.
    #
    # Ground truth: B sequential single-episode forward passes, each with
    #   _evict_slot(cache, l, slot_b_l) for every layer l.
    # Batched:      one forward pass with batch=B, where we physically remove
    #   slot_b_l from cache.key_cache[l][b] before the forward pass.

    B = 2
    # Random but deterministic per-layer eviction slots for each episode
    rng = np.random.default_rng(42)
    # slots[b, l] = which cache slot episode b evicts at layer l
    slots = rng.integers(1, L - 1, size=(B, n_layers))  # avoid first/last slot

    # --- Sequential ground truth ---
    # Episode a: evict slots[0, l] at each layer l
    kv_a_pl = clone_cache(kv_a1)
    for l in range(n_layers):
        _evict_slot(kv_a_pl, l, int(slots[0, l]))

    # Episode b: evict slots[1, l] at each layer l
    kv_b_pl = clone_cache(kv_b1)
    for l in range(n_layers):
        _evict_slot(kv_b_pl, l, int(slots[1, l]))

    logits_a_pl_seq, _ = decode_step(model, next_a, kv_a_pl, pos)
    logits_b_pl_seq, _ = decode_step(model, next_b, kv_b_pl, pos)

    # --- Batched per-layer physical eviction ---
    # Start from the batched cache and remove per-(batch, layer) slots in-place.
    kv_bat_pl = cache_to_batched(kv_a1, kv_b1)

    slots_t = torch.tensor(slots, device=DEVICE)   # [B, n_layers]
    for l in range(n_layers):
        K, V = _get_kv(kv_bat_pl, l)   # [B, H, S, D]
        S = K.shape[2]
        H = K.shape[1]
        D = K.shape[3]
        # Build keep-mask: [B, S] True for kept positions
        keep = torch.ones(B, S, dtype=torch.bool, device=DEVICE)
        keep[torch.arange(B), slots_t[:, l]] = False  # [B, S]
        # Gather kept positions: reshape K to [B*H, S, D], apply mask, reshape back
        K_new = K[keep.unsqueeze(1).unsqueeze(-1).expand(B, H, S, D)].reshape(B, H, S-1, D)
        V_new = V[keep.unsqueeze(1).unsqueeze(-1).expand(B, H, S, D)].reshape(B, H, S-1, D)
        _set_kv(kv_bat_pl, l, K_new, V_new)

    # One batched forward pass — all sequences now have cache size S-1
    attn_mask_pl = torch.ones(B, (L - 1) + 1, device=DEVICE, dtype=torch.long)
    with torch.no_grad():
        out4 = model(
            input_ids=next_batch,
            past_key_values=kv_bat_pl,
            position_ids=pos_batch,
            attention_mask=attn_mask_pl,
            use_cache=True,
        )

    all_pass &= check("per-layer evict batch logits[0]", out4.logits[0:1], logits_a_pl_seq)
    all_pass &= check("per-layer evict batch logits[1]", out4.logits[1:2], logits_b_pl_seq)

    print()
    print("=" * 60)
    result = "ALL PASS ✓" if all_pass else "SOME TESTS FAILED ✗"
    print(f"Result: {result}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
