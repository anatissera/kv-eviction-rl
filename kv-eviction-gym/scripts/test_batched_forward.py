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

def clone_cache(cache):
    new = DynamicCache()
    for l in range(len(cache.key_cache)):
        new.key_cache.append(cache.key_cache[l].clone())
        new.value_cache.append(cache.value_cache[l].clone())
    return new

def cache_to_batched(cache_a, cache_b):
    """Stack two single-sequence caches into one batch=2 cache."""
    batched = DynamicCache()
    for l in range(len(cache_a.key_cache)):
        batched.key_cache.append(
            torch.cat([cache_a.key_cache[l], cache_b.key_cache[l]], dim=0)
        )
        batched.value_cache.append(
            torch.cat([cache_a.value_cache[l], cache_b.value_cache[l]], dim=0)
        )
    return batched

def split_cache(batched_cache, idx):
    """Extract one sequence from a batched cache."""
    single = DynamicCache()
    for l in range(len(batched_cache.key_cache)):
        single.key_cache.append(batched_cache.key_cache[l][idx:idx+1])
        single.value_cache.append(batched_cache.value_cache[l][idx:idx+1])
    return single

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
    ka_batch = kv_batch.key_cache[0][0]   # [H, S, D]
    kb_batch = kv_batch.key_cache[0][1]
    ka_seq   = kv_a1.key_cache[0][0]
    kb_seq   = kv_b1.key_cache[0][0]
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

    logits_a_evict_seq, _ = decode_step(model, next_a, kv_a_evicted, pos - 1)

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
    result = "ALL PASS ✓" if all_pass else "SOME TESTS FAILED ✗"
    print(f"Result: {result}")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
