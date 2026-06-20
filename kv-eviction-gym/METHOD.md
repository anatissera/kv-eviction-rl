# Method

This document defines what this project does, how it does it, and what it
explicitly does NOT do.

---

## What we are building

A reinforcement learning agent that learns to evict tokens from a
transformer's KV cache while preserving answer correctness on GSM8K.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 12 Q-heads, 2 KV-heads — GQA)
**Dataset**: GSM8K grade-school math problems (~100–200 token prompts)

---

## Agent design

### One policy, one environment per (layer, KV-head)

Qwen2.5-1.5B has 28 layers × 2 KV-heads = **56 independent KV caches**.
Each has its own (K, V) tensors. We train one shared policy across all 56,
with per-head environments that each make independent eviction decisions.

### Sequential eviction — one token per step

At each step the policy picks exactly one token to evict from its head's
cache (Discrete action, MaskablePPO masks already-evicted positions).
This repeats until the per-head count reaches `budget`.

### Shared prefill, lockstep episodes

All 56 environments share one LLM prefill per episode reset. One GSM8K
example → one forward pass → K/V for all 56 heads simultaneously.
All envs terminate together (same prompt length, same budget).

---

## Observations

**3 features per token position** (feature_dim = 3):

| Feature | Notes |
|---------|-------|
| `‖K[t]‖` | L2 norm of the key vector — proxy for attention importance |
| `‖V[t]‖` | L2 norm of the value vector |
| `t / max_len` | Relative position in the prompt |

No raw K/V vectors, no attention scores, no `is_resident` (mask handles
that), no layer/head fracs (add back if ablations show benefit).

**Why no attention scores**: computing softmax attention weights requires
`output_attentions=True` which forces `attn_implementation="eager"` and
disables FlashAttention. Key/value norms are cheaper, sufficient, and
compatible with any attention backend.

The observation is constant throughout the episode (features don't change
as tokens are evicted). Only the action mask shrinks.

---

## Reward

One LLM call per episode — at the terminal step only.

1. **Aggregate**: each token's keep-score = fraction of the 56 heads that
   kept it. Take the top-`budget` tokens by this score.
2. **Generate**: re-run `model.generate()` with an attention mask that
   zeros out the evicted positions. No KV cache manipulation needed.
3. **Score**: `flexible_extract` from `lm-evaluation-harness` — extracts
   the last number from the generated text and compares to the gold answer.
   Returns `1.0` if correct, `0.0` otherwise.

All 56 environments receive the same reward (cooperative multi-agent).

The reward is sparse (0 or 1) but episodes are short (~100–150 steps),
so PPO with GAE handles the credit assignment.

---

## What we do NOT do

| What | Why |
|------|-----|
| Eager attention during training | No hooks needed; FlashAttention ok |
| AUC proxy reward | Correctness is the real signal; no need for a proxy |
| Separate Phase 1 / Phase 2 | One unified loop from the start |
| Online attention recompute after each eviction | Features are fixed; too slow on CPU |
| Per-head generate at terminal | Requires custom attention or 56 separate decodes; global consensus used instead |
| layer_frac / head_frac features | Start without; add if ablations show benefit |

---

## Success criteria

On a held-out set of GSM8K examples, the policy achieves correctness
within 5 percentage points of the full-cache baseline at `budget` tokens.
Random eviction baseline serves as the lower bound.
