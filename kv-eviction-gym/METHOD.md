# Method and Restrictions

This document defines what this project does, how it does it, and what it
explicitly does NOT do. Read it before making changes to avoid scope creep.

---

## What we are building

A reinforcement learning agent that learns to selectively evict tokens from
the KV cache of a large language model while preserving generation quality.

**Task**: Given a prompt cached in a transformer's KV cache, evict tokens
one at a time until a budget is reached. The agent must choose which tokens
to remove so that the model's future generation is least affected.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 16 Q-heads, 2 KV-heads → GQA).

**Dataset**: GSM8K grade-school math problems (~100–200 token prompts after
formatting with a zero-shot CoT template).

---

## Agent design

### One shared policy across all heads

There are 28 layers × 2 KV-heads = 56 (layer, kv_head) pairs, each with
its own independent KV cache. Only one policy network is trained. The policy
is conditioned on `(layer_frac, head_frac)` features so it can distinguish
heads, but weights are shared.

**GQA**: Qwen2.5-1.5B has 12 Q-heads and 2 KV-heads per layer. Each KV-head
is shared by 6 Q-heads. Eviction decisions are at the KV-head level (one
decision affects all Q-heads in the group). Q vectors and attention scores are
averaged over the Q-group before being used as features.

**Why share weights**: 56 environments all contribute gradients per update,
giving the policy far more training signal per LLM call.

### Sequential eviction: one token per step

At each step the policy picks exactly one token to evict (Discrete action
space). This repeats until the per-head token count reaches `budget`.

**Why not one-shot ranking (like the paper)**: one-shot Plackett-Luce
scoring is a contextual bandit — there is no state evolution between
decisions. Sequential eviction is a true MDP: after each eviction the
attention pattern changes, giving the policy updated information for the
next decision. This lets it learn adaptive strategies (e.g., "if the most
attended token is gone, the next best changes").

### Action masking

Evicted tokens are masked out (logit → -∞) so the policy can only pick
from currently resident tokens. MaskablePPO from sb3-contrib handles this.

The observation always has shape `[max_len, feature_dim]` (fixed, padded).
The mask communicates which positions are valid. No variable-size tensors.

---

## Shared environment: one LLM call per episode

All 56 (layer, kv_head) environments share a single LLM prefill+decode per
episode reset. One GSM8K example → one forward pass → Q, K, V, and
future attention weights for every KV-head simultaneously.

All 56 sub-environments step in lockstep (same prompt, same budget, same
episode length), so they always terminate together. One reset = one LLM call.

**Efficiency**: 56 episodes worth of training signal from a single ~10s
model call on MPS/CPU, or ~1s on GPU.

---

## Observations

Per-token features (feature_dim = 2 × head_dim + 5 = 261 for Qwen2.5-1.5B):

| Feature | Shape | Notes |
|---------|-------|-------|
| K vector | (head_dim,) | Key for this token; RMS-normalized per episode |
| V vector | (head_dim,) | Value for this token; RMS-normalized per episode |
| attn_score | (1,) | How much attention this token receives from all queries |
| position | (1,) | Token index / max_len |
| is_resident | (1,) | 1 if still in cache, 0 if evicted (mirrors action mask) |
| layer_frac | (1,) | layer_idx / n_layers, broadcast to all tokens |
| head_frac | (1,) | head_idx / n_heads, broadcast to all tokens |

**Attention scores** are captured from the actual softmax weights during
prefill (via hook), then recomputed after each eviction step using exact
online attention (`Q @ K_resident.T / sqrt(D)` → softmax → column sum).

---

## Rewards

### Phase 1 — Future-attention AUC (current)

No LLM call during RL steps. The reward at episode end is:

```
AUC_policy = Σ future_attn[i]  for i in resident tokens
AUC_oracle = Σ top-k future_attn values  (best possible)
reward = AUC_policy / AUC_oracle  ∈ (0, 1]
```

`future_attn[i]` = how much attention the generated tokens paid to prompt
token `i` during a reference decode. Captured at reset via hooks (one hook
per layer per decode step, immediately reduced and freed — only one layer's
`[H, 1, T]` attention matrix lives in memory at a time).

AUC_oracle = 1.0 only when the policy keeps exactly the tokens that received
the most future attention. Random eviction typically gives ~0.3–0.6.

### Phase 2 — GSM8K correctness (future)

Run generation from the evicted cache and score against the gold answer.
Requires a second LLM call per episode. Use only after Phase 1 converges.

---

## Policy architecture

`PerTokenMLP`: the same small MLP is applied independently to each token
position (shared weights across positions). This is the right inductive bias
— importance of a token doesn't depend on which slot it occupies.

```
Input per token: [feature_dim]
Linear(feature_dim → hidden) → LayerNorm → SiLU →
Linear(hidden → hidden) → LayerNorm → SiLU
Output: flatten [max_len × hidden] → SB3 actor adds Categorical head
```

Hidden size: 64 (increase to 128 if underfitting).

---

## Training algorithm

**MaskablePPO** (sb3-contrib) with standard hyperparameters:

| Param | Value | Reason |
|-------|-------|--------|
| n_steps | 256 per env | ~1–2 episodes per rollout |
| n_epochs | 4 | standard |
| gamma | 0.99 | standard |
| gae_lambda | 0.95 | standard |
| clip_range | 0.2 | standard |
| ent_coef | **0.0** | matches the paper — Gumbel-sort exploration not needed here |
| batch_size | 256 | |

---

## What we explicitly do NOT do

| What | Why |
|------|-----|
| Block eviction (evict >1 per step) | Complicates the action space; start simple |
| Separate policy per head | Reduces sample efficiency; shared policy with (layer, head) features is enough |
| RULER long-context data | Too long for CPU/MPS local runs; episodes of 3000+ steps make GAE unstable |
| Distributed training | Out of scope for now; one GPU is the target |
| Masking trick for attention recompute | We use exact online recompute (`Q @ K_res.T → softmax`), which is correct |
| output_attentions=True globally | Materialises all-layer attention simultaneously — O(L × H × T²) memory. We inject it per-layer via pre-hooks and free immediately |
| PPO value critic | MaskablePPO uses a learned critic; that's fine. We do NOT add the extra RLOO leave-one-out baseline from ml-learning-to-evict |
| Phase 2 before Phase 1 converges | Correctness reward is sparse and expensive; AUC reward must train first |

---

## What "done" looks like

**Phase 1 success**: on a held-out set of GSM8K examples, the learned
policy achieves AUC ratio > 0.8 (i.e., keeps ≥80% of the attention mass
that the oracle would keep). Random baseline is ~0.4–0.6.

**Phase 2 success**: generation accuracy on GSM8K with the evicted cache
is within 5 percentage points of the full-cache baseline at the same budget.

---

## Compute targets

| Setting | Hardware | Expected speed |
|---------|----------|----------------|
| Quickstart (1 example, budget=8) | Mac MPS | ~1100 steps/s, ~77s/rollout |
| Phase 1 full (200 examples, budget=32) | A100/A6000 GPU | ~5000 steps/s |
| Phase 2 correctness | A100/A6000 GPU | ~200 episodes/h |
