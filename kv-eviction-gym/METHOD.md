# Method

This document defines what this project does, how it does it, and what it
explicitly does NOT do.

---

## What we are building

A reinforcement learning agent that learns to score and evict tokens from a
transformer's KV cache while preserving answer correctness on GSM8K.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 12 Q-heads, 2 KV-heads — GQA)
**Dataset**: GSM8K grade-school math problems (~100–200 token prompts)

---

## Core idea

The policy is fundamentally a **learned scoring function** over KV cache
positions:

```
score(K[t], V[t], t/max_len) → importance
```

Tokens are evicted in order of lowest score until `budget` tokens remain.
Since the features `[K[t], V[t], position]` are fixed throughout the
episode, the policy implicitly learns a ranking — not a dynamic strategy.

The Learning to Evict paper does this in one shot with a Plackett-Luce
ranking policy. We implement it sequentially (one eviction per step) so we
can use standard MaskablePPO and get T−budget gradient updates per episode
instead of one, which helps with the sparse correctness reward.

---

## Agent design

### One policy, one environment per (layer, KV-head)

Qwen2.5-1.5B has 28 layers × 2 KV-heads = **56 independent KV caches**.
We train one shared policy across all 56, with per-head environments that
each make independent eviction decisions.

### Sequential eviction — one token per step

At each step the policy picks exactly one token to evict (Discrete action).
MaskablePPO masks already-evicted positions so the policy never selects the
same token twice. This repeats until the per-head count reaches `budget`.

### Shared prefill, lockstep episodes

All 56 environments share one LLM prefill per episode reset. One GSM8K
example → one forward pass → K/V for all 56 heads simultaneously.
All envs terminate together (same prompt length, same budget).

---

## Observations

**Features per token position t** (`feature_dim = 2 × head_dim + 1 = 129`):

| Feature | Shape | Notes |
|---------|-------|-------|
| `K[t]` | (head_dim,) | Full key vector with RoPE — encodes content + position |
| `V[t]` | (head_dim,) | Full value vector |
| `t / max_len` | (1,) | Explicit relative position (easy positional prior) |

The observation is **constant throughout the episode** — features do not
change as tokens are evicted. Only the action mask shrinks.

**Why not Q?** K and V are what stay in the cache; they are the natural
features for the eviction decision. Q is only needed at decode time to
decide what to attend to, which we don't have access to during training.

**Why not attention scores?** Computing softmax weights requires
`output_attentions=True`, which forces `attn_implementation="eager"` and
disables FlashAttention. K/V capture happens for free from `use_cache=True`
with any attention backend.

**Why not hidden states?** The paper uses the residual stream `h[t]` as
features (richer than K/V because it integrates all layers). This requires
a hook and more memory. K/V are a good starting point; add `h[t]` if the
policy underperforms.

---

## Reward

One LLM call per episode — at the terminal step only.

1. **Aggregate**: each token's keep-score = fraction of the 56 heads that
   kept it. Take the top-`budget` tokens by this score.
2. **Generate**: re-run `model.generate()` with an attention mask that
   zeros out the evicted positions.
3. **Score**: `flexible_extract` — extracts the last number from the
   generated text and compares to the gold answer.
   Returns `1.0` if correct, `0.0` otherwise.

All 56 environments receive the same reward (cooperative multi-agent).

The reward is sparse (0 or 1) but episodes are short (~100–150 steps),
so PPO with GAE handles credit assignment.

---

## What we do NOT do

| What | Why |
|------|-----|
| One-shot ranking (Plackett-Luce) | Sequential gives more gradient signal per episode with sparse reward |
| Eager attention / hooks during training | Not needed; K/V come free from `use_cache=True` |
| AUC proxy reward | Correctness is the real signal |
| Online attention recompute after each eviction | Features are fixed; recompute would add cost with no benefit |
| Per-head generate at terminal | Requires custom attention or 56 separate decodes; global top-k consensus used instead |
| layer/head fracs in features | Start without; add if ablations show benefit |
| Hidden state `h[t]` features | Good next step if K/V features underfit |

---

## Success criteria

On a held-out set of GSM8K examples, the policy achieves correctness
within 5 percentage points of the full-cache baseline at `budget` tokens.
Random eviction serves as the lower bound.
