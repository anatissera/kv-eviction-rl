# Method

This document defines what this project does, how it does it, and what it
explicitly does NOT do.

---

## What we are building

A reinforcement learning agent that learns to score and evict tokens from a
transformer's KV cache while preserving answer correctness on GSM8K.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 12 Q-heads, 2 KV-heads — GQA)
**Dataset**: GSM8K grade-school math problems (~100–200 token prompts, train split)

---

## Core idea

The policy is fundamentally a **learned scoring function** over KV cache
positions:

```
score(K_avg[t], V_avg[t]) → importance
```

Tokens are evicted in order of lowest score until `budget` tokens remain.
The budget is sampled uniformly from `[budget_min, budget_max]` each episode,
so the policy learns a **total ranking** of all T tokens rather than a
budget-specific selector.

The Learning to Evict paper does this in one shot with a Plackett-Luce
ranking policy. We implement it sequentially (one eviction per step) so we
can use standard MaskablePPO and get T−budget gradient updates per episode
instead of one, which helps with the sparse correctness reward.

---

## Agent design

### One policy, one environment per layer

Qwen2.5-1.5B has 28 layers × 2 KV-heads = 56 independent KV caches.
However, the final eviction is evaluated with **one global attention mask**
applied identically across all layers and heads — `model.generate()` has no
API for per-layer or per-head masks. Given this constraint, the correct
aggregation granularity is **per layer**: we run one sub-environment per
layer (28 total), feeding it K/V features averaged over that layer's KV-heads.
At episode end the per-layer eviction decisions are averaged into a consensus
global mask for the generate call.

Going finer than per-layer (i.e., back to per-KV-head, 56 envs) would produce
56 binary votes that get averaged into the same global mask — no finer
eviction granularity in practice, just more credit-assignment noise.

### GQA and head aggregation

In GQA, each KV-head is shared by a **group** of Q-heads (Qwen: group size 6).
For observation features we average K and V over the 2 KV-heads per layer
(`cap.K.mean(dim=1)` → `[L, T, D]`). For attention-based importance we
follow SnapKV / Learning-to-Evict:

> **Max within GQA group, then mean over KV-heads.**
>
> For each layer, group the 12 Q-heads into 2 groups of 6. Take **max** over
> each group's attention weights before averaging over the 2 KV-heads.
> Rationale: a token should be preserved if **any** Q-head in the group
> attends to it; mean would dilute tokens that are critical to a single head.

The same max-over-heads logic applies to the KV-norm fallback:
`(K.norm + V.norm).amax(dim=(layers, kv_heads))`.

### Sequential eviction — one token per step

At each step the policy picks exactly one token to evict (Discrete action).
MaskablePPO masks already-evicted positions so the policy never selects the
same token twice. This repeats until the per-layer count reaches `budget`.

### Shared prefill, lockstep episodes

All 28 environments share one LLM prefill per episode reset. One GSM8K
example → one forward pass → K/V for all 28 layers simultaneously.
All envs terminate together (same prompt length, same budget).

---

## Observations

**Features per token position t** (`feature_dim = 2 × head_dim = 128`):

| Feature | Shape | Notes |
|---------|-------|-------|
| `K_avg[t]` | (head_dim,) | Key vector averaged over KV-heads for this layer |
| `V_avg[t]` | (head_dim,) | Value vector averaged over KV-heads |

Position is already embedded in `K[t]` via the RoPE rotation, so an
explicit `t/max_len` scalar is redundant.

**Evicted positions are zeroed.** Once a token is evicted, its K/V entry
in the observation is set to zero. This makes the observation **non-constant
across steps** — the value function can observe how many tokens remain and
which ones have been removed, giving it a meaningful signal for advantage
estimation.

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

## Policy architecture

`PerTokenMLP` — a shared MLP applied independently to each token position:

```
Input:  [K_avg[t] || V_avg[t]]           shape: (2 * head_dim,) = (128,)
  → LayerNorm(K half) || LayerNorm(V half)    # separate norms: K-norms vary
  → Linear(128 → hidden) → LayerNorm → SiLU  #   5–50× across layers
  → Linear(hidden → hidden) → LayerNorm → SiLU
  → Linear(hidden → 1)                       # one scalar keep-score per token
Output: scalar keep-score                  shape: (1,)
```

Applying the same weights to all positions means the policy learns
**token-agnostic importance features** that generalise across prompts and
positions.

The MLP outputs **one scalar per token** (`features_dim = max_len`). SB3
attaches a `Linear(max_len → max_len)` actor head (65K params) and a
`Linear(max_len → 1)` critic head. This is much smaller than the previous
design (`max_len × hidden → max_len` = 4.2M params) and avoids the actor
head entangling all token positions through a massive projection.

**Per-half LayerNorm**: K and V halves are normalised separately before the
MLP projection. Without this, K-norms vary 5–50× across layers (attention
sinks, massive-activation tokens), causing high-norm layers to dominate
gradients.

---

## Reward

### Motivation

The correctness signal (0 or 1) is very sparse — many episodes end with
score 0 until the policy is nearly optimal. Reward shaping supplements it
with a dense proxy: how much of the model's **future attention** falls on
the tokens we kept?

Crucially, the attention alignment signal is always > 0 (some tokens always
receive attention), so the policy gets an informative gradient even when
correctness is 0 throughout early training.

### Two LLM calls per episode (shaping enabled, default)

| Call | When | Purpose | attn backend |
|------|------|---------|-------------|
| Clean reference run | episode `reset()` | Full-context generate → per-token importance | `eager` (needs attention weights) |
| Eviction generate | episode terminal | Masked generate → correctness score | any |

`train.py` automatically selects `attn_implementation="eager"` when
`use_attention_shaping: true`, and the faster `sdpa` otherwise.

### Clean reference run (at reset)

`model.generate()` on the **full** prompt with `output_attentions=True`.
For each (step, layer), query-head attention weights are collapsed within
each GQA group using **max** (SnapKV convention), then averaged over KV-heads
and accumulated across decode steps. The result is `importance[t]` — how much
the model attended to prompt position `t` while producing the answer,
normalised to sum to 1.

**Fallback**: if the attention backend returns empty matrices (SDPA / flash),
we use `importance[t] ∝ max_over_layers_heads(||K[t]|| + ||V[t]||)`
(KV-norm heuristic, same max convention). Shaping always provides a signal.

### Eviction generate (at terminal)

`model.generate()` with the attention mask that zeros out evicted positions.
Explicit `position_ids = [0, 1, ..., T-1]` ensure each surviving token
retains its original RoPE rotation (without this, HF shifts positions via
cumsum(attn_mask)-1 after each masked gap in older versions).

### Combined terminal reward

```
alignment   = Σ importance[t]  for t in kept_tokens       ∈ [0, 1]
correctness = flexible_extract(generated_text, gold)        ∈ {0, 1}

reward = (1 − attention_weight) × correctness
       + attention_weight       × alignment
```

Default `attention_weight = 0.3`.

To disable shaping (e.g., GPU with flash_attention_2):
```yaml
use_attention_shaping: false
```
This reduces to pure correctness reward and uses the faster SDPA backend.

All 28 environments receive the same scalar reward (cooperative setting).

`gamma = 1.0` — no discounting. Every eviction step contributes equally to
the outcome; discounting would introduce an arbitrary credit bias.

`gae_lambda = 1.0` — with `gamma=1.0` and `lambda<1`, later eviction steps
get higher advantage weight than earlier ones (0.95^k decay). Setting
`lambda=1.0` gives Monte Carlo returns: identical advantage estimates for
all steps in an episode.

---

## Training loop

Gradients **never flow through the LLM**. The LLM is frozen throughout;
only the policy MLP (PerTokenMLP + PPO actor/critic heads) is trained.

```
# Frozen: Qwen2.5-1.5B weights
# Trained: PerTokenMLP (scalar output per token) + SB3 actor/critic heads

──────────────────────────────────────────────────────────────────────
OUTER LOOP  (repeat until total_timesteps reached)
──────────────────────────────────────────────────────────────────────

  ── ROLLOUT COLLECTION  (n_steps transitions, no gradients) ──────────

  for each episode in rollout:

    # ① Capture: one frozen LLM forward pass
    K, V = LLM.prefill(prompt)            # torch.no_grad()
    input_ids, gold = prompt, example.gold_answer

    # ② Reference run: collect future attention for reward shaping
    importance = LLM.generate(            # torch.no_grad(), eager attn
        input_ids, output_attentions=True
    )  →  importance[t] ∈ [0,1], sums to 1   (or KV-norm proxy)
         (GQA-aware: max over Q-heads within each KV group)

    # budget sampled fresh each episode from [budget_min, budget_max]
    budget ~ Uniform(budget_min, min(budget_max, T-1))

    # Initial obs: K/V for all T tokens (evicted positions will be zeroed)
    obs = [K_avg[t], V_avg[t]]  for t in 0..T-1  per layer

    # ③ Sequential eviction (T − budget steps per layer-env)
    for step in range(T - budget):
        action_mask = resident_tokens        # which positions still valid
        action = policy.predict(obs, mask)   # no_grad: rollout collection
        evict token[action] from resident
        obs[action] = 0                      # zero out evicted position

        if not done:
            reward = 0                       # intermediate steps: no reward

        if done:                             # last eviction step
            # Aggregate per-layer decisions → global consensus mask
            token_scores = resident.float().mean(dim=0)  # [T], mean over layers
            kept_tokens  = token_scores.topk(budget).indices

            # ④ Eviction generate: score correctness of kept tokens
            text = LLM.generate(            # torch.no_grad()
                input_ids, attention_mask=kept_tokens
            )
            correctness = (text matches gold)    ∈ {0, 1}
            alignment   = sum(importance[kept])  ∈ [0, 1]
            reward = 0.7 × correctness + 0.3 × alignment

        store (obs, action, reward, value, log_prob) in rollout_buffer

  ── PPO UPDATE  (n_epochs passes over the rollout, WITH gradients) ───

  compute GAE advantages from rollout_buffer  (γ=1.0, λ=1.0)

  for epoch in range(n_epochs):
    for minibatch in rollout_buffer:

      # ⑤ Gradient flows HERE — through policy MLP only
      logits, value = policy.forward(obs)         # ← gradients ON
      ppo_loss = clip_loss(logits, actions, adv)
               + value_coef × value_loss(value, returns)

      ppo_loss.backward()    # updates PerTokenMLP weights
      optimizer.step()

──────────────────────────────────────────────────────────────────────
```

**Key points:**
- Steps ①②③④ are all `torch.no_grad()` — the LLM is a black-box reward oracle.
- Step ⑤ is the only place gradients flow, and only through the tiny policy MLP.
- Every episode produces `(T − budget) × 28` transitions in the rollout buffer,
  all with reward 0 except the terminal step which carries the shaped reward.
- GAE with `gamma=1.0, lambda=1.0` gives identical advantages for all steps
  (pure Monte Carlo returns from the terminal reward, no temporal bias).
- The observation changes each step as evicted positions are zeroed, so the
  value function has a non-trivial signal to learn from.

---

## What we do NOT do

| What | Why |
|------|-----|
| One-shot ranking (Plackett-Luce) | Sequential gives more gradient signal per episode with sparse reward |
| Per-layer or per-KV-head attention masks at generate time | `model.generate()` only accepts a global attention mask; true per-layer eviction requires patching the attention kernels |
| Eager attention for prefill / eviction generate | Only used once per episode (clean reference run); main prefill and eviction generate use `sdpa` or `flash` |
| AUC proxy reward | Correctness is the real signal |
| Online attention recompute after each eviction | Features are fixed at prefill; recompute would add cost with no benefit |
| Mean over Q-heads in GQA | Max within each KV-head group (SnapKV convention) — mean dilutes tokens critical to specific heads |
| layer/head fracs in features | Start without; add if ablations show benefit |
| Hidden state `h[t]` features | Good next step if K/V features underfit |

---

## Success criteria

On a held-out GSM8K test set, the policy achieves correctness within 5
percentage points of the full-cache baseline at the evaluation budget.
Random eviction and KV-norm oracle serve as lower and upper bounds.
Eval reports Wilson 95% confidence intervals; a 5pp difference requires
≥200 examples to be statistically distinguishable.
