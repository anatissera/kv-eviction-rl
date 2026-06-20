# Method

This document defines what this project does, how it does it, and what it
explicitly does NOT do.

---

## What we are building

A reinforcement learning agent that learns to score and evict tokens from a
transformer's KV cache while preserving answer correctness on GSM8K.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 12 Q-heads, 2 KV-heads — GQA)
**Dataset**: GSM8K grade-school math problems (~100–200 token prompts, train split for training / test split for eval)

---

## Core idea

The policy is a **learned scoring function** over KV cache positions:

```
score(K_avg[t], V_avg[t]) → keep-importance
```

Tokens are evicted in order of lowest score until `budget` tokens remain per
layer. The budget is sampled uniformly from `[budget_min, budget_max]` each
episode, so the policy learns a **total ranking** of all T tokens rather than
a budget-specific selector.

The Learning to Evict paper does this in one shot with a Plackett-Luce ranking
policy. We implement it sequentially (one eviction per step) so we can use
standard MaskablePPO and get T−budget gradient updates per episode instead of
one, which helps with the sparse correctness reward.

---

## Agent design

### One policy, one environment per layer

Qwen2.5-1.5B has 28 layers × 2 KV-heads = 56 KV caches. We run **one
sub-environment per layer** (28 total), feeding it K/V features averaged over
that layer's 2 KV-heads. Each layer makes its own independent eviction decision
— layer 0 might keep tokens {5, 12, 47, …} while layer 15 keeps a completely
different set.

We empirically validated this granularity choice. A diagnostic over 20 GSM8K
examples measured the Spearman rank correlation between the two KV-heads'
actual decode-time attention patterns (per-layer, using `output_attentions=True`
with GQA grouping: max over Q-head group, per KV-head). Mean ρ = 0.67 across
layers — borderline. Given that sparse correctness reward and ~200 training
examples already make credit assignment hard, doubling to 56 agents for a
moderate precision gain is not justified. Per-layer (28 agents) is the right
trade-off. See `diagnostics/per_head_importance_correlation.py`.

At episode end, eviction is applied per-layer by **slicing the DynamicCache**:

```python
# For each layer l, keep only the positions where resident[l, t] is True
layer.keys   = layer.keys.index_select(dim=2, index=keep_idx[l])
layer.values = layer.values.index_select(dim=2, index=keep_idx[l])
```

A manual greedy decode loop then runs with explicit `position_ids` set to the
token's **true original position** in the sequence (not the trimmed cache
length). RoPE rotations are baked into K values at their original positions at
prefill time; the new Q must use the same coordinate system to get the correct
attention logit Q[T+step] · K[i].

This approach (from H2O / SnapKV / PyramidKV) is exact and works with any
attention backend — no eager attention required for eviction.

Going finer than per-layer (per-KV-head, 56 envs) would require different
keep-sets per head within a layer, which `index_select` on
`[batch, n_kv_heads, seq, dim]` supports, but adds credit-assignment noise
with no clear benefit over per-layer.

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
same token twice. This repeats until the per-layer resident count reaches
`budget`. Since all layers start with T tokens and each evicts one per step,
all layers have exactly `budget` tokens after `T − budget` steps.

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

**Evicted positions are zeroed.** Once a token is evicted, its K/V entry in
the observation is set to zero. This makes the observation **non-constant
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
Input:  [K_avg[t] || V_avg[t]]            shape: (2 * head_dim,) = (128,)
  → LayerNorm(K half) || LayerNorm(V half)     # separate norms: K-norms vary
  → Linear(128 → hidden) → LayerNorm → SiLU   #   5–50× across layers
  → Linear(hidden → hidden) → LayerNorm → SiLU
  → Linear(hidden → 1)                        # one scalar keep-score per token
Output: scalar keep-score                   shape: (1,)
```

Applying the same weights to all positions means the policy learns
**token-agnostic importance features** that generalise across prompts and
positions.

The MLP outputs **one scalar per token** (`features_dim = max_len`). SB3
attaches a `Linear(max_len → max_len)` actor head (65K params) and a
`Linear(max_len → 1)` critic head. This is much smaller than a naive
`max_len × hidden → max_len` projection (4.2M params) and avoids the actor
head entangling all token positions through a massive projection.

**Per-half LayerNorm**: K and V halves are normalised separately before the
MLP projection. Without this, K-norms vary 5–50× across layers (attention
sinks, massive-activation tokens), causing high-norm layers to dominate
gradients.

---

## Reward

### Motivation

The correctness signal (0 or 1) is very sparse — many episodes end with score
0 until the policy is nearly optimal. Reward shaping supplements it with a
dense proxy: how much of the model's **future attention** falls on the tokens
we kept?

The attention alignment signal is always > 0 (some tokens always receive
attention), so the policy gets an informative gradient even when correctness
is 0 throughout early training.

### Two LLM calls per episode (shaping enabled, default)

| Call | When | Purpose | Attn backend |
|------|------|---------|--------------|
| Clean reference run | episode `reset()` | Full-context generate → per-token importance | `eager` (needs `output_attentions=True`) |
| Eviction generate   | episode terminal  | Per-layer cache slicing + manual decode → correctness | any |

`train.py` automatically selects `attn_implementation="eager"` when
`use_attention_shaping: true`, and the faster `sdpa` otherwise.

### Clean reference run (at reset)

`model.generate()` on the **full** prompt with `output_attentions=True`.
For each (step, layer), query-head attention weights are collapsed within each
GQA group using **max** (SnapKV convention), then averaged over KV-heads and
accumulated across decode steps. The result is `importance[t]` — how much the
model attended to prompt position `t` while producing the full-context answer,
normalised to sum to 1.

**Fallback**: if the attention backend returns empty matrices (SDPA / flash),
we use `importance[t] ∝ max_over_layers_heads(||K[t]|| + ||V[t]||)`
(KV-norm heuristic, same max convention). Shaping always provides a signal.

### Eviction generate (at terminal)

Per-layer DynamicCache slicing followed by a manual greedy decode:

1. Fresh prefill on the full prompt → `DynamicCache` with T entries per layer.
2. For each layer l: `layer.keys = layer.keys.index_select(2, keep_idx[l])`.
3. Manual decode loop with explicit `position_ids = [[true_position]]` and
   `cache_position = [true_position]`, where `true_position` increments from T
   upward — the true original index in the full sequence, not the trimmed
   cache length (which would give wrong RoPE).

### Combined terminal reward

```
alignment   = Σ_t importance[t] * mean_l(resident[l, t])       ∈ [0, 1]
correctness = flexible_extract(generated_text, gold_answer)      ∈ {0, 1}

reward = (1 − attention_weight) × correctness
       + attention_weight       × alignment
```

Default `attention_weight = 0.3`.

**Why soft alignment?** Each layer has its own `resident[l, t]` mask.
`mean_l(resident[l, t])` is the fraction of layers that kept token t — tokens
kept by all 28 layers contribute full weight; tokens kept by half contribute
half weight. This is the natural per-layer generalisation of the hard
kept/evicted binary from a global mask.

To disable shaping (e.g., GPU with flash_attention_2):
```yaml
use_attention_shaping: false
```
This reduces to pure correctness reward and uses the faster SDPA backend.

All 28 environments receive the same scalar reward (cooperative setting).

`gamma = 1.0` — no discounting. Every eviction step contributes equally to
the outcome; discounting would introduce an arbitrary credit bias toward later
evictions.

`gae_lambda = 1.0` — with `gamma=1.0` and `lambda<1`, later steps get
higher advantage weight (0.95^k decay). Setting `lambda=1.0` gives Monte
Carlo returns: identical advantage estimates for all steps in an episode.

---

## Training loop

Gradients **never flow through the LLM**. The LLM is frozen throughout; only
the policy MLP (PerTokenMLP + PPO actor/critic heads) is trained.

```
# Frozen: Qwen2.5-1.5B weights
# Trained: PerTokenMLP (scalar output per token) + SB3 actor/critic heads

──────────────────────────────────────────────────────────────────────
OUTER LOOP  (repeat until total_timesteps reached)
──────────────────────────────────────────────────────────────────────

  ── ROLLOUT COLLECTION  (n_steps transitions, no gradients) ──────────

  for each episode in rollout:

    # ① Capture: one frozen LLM forward pass per episode
    K, V = LLM.prefill(prompt)          # torch.no_grad(), use_cache=True
    input_ids, gold = prompt, example.gold_answer

    # ② Reference run: collect future attention for reward shaping
    importance = LLM.generate(          # torch.no_grad(), eager attn
        input_ids, output_attentions=True
    )  →  importance[t] ∈ [0,1], sums to 1   (or KV-norm proxy)
         (GQA-aware: max over Q-heads within each KV group)

    # budget sampled fresh each episode — policy learns a total ranking
    budget ~ Uniform(budget_min, min(budget_max, T-1))

    # Initial obs per layer: K/V for all T tokens (evicted positions zeroed)
    resident = ones(L, T, bool)
    obs = build_obs(K_avg, V_avg, resident_mask=resident)  # [L, T, 2D]

    # ③ Sequential eviction — T−budget steps, one token per layer-env per step
    for step in range(T - budget):
        action_mask = resident          # [L, T] — already-evicted are False
        action = policy.predict(obs, mask)     # no_grad: rollout collection
        resident[l, action[l]] = False         # evict chosen token in each layer
        obs[l, action[l]] = 0                  # zero evicted position in obs

        if not done:
            reward = 0                 # intermediate steps: no reward

        if done:                       # last eviction step
            # ④ Per-layer cache slicing + manual decode
            #    - fresh prefill → DynamicCache
            #    - for each layer l: index_select(dim=2, keep_idx[l])
            #    - greedy decode with position_ids = true_original_position
            text = generate_with_per_layer_eviction(
                model, tokenizer, input_ids,
                resident_mask=resident,   # [L, T] bool, independent per layer
                max_new_tokens=max_new_tokens,
            )
            correctness = flexible_extract(text, gold)         ∈ {0, 1}
            soft_keep   = resident.float().mean(dim=0)         ∈ [0, 1]^T
            alignment   = (importance * soft_keep).sum()       ∈ [0, 1]
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

      ppo_loss.backward()    # updates PerTokenMLP weights only
      optimizer.step()

──────────────────────────────────────────────────────────────────────
```

**Key points:**
- Steps ①②③④ are all `torch.no_grad()` — the LLM is a black-box reward oracle.
- Step ⑤ is the only place gradients flow, and only through the tiny policy MLP.
- Every episode produces `(T − budget) × 28` transitions in the rollout buffer,
  all with reward 0 except the terminal step which carries the shaped reward.
- GAE with `gamma=1.0, lambda=1.0` gives identical advantages for all eviction
  steps (pure Monte Carlo returns, no temporal bias).
- The observation changes each step as evicted positions are zeroed, so the
  value function has a non-trivial signal to learn from.

---

## What we do NOT do

| What | Why |
|------|-----|
| One-shot ranking (Plackett-Luce) | Sequential gives more gradient signal per episode with sparse reward |
| Per-KV-head independent eviction | Per-layer is already finer than global; per-head within a layer adds credit-assignment noise with no clear benefit |
| Global consensus mask at generate time | Each layer applies its own independent eviction via DynamicCache slicing — no averaging across layers |
| Eager attention for eviction generate | Cache slicing works with any backend; only the clean reference run needs eager (for `output_attentions=True`) |
| AUC proxy reward | Correctness is the real signal; alignment is shaping only |
| Online attention recompute after each eviction | Features are fixed at prefill; recompute would add cost with no benefit |
| Mean over Q-heads in GQA | Max within each KV-head group (SnapKV convention) — mean dilutes tokens critical to specific heads |
| Hidden state `h[t]` features | Good next step if K/V features underfit |

---

## Success criteria

On the held-out GSM8K test set, the policy achieves correctness within 5
percentage points of the full-cache baseline at the evaluation budget.
Random eviction and KV-norm oracle serve as lower and upper bounds.
Eval reports Wilson 95% confidence intervals; a 5pp difference requires
≥200 examples to be statistically distinguishable.
