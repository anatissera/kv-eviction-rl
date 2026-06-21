# Method

This document defines what this project does, how it does it, and what it
explicitly does NOT do.

---

## What we are building

A reinforcement learning agent that learns to evict tokens from a transformer's
KV cache **during generation** while preserving answer correctness on GSM8K.

**Model**: Qwen2.5-1.5B-Instruct (28 layers, 12 Q-heads, 2 KV-heads — GQA)
**Dataset**: GSM8K grade-school math problems (~100–200 token prompts, train split for training / test split for eval)

---

## Core idea

The policy is a **learned scoring function** over KV cache positions:

```
score(K_h0[t] || K_h1[t], V_h0[t] || V_h1[t]) → keep-importance
```

At each decode step, if the cache exceeds a capacity budget, the policy evicts
one cached token (prompt OR previously generated). The budget is sampled
uniformly from `[budget_min, budget_max]` each episode, where
`budget ≥ prompt_length` so no pre-eviction of the prompt is needed.

The episode IS the generation. Generated tokens are accumulated inline;
there is no separate "eviction generate" call. This means the policy
must learn to preserve tokens that will be needed later in the chain-of-thought,
not just tokens that look important from the static prompt.

The Learning to Evict paper does eviction in one shot with a Plackett-Luce
ranking policy. We implement it sequentially (one eviction per decode step)
so we can use standard MaskablePPO and get up to `max_new_tokens` gradient
updates per episode — important with the sparse 0/1 correctness reward.

---

## Agent design

### One policy, one environment per layer

Qwen2.5-1.5B has 28 layers × 2 KV-heads = 56 KV caches. We run **one
sub-environment per layer** (28 total), feeding it K/V features averaged over
that layer's 2 KV-heads. Each layer makes its own independent eviction decision
— layer 0 might evict token 47 while layer 15 evicts token 12 at the same step.

Per-layer is the finest granularity that standard transformers supports.
DynamicCache stores each layer's K and V as `[1, n_kv_heads, seq_len, head_dim]`;
`index_select(dim=2)` evicts the same token positions from both KV-heads within
a layer — the seq dimension is shared. Truly independent per-head caches would
require a custom attention kernel (paged attention / block-sparse), which is out
of scope.

We empirically validated that per-layer is sufficient. A diagnostic over 20 GSM8K
examples measured the Spearman rank correlation between the two KV-heads'
importance rankings using actual decode-time attention weights (`output_attentions=True`,
GQA-aware: max over Q-head group per KV-head). Mean ρ = 0.67 across layers —
the heads largely agree on which tokens matter. See `diagnostics/per_head_importance_correlation.py`.

Eviction is applied per-layer by **slicing the live DynamicCache**:

```python
# Remove slot s from layer l's cache (contiguous; positions after s shift left)
cache.layers[l].keys   = cache.layers[l].keys.index_select(dim=2, index=keep_idx)
cache.layers[l].values = cache.layers[l].values.index_select(dim=2, index=keep_idx)
```

The decode loop passes explicit `position_ids` set to the token's **true
original position** in the full sequence (not the trimmed cache length).
RoPE rotations are baked into K values at their original positions at prefill
time; the new Q must use the same coordinate to compute the correct attention
logit Q[true_pos] · K[i].

This approach (from H2O / SnapKV / PyramidKV) is exact and works with any
attention backend — no eager attention required for eviction.

### GQA and head aggregation

In GQA, each KV-head is shared by a **group** of Q-heads (Qwen: group size 6).
For observation features we concatenate K and V across the 2 KV-heads per layer:
`[K_h0 || K_h1]` and `[V_h0 || V_h1]` → `[L, S, 2D]` per half. For the KV-norm importance proxy we follow
the SnapKV / Learning-to-Evict convention:

> **Max within GQA group, then mean over KV-heads.**
>
> For each layer, group the 12 Q-heads into 2 groups of 6. Take **max** over
> each group before averaging. Rationale: a token should be preserved if **any**
> Q-head in the group attends to it; mean would dilute tokens critical to a
> single head.

### Online sequential eviction — one token per decode step

The episode runs as follows:

1. **Prefill**: tokenize the prompt → one frozen LLM forward pass → full
   prompt cache (T tokens per layer). The first generated token is read from
   the prefill logits.

2. **Free-growth** (inside `reset()`, no RL steps): decode tokens without
   eviction until `cache_size > budget`. This runs `budget − T` decode steps
   internally, so the RL agent never sees a no-op transition.

3. **Decode loop** (RL steps, up to `max_new_tokens − free_growth` steps or EOS):
   - Each layer-env evicts the token at its chosen slot from the live DynamicCache.
     Eviction **always fires** — no conditional needed after free-growth.
   - Run one greedy decode step → new K/V appended → new `next_token`.
   - Accumulate the token in `generated`.
   - Return the updated cache K/V as the next observation.

4. **Terminal**: decode `generated` → correctness check against gold answer.

After free-growth, the cache sits at exactly `budget + 1` tokens and stays
there for the rest of the episode (one evicted per step, one generated per step).

**Why free-growth belongs in `reset()`, not `step_wait()`:**
With terminal-only reward, `gamma=1`, and `gae_lambda=1`, every step in the
rollout buffer receives identical advantages.  If no-op steps (where
`cache ≤ budget`) were stored, the gradient would be diluted by a factor of
`max_new_tokens / n_eviction_steps` — in the worst case, 524 / 124 ≈ 4×.
Moving free-growth into `reset()` means the buffer contains only genuine
eviction decisions, eliminating this dilution entirely.

### Shared prefill, lockstep episodes

All 28 environments share one LLM prefill per episode reset. One GSM8K
example → one forward pass → the same live DynamicCache is shared and mutated
by all layer-envs. All envs terminate together (EOS or `max_new_tokens`).

---

## Observations

**Features per token position t** (`feature_dim = 2 × n_kv_heads × head_dim = 256`):

| Feature | Shape | Notes |
|---------|-------|-------|
| `K_h0[t] \|\| K_h1[t]` | (n_kv_heads × head_dim,) = (128,) | All key vectors concatenated across KV-heads |
| `V_h0[t] \|\| V_h1[t]` | (n_kv_heads × head_dim,) = (128,) | All value vectors concatenated across KV-heads |

Heads are **concatenated, not averaged**.  Averaging is lossy — K vectors from
different heads can point in different directions, so their mean can be
geometrically meaningless.  Concatenation lets the MLP learn per-head weights
independently at the cost of doubling the K/V feature size (64→128 per half),
which is negligible.

Position is already embedded in each `K[t]` via the RoPE rotation, so an explicit
`t/max_len` scalar is redundant.

**Observation includes generated tokens.** At each decode step, the new
generated token's K/V is appended to the cache before the observation is built.
The policy can see the full current cache — prompt tokens and any already-
generated tokens — and evict from either.

**Zeroing is not needed.** In the offline (pre-eviction) design, evicted
positions were zeroed to signal removal. In the online design, eviction removes
a slot entirely (DynamicCache `index_select` compacts the array). The
observation always shows the `cache_size` currently cached tokens,
zero-padded to `max_len`.

**Why not Q?** K and V are what stay in the cache; they are the natural
features for the eviction decision. Q is only needed at decode time, which
differs step-by-step and cannot be pre-computed.

**Why not attention scores?** Computing softmax weights requires
`output_attentions=True`, which forces `attn_implementation="eager"` and
disables FlashAttention. K/V are captured for free from `use_cache=True`
with any attention backend.

**Why not hidden states?** The paper uses the residual stream `h[t]` as
features. This requires hooks and more memory. K/V are a good starting
point; add `h[t]` if the policy underperforms.

---

## Policy architecture

`PerTokenMLP` — a shared MLP applied independently to each token position:

```
Input:  [K_h0[t] || K_h1[t] || V_h0[t] || V_h1[t]]   shape: (2 * n_kv_heads * head_dim,) = (256,)
  → is_real = (input.abs().sum() > 0)         # detect zero-padded slots
  → LayerNorm(K half) || LayerNorm(V half)     # separate norms: K-norms vary
  → Linear(128 → hidden) → LayerNorm → SiLU   #   5–50× across layers
  → Linear(hidden → hidden) → LayerNorm → SiLU
  → Linear(hidden → 1)                        # one scalar keep-score per token
  → score * is_real                           # zero padded slots; blocks gradient
Output: scalar keep-score (0 for padding)   shape: (1,)
```

Applying the same weights to all positions means the policy learns
**token-agnostic importance features** that generalise across prompts and
positions.

The MLP outputs **one scalar per token** (`features_dim = max_len`). SB3
attaches a `Linear(max_len → max_len)` actor head and a `Linear(max_len → 1)`
critic head.

**Per-half LayerNorm**: K and V halves are normalised separately before the
MLP projection. Without this, K-norms vary 5–50× across layers (attention
sinks, massive-activation tokens), causing high-norm layers to dominate
gradients.

---

## Reward

### Motivation

The correctness signal (0 or 1) is sparse — many early episodes score 0.
Reward shaping supplements it with a dense proxy: how much of the model's
attention mass falls on the tokens we kept?

### Two LLM calls per episode (shaping enabled, default)

| Call | When | Purpose | Attn backend |
|------|------|---------|--------------|
| Prefill | episode `reset()` | Full prompt → live DynamicCache | any |
| Clean reference run | episode `reset()` | Full-context generate → per-token importance | `eager` (needs `output_attentions=True`) |

`train.py` automatically selects `attn_implementation="eager"` when
`use_attention_shaping: true`, and the faster `sdpa` otherwise.

### Clean reference run (at reset)

`model.generate()` on the **full** prompt with `output_attentions=True`.
For each (step, layer), query-head attention weights are collapsed within each
GQA group using **max** (SnapKV convention), then averaged over KV-heads and
accumulated across decode steps. The result is `importance[t]` — how much the
model attended to prompt position `t` while producing the full-context answer,
normalised to sum to 1.

The reference run requires `attn_implementation="eager"` — SDPA and
`flash_attention_2` do not return attention tensors and will raise a
`RuntimeError` rather than silently fall back.

To disable shaping entirely and skip the reference run (e.g., on GPU with
`flash_attention_2` where you want maximum throughput):
```yaml
use_attention_shaping: false
```

### Combined terminal reward

```
soft_keep[t]  = mean_l( t still cached in layer l at terminal )   ∈ [0, 1]
alignment     = Σ_t importance[t] × soft_keep[t]                  ∈ [0, 1]
correctness   = flexible_extract(decoded_generated, gold_answer)   ∈ {0, 1}

reward = (1 − attention_weight) × correctness
       + attention_weight       × alignment
```

Default `attention_weight = 0.3`.

`soft_keep[t]` uses the per-layer `slot_to_pos` map maintained during the
episode: for each layer, which original sequence positions are still present in
the live cache. Tokens kept by all 28 layers contribute full alignment weight;
tokens evicted from every layer contribute zero.

`alignment` covers prompt tokens only (importance is defined over positions
0..T-1 from the prefill). Generated tokens contribute 0 to alignment but can
still be evicted — the policy must balance keeping useful prompt context vs.
keeping recent generated context.

All 28 environments receive the same scalar reward (cooperative setting).

`gamma = 1.0` — no discounting. Every decode step contributes equally to
the outcome; discounting would introduce an arbitrary credit bias.

`gae_lambda = 1.0` — Monte Carlo returns: identical advantage estimates for
all steps in an episode. With `gamma=1.0` and `lambda<1`, later steps get
higher advantage weight which has no principled justification here.

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

  ── ROLLOUT COLLECTION  (n_steps × n_envs transitions, no gradients) ──

  for each episode in rollout:

    # ① Prefill — one frozen LLM forward pass per episode
    past_kv = LLM.prefill(prompt)          # torch.no_grad(), use_cache=True
    next_token = prefill_logits[-1].argmax()
    cache_size = T                          # prompt length

    # ② Reference run — full-context generate with output_attentions=True
    #    (only when use_attention_shaping=True; requires attn_implementation="eager")
    #    Raises RuntimeError if backend doesn't support output_attentions (SDPA/flash).
    importance = compute_token_importance(model, input_ids)  # [T], sums to 1

    # budget sampled fresh each episode; budget ≥ T so full prompt fits (when T ≤ budget_max).
    # If T > budget_max, budget = budget_max and eviction fires from step 0.
    budget ~ Uniform(max(budget_min, T), budget_max)  if T ≤ budget_max
           = budget_max                               if T > budget_max

    # ② Free-growth (no RL steps — runs inside reset())
    #    Decode until cache_size > budget.  Every subsequent step_wait() evicts.
    while cache_size <= budget:
        decode one step; cache_size += 1

    # Initial obs per layer: K/V for all T prompt tokens
    obs = _obs(past_kv)                    # [L, max_len, 2D], zero-padded

    # ③ Online decode — up to max_new_tokens steps (only eviction steps; free-growth already done)
    for step in range(max_new_tokens):

        # action[l] = which slot layer l should evict — INDEPENDENT per layer.
        # All layers share the same cache_size = budget+1 (invariant after free-growth).
        # But action[0] ≠ action[1] ≠ … in general: each layer decides for itself.
        action_mask = valid cache slots    # [L, cache_size] — same shape, independent values
        action = policy.predict(obs, mask)          # no_grad; action shape [L]

        # Per-layer eviction — always fires (free-growth guarantees cache_size > budget)
        for l in range(L):
            index_select(past_kv.layer[l], remove=action[l])   # action[l] differs per layer
        cache_size -= 1   # → budget

        # One greedy decode step
        # position_ids = cache_position = true_position (original sequence coordinate).
        # This is correct for the causal mask: row true_position allows attending to
        # all positions 0..true_position-1, which includes every cached token.
        # Passing cache_size instead would wrongly block tokens whose original
        # position > cache_size (common after many evictions of early tokens).
        step_out = LLM(next_token, past_kv, position_ids=[[true_position]], cache_position=[true_position])
        generated.append(next_token)
        past_kv       = step_out.past_key_values   # new K/V appended
        next_token    = step_out.logits[-1].argmax()
        true_position += 1
        cache_size    += 1   # → budget+1 again

        done = (next_token == EOS or step == max_new_tokens - 1)

        reward = 0.0 if not done else terminal_reward()

        store (obs, action, reward, value, log_prob) in rollout_buffer
        if done: break

        obs = _obs(past_kv)               # [L, max_len, 2D] from live cache

    # ④ Terminal reward (only computed once, at episode end)
    text         = tokenizer.decode(generated)
    correctness  = flexible_extract(text, gold)         ∈ {0, 1}
    soft_keep[t] = mean_l( t in slot_to_pos[l] )       ∈ [0, 1]^T
    alignment    = (importance × soft_keep).sum()       ∈ [0, 1]
    reward       = 0.7 × correctness + 0.3 × alignment

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
- Steps ①②③ are all `torch.no_grad()` — the LLM is a black-box oracle.
- Step ⑤ is the only place gradients flow, through the tiny policy MLP only.
- **Per-layer independent eviction**: `action[l]` differs across the 28 layers.
  Layer 0 may evict slot 47 while layer 15 evicts slot 3 at the same step.
  There is no global consensus mask — each layer maintains its own cache.
- The episode is the generation: `generated` tokens accumulate inline; no
  separate "eviction generate" call at terminal.
- Every episode produces up to `max_new_tokens × 28` transitions; only the
  terminal step carries a nonzero reward.
- GAE with `gamma=1.0, lambda=1.0` gives Monte Carlo returns: identical
  advantage for all steps (no temporal credit bias).
- The observation changes each step as new K/V enter the cache (and evicted
  ones are removed), giving the value function a meaningful signal.

---

## What we do NOT do

| What | Why |
|------|-----|
| One-shot ranking (Plackett-Luce) | Sequential gives more gradient signal per episode with sparse reward |
| Per-KV-head independent eviction | DynamicCache stores each layer as `[1, n_kv_heads, seq_len, head_dim]`; `index_select(dim=2)` evicts the same positions from all heads — truly independent per-head caches require a custom attention kernel (paged attention / block-sparse), which standard transformers does not provide |
| Global consensus mask | Each layer applies its own independent eviction via DynamicCache slicing |
| Pre-evict the prompt before generation | Eviction starts during decoding; the policy decides which tokens to drop as the answer unfolds |
| Eager attention for cache eviction | Cache slicing works with any backend; eager is only needed for the optional reference run (`use_attention_shaping: true`) |
| Hidden state `h[t]` features | Good next step if K/V features underfit; adds hook complexity |
| Mean over Q-heads in GQA | Max within each KV-group (SnapKV convention) — mean dilutes tokens critical to specific heads |
| No-op steps in rollout buffer | Free-growth (cache ≤ budget) runs inside `reset()`, not `step_wait()`; every rollout transition is a real eviction decision |
| `ent_coef: 0.0` | Set to 0.01 — with `Discrete(max_len)` and sparse terminal reward, zero entropy bonus collapses the policy to a deterministic local optimum immediately |
| `cache_position = cache_size` | We pass `true_position` for both `position_ids` and `cache_position`; the causal mask row at `true_position` allows attending to all positions 0..true_position-1, which covers every cached token. Using `cache_size` instead would block cached tokens whose original position > cache_size |
| Score zero-padded positions | `PerTokenMLP` multiplies output by `is_real = (input.abs().sum() > 0)`, zeroing padded slots and blocking gradient flow through them |

---

## Success criteria

On the held-out GSM8K test set, the policy achieves correctness within 5
percentage points of the full-cache baseline at the evaluation budget.
Random eviction and KV-norm oracle serve as lower and upper bounds.
Eval reports Wilson 95% confidence intervals; a 5pp difference requires
≥200 examples to be statistically distinguishable.
