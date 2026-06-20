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

**Features per token position t** (`feature_dim = 2 × head_dim = 128`):

| Feature | Shape | Notes |
|---------|-------|-------|
| `K[t]` | (head_dim,) | Full key vector — RoPE already applied, encodes content + position |
| `V[t]` | (head_dim,) | Full value vector |

Position is already embedded in `K[t]` via the RoPE rotation, so an
explicit `t/max_len` scalar is redundant.

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

### Motivation

The correctness signal (0 or 1) is very sparse — most episodes end with
score 0 until the policy is nearly optimal.  Reward shaping supplements it
with a dense proxy: how much of the model's **future attention** falls on
the tokens we kept?

### Two LLM calls per episode (shaping enabled, default)

| Call | When | Purpose | attn backend |
|------|------|---------|-------------|
| Clean reference run | episode `reset()` | Full-context generate → per-token importance | `eager` (needs attention weights) |
| Eviction generate | episode terminal | Masked generate → correctness score | any |

`train.py` automatically selects `attn_implementation="eager"` when
`use_attention_shaping: true`, and the faster `sdpa` otherwise.

### Clean reference run (at reset)

`model.generate()` on the **full** prompt with `output_attentions=True`.
Produces `importance[t]` = mean attention mass on prompt position `t` from
all generated tokens, averaged over all layers and heads, normalised to
sum to 1.

**Fallback**: if the attention backend returns empty matrices (SDPA / flash),
we use `importance[t] ∝ ||K[t]|| + ||V[t]||` (KV-norm heuristic, the same
signal as the oracle baseline). Shaping always provides a signal.

### Eviction generate (at terminal)

`model.generate()` with the attention mask that zeros out evicted positions.
Explicit `position_ids = [0, 1, ..., T-1]` ensure each surviving token
retains its original RoPE rotation.

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

All 56 environments receive the same scalar reward (cooperative setting).

`gamma = 1.0` — no discounting. Since the observation is constant
throughout the episode, every eviction step contributes equally to the
outcome; discounting would introduce an arbitrary credit bias where later
evictions appear to deserve more reward than earlier ones.

---

## Training loop

Gradients **never flow through the LLM**. The LLM is frozen throughout;
only the policy MLP (PerTokenMLP + PPO actor/critic heads) is trained.

```
# Frozen: Qwen2.5-1.5B weights
# Trained: PerTokenMLP (≈ 2 × hidden × feature_dim params)

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

    obs = [K[t], V[t]]  for t in 0..T-1   # constant throughout episode

    # ③ Sequential eviction (T − budget steps per env)
    for step in range(T - budget):
        action_mask = resident_tokens        # which positions still valid
        action = policy.predict(obs, mask)   # no_grad: rollout collection
        evict token[action] from resident

        if not done:
            reward = 0                       # intermediate steps: no reward

        if done:                             # last eviction step
            # ④ Eviction generate: score correctness of kept tokens
            text = LLM.generate(            # torch.no_grad()
                input_ids, attention_mask=kept_tokens
            )
            correctness = (text matches gold)   # ∈ {0, 1}
            alignment   = sum(importance[kept])  # ∈ [0, 1]
            reward = 0.7 × correctness + 0.3 × alignment

        store (obs, action, reward, value, log_prob) in rollout_buffer

  ── PPO UPDATE  (n_epochs passes over the rollout, WITH gradients) ───

  compute GAE advantages from rollout_buffer  (γ=1.0, λ=0.95)

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
- Every episode produces `(T − budget) × 56` transitions in the rollout buffer,
  all with reward 0 except the terminal step which carries the shaped reward.
- GAE with `gamma=1.0` propagates the terminal reward backwards with no
  discounting, so all eviction steps get identical advantage estimates (up to
  the baseline subtraction by the value function).

---

## What we do NOT do

| What | Why |
|------|-----|
| One-shot ranking (Plackett-Luce) | Sequential gives more gradient signal per episode with sparse reward |
| Eager attention for the training forward passes | Only used once per episode (clean reference run); main prefill uses `sdpa` or `flash` |
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
