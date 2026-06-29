# PLAN — Break the cold-start and measure causal reward shaping (answers HANDOFF.md)

Start date: 2026-06-29. Authors: Ana (+ Claude). Read `HANDOFF.md` (Alex's) first.

## Consolidated diagnosis (where we start from)

Every run so far died against the **same wall**, regardless of the reward:

| Run | Reward | Result |
|---|---|---|
| `corr_reward_v1` (Alex) | correctness + entropy | 100% trunc, retention=0, evict_generated_frac≈0.90 |
| `s4_kl_v1` (ours) | correctness + KL-to-full (S4-exact) | 100% trunc, retention 0-5%, identical pattern |

**Root cause (not the reward, it is structural):** the *free-growth skip*
(`batched_env._reset_episode`: `if eos_id in fg_tokens: continue`) silently discards
every example where EOS appears during free-growth → the training set is biased
towards purely hard examples (gen_len ≫ max_new_tokens) → **every episode truncates →
correctness=0% → EV=NaN → nothing is learned.** The `FreeGrowthCache` (694 entries) contains
ONLY hard examples by construction.

**Key handoff finding (rules out one route):** the attention signal is **useless for GSM8K**
(`attn_oracle = 0.100 = random`, worse than `kv_norm = 0.133`). Warm-starting from attention is
NOT worth it. The best heuristic is `kv_norm`.

## Hypotheses

- **H0 (gate):** with a genuinely easy example set (gen_len < 400) + prompt protection,
  **correctness fires** (trunc < 100%, `correct_learned > 0`) within the first rollouts.
  Without this, nothing else is measurable.
- **H1 (ours):** once the task is learnable, **causal S4 KL shaping** improves retention
  and/or learning speed vs pure correctness.
- **H2 (collapse):** prompt protection + a recency window prevents the collapse to
  evict-recent-generated (`evict_generated_frac` stops pinning at ~0.90).

## Two unblocks NOBODY has implemented yet

1. **Easy example set:** `find_easy_examples.py` (Alex wrote it, **never ran it** —
   `per_example_budgets_easy.json` does not exist). Greedy full-cache inference over the 1000
   examples, keeping `gen_len ∈ [150,400]`. Those are the examples the skip was throwing away.
2. **Prompt protection:** today `valid_action_mask` only protects `n_sinks + n_recent`, **NOT**
   the question. The agent can evict the question. The right inductive bias: preserve the
   question, compress only the generated reasoning.

---

## PHASE 0 — Unblock (1 VM, sequential). IN PROGRESS.

0a. **Prompt protection** (code, no VM):
    - `eval_core.valid_action_mask(..., n_prompt=0)`: protects the first `max(n_sinks, n_prompt)`
      slots and the last `n_recent`. Graceful degradation: if the window covers everything, it
      falls back to sink+recent only before relaxing to all-evictable (never an empty mask).
    - Thread `n_prompt` through `_valid_slots`, `make_{learned,kv_norm,random}_evict_fn`,
      `run_online_episode(protect_prompt=...)` so train and eval use the SAME rules.
    - `batched_env.action_masks`: `n_prompt = ep.prompt_len if self.protect_prompt else 0`.
    - `protect_prompt` flag in config → `train.py` → env and probe.

0b. **Generate the easy set** (1 VM, GPU, ~30-60 min):
    - Run `find_easy_examples.py --max-gen-len 400 --min-gen-len 150
      --output per_example_budgets_easy.json`.

0c. **Validation smoke** (1 VM, ~10-15 min):
    - Small config over the easy set + `protect_prompt: true`.
    - **H0 success criterion:** `truncation_rate < 1.0` and `correct_learned > 0` in the first
      rollouts. If it does not fire → debugging THIS is the work (parallelise nothing).

## PHASE 1 — Real A/B (both VMs in parallel), ONLY if Phase 0 passes H0

Everything identical except the reward. Easy set + `protect_prompt: true` on both.

- **VM A (control):** pure correctness (`run_none` over the easy set).
- **VM B (treatment):** correctness + **S4 KL shaping** (our causal contribution, `kl_shaping`).

Headline: the probe's `retention` curve (B ≥ A, ideally rising earlier/higher).
Collapse diagnosis: healthy `evict_generated_frac` on both.

## PHASE 2 — Optional / future

- A **repaired** recency-attention shaping arm: `per_step_recency` (NOT raw attention —
  the evidence says attn==random). Only if the A/B leaves an appetite for it.
- Warm-start from `kv_norm` as an accelerator (it is not the blocker; from-scratch will
  probably be enough once correctness fires).

## Operations / cost

- VMs: `ppo-kvp` (ours, tp-final-rl-kv-eviction/us-central1-b) + `kv-none-v2` (Alex's,
  proyecto-final-425415/us-west4-a). **Both stopped** while we plan.
- Cost-sensitive (edu credits): ALWAYS `stop` the VM when done. NEVER kill a live run of
  Alex's without authorisation (kv-none-v2 is idle now, authorised to stop it).

## Invariants to maintain (from HANDOFF.md)

- `min_cached_len ≥ eviction_k_end`; `base_budget[i] - eviction_k_end ≥ T[i]`;
  `max_len ≥ max(base_budget)+1`; `max_new_tokens ≥ max(base_budget) - T_min`.
- With prompt protection: you need `budget ≥ T + n_recent` for there to be evictable slots
  when eviction fires. The easy set (T≈91, gen_len≥150, n_recent=32, K≤100) satisfies it.
