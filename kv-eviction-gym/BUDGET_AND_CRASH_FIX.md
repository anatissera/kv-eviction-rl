# The `_replace_slice` crash and the budget floor (chat-format prompts)

**Status:** analysed 2026-06-29 (Ana Paula). `chat_full_v1` (Alex's control)
crashed at 309k steps. This explains the crash and proposes the fix that makes
BOTH arms (control + S4) crash-proof **without touching `batched_env.py`**.

## The crash

```
_replace_slice tensor size mismatch (195 vs 218) at ~309k steps
```

Root cause (Alex's trace, confirmed in code):
- The budget curriculum annealed `budget_min` to 193.
- A rollout sampled `_shared_budget = 194` → the batched KV is allocated with
  `195 = budget+1` slots, shared by all N episodes.
- Mid-rollout, episode 0 finished and was reset with a new example whose
  **chat-format prompt is 217 tokens** (some GSM8K questions are wordy).
- `batched_env.py:554` does `ep.budget = max(self._shared_budget, T)`. Since
  `T = 217 > 194`, `ep.budget = 217`, and the reset episode's KV grows to
  `218 = T+1` slots.
- `batched_env.py:450` then calls `_replace_slice(batch_kv, b, ep.past_kv, …)`
  to splice the reset episode back into the batch — trying to fit a 218-slot
  cache into the 195-slot batch → **tensor size mismatch → crash**.

The `max(shared_budget, T)` was meant to stop the budget falling below the
prompt length, but it **violates the batched env's uniform-cache-size invariant**
whenever `T > shared_budget`.

## Why the curriculum floor was the real problem

Measured chat-format prompt length `T` over the 1000 training examples
(`Qwen2.5-1.5B-Instruct` tokenizer, `format_gsm8k_chat`, seed 0):

| min | p50 | p90 | p95 | p99 | **max** |
|---|---|---|---|---|---|
| 80 | 116 | 147 | 158 | 186 | **232** |

| threshold | examples with `T >` threshold |
|---|---|
| 120 | **406 (40.6%)** |
| 150 | 85 (8.5%) |
| 200 | 3 (0.3%) |
| 232 | 0 |

With `protect_prompt = true` the prompt cannot be evicted, so the budget must be
**≥ T**. The curriculum `200 → 120` is therefore infeasible: at the 120 floor,
**40.6 % of examples have a prompt longer than the entire budget** — you cannot
fit a 232-token protected prompt into 120 KV slots. So the curriculum either
crashes (current code) or, with a skip fix, silently discards ~40 % of the data
at low budgets — a re-run of the cold-start bias.

## The fix: fixed budget ≥ max(T)

Set a **fixed budget = 256** (no curriculum). Since `256 > max(T) = 232`,
`max(256, T) = 256` for every example → the cache stays uniform → **no crash,
no `batched_env.py` change required**. Compression still happens where it should:
on the **generated CoT** (~110 of ~250 generated tokens evicted at budget 256),
not on the question. This is also the realistic deployment regime — a fixed KV
memory budget, independent of the (unknown-in-advance) output length.

`max_len` must be `> budget + 1 = 257` and `> max(T) = 232`; we use `max_len = 288`.

Applied to `configs/chat_full_s4_kl.yaml` and `configs/chat_smoke_s4_kl.yaml`.

### Recommendation for the A/B (control vs S4)

For the control-vs-treatment comparison to be valid, **both arms must use the
same budget scheme.** Recommend `chat_full_v1` also switch to fixed `budget=256`
(drop the `200→120` curriculum, set `max_len=288`). If a compression *curriculum*
is still wanted, its floor must be `≥ max(T) + n_recent ≈ 264` (e.g. `320 → 264`),
never 120.

If instead the env-level skip fix is preferred (skip examples with
`T > shared_budget` on mid-rollout reset), it stops the crash but should be paired
with a budget floor `≥ ~232`, otherwise low-budget rollouts skip ~40 % of the data.
