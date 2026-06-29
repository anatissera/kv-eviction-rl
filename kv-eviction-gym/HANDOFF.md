# kv-eviction-gym — Handoff (sequential KV-eviction RL on GSM8K)

Status as of this commit. Read this first if you're a new session picking up the work.

## What this is

Sequential KV-cache eviction as a Gym MDP trained with SB3 `MaskablePPO` on GSM8K
(Qwen2.5-1.5B). One shared `PerTokenMLP` policy stepped by `n_envs = n_layers` (28); per-layer
eviction via DynamicCache slicing; frozen LLM in the loop (gradients only through the policy).
See `METHOD.md` for the full design. Reward = correctness (answer-extraction) ± attention
shaping. Eval/monitoring compare against full / streaming / attn-oracle / kv_norm / random.

## What ran (overnight, 2026-06-21) and what we learned

Config: tuned-for-feasibility (n_steps=128, max_len=288, budgets 128/256, max_new_tokens=256),
`shaping_mode=per_step` (dense attention shaping), `total_timesteps=520k` (~145 rollouts ~100
episodes, ~3.8h on an L4 with eager attention). Probe: 24 fixed held-out test problems every 8
rollouts. Results in `results_overnight/`.

**Key finding — the policy COLLAPSED to evicting its own most-recent generated tokens:**
- probe `evict_mean_pos_frac` 0.45 → 0.91 (evicts recent), `evict_attn_percentile` → 0.0.
- eval (n=30, budget=200): learned **0.033** (final) / **0.067** (best_probe_model), WORST of all;
  full=0.467, kv_norm=0.133, random=0.100, attn_oracle=0.100, streaming=0.067.

**Two diagnoses that drive the current fixes:**
1. **The attention-importance proxy is uninformative for GSM8K**: `attn_oracle == random` (both
   0.100) < kv_norm (0.133). Evicting "least-attended prompt tokens" is no better than random
   here — the chain-of-thought needs recent *generated* tokens, which the prompt-attention signal
   ignores.
2. **The per-step shaping amplified a latent bug**: `importance` is defined over PROMPT tokens
   only, so generated tokens have importance 0 → `−w·importance[evicted]` makes evicting a
   generated token the max-reward (zero-cost) move *every step*, while correctness is sparse
   (terminal). The dense bias overwhelmed correctness → collapse. (Alex's original `terminal`
   shaping had the same latent bias but kept it in check by sharing one terminal scalar with
   correctness.) **The per-step form was the mistake.**

Also note: `best_probe_model` (saved at the probe-retention peak, ~55% through training) evals
better than `final_model` (saved at the end, where retention had collapsed to 0). Always eval
`best_probe_model`, not `final_model`.

## What this commit changes (the fixes)

1. **Hard recency + sink window** (`eval_core.valid_action_mask`, used by `env.action_masks()` and
   all eval/probe evict-fns): the policy literally cannot evict the first `n_sinks` or last
   `n_recent` cache slots (StreamingLLM/H2O/SnapKV standard). Structural, ungameable; directly
   prevents the "evict recent generated" collapse. Applied to baselines too (random/kv_norm/attn)
   so comparisons are fair under identical rules.
2. **Shaping modes** (`shaping_mode` in config):
   - `none` (NEW DEFAULT): pure correctness. Recommended primary — optimizes the true objective,
     avoids the misleading attention proxy, and (with `use_attention_shaping: false` +
     `attn_implementation: sdpa`) trains MUCH faster (no per-episode reference-attention generate;
     eager was the bottleneck at ~64s/rollout).
   - `per_step_recency` (NEW): the repaired dense shaping. Cost `−w·keep_value[p]`,
     `keep_value = max(norm_attn_importance[p], recency_value[p])` ∈ [0,1] — recent tokens (prompt
     OR generated) become costly to evict, removing the "generated = free" bug. No new hyperparam.
   - `terminal`: Alex's original `0.7·correctness + 0.3·alignment` (reference/comparison only).
   - `per_step`: the unfixed mode that caused the collapse — kept selectable for reproduction.
3. **`probe/evict_generated_frac`** metric (fraction of evictions that hit generated tokens) — the
   metric that makes this failure mode visible at a glance (was ~1.0 in the collapse).

## How to run (on a GPU VM with the venv set up — see "Environment" below)

```bash
cd kv-eviction-gym
export VIRTUAL_ENV=$PWD/.venv; export HF_HOME=$HOME/.cache/huggingface
PY=.venv/bin/python

# Smoke first (tiny, ~5 min) — validates the whole pipeline + the fix:
$PY scripts/train.py --config configs/quickstart.yaml --run-name smoke
#   check runs/smoke/probe_curve.csv: evict_generated_frac should NOT be ~1.0 and
#   evict_mean_pos_frac should NOT pin to ~0.9 (window prevents dumping recent tokens).

# The three arms (each ~3-4h on an L4; run_none is fastest via sdpa):
$PY scripts/train.py --config configs/run_none.yaml     --run-name none      # PRIMARY (pure correctness, fast)
$PY scripts/train.py --config configs/run_recency.yaml  --run-name recency   # dense shaping, bias fixed
$PY scripts/train.py --config configs/run_terminal.yaml --run-name terminal  # Alex's reward (reference)

# Plot + eval (ALWAYS eval best_probe_model, not final_model):
$PY scripts/plot_curves.py --run runs/none
$PY scripts/eval.py --model runs/none/best_probe_model --config configs/run_none.yaml \
    --budget 200 --n 30 --output runs/none/eval_results_best.json
```

Recommended A/B: **`run_none` vs `run_recency`** (terminal as a reference point). Compare learned
vs kv_norm/random under the SAME window. The win condition: learned retention rising above the
random/kv_norm anchors on the probe, and learned ≥ kv_norm on the held-out eval with
non-overlapping-ish CIs.

## Config knobs that matter

`shaping_mode`, `attention_weight` (only used by terminal/per_step*), `n_sinks`/`n_recent`
(window), `n_steps` (rollout buffer size — keep ≤128 to fit RAM; 524 OOMs a 32GB box),
`max_len`≥`budget_max+1`, `probe_n`/`probe_every_n_rollouts`/`probe_budget` (∈[budget_min,budget_max]),
`total_timesteps`. The full paper config (10M steps) is infeasible with eager attention
(~130h); that's why `run_none` (sdpa) matters for getting enough episodes.

## Environment (fresh VM)

```bash
cd kv-eviction-gym
uv venv .venv --python 3.10
export VIRTUAL_ENV=$PWD/.venv
uv pip install -e . matplotlib tensorboard
# VM needs OUTBOUND internet (pip + HF download). If the instance has no external IP:
#   gcloud compute instances add-access-config <vm> --zone <z> --access-config-name external-nat
# (SSH still works via --tunnel-through-iap regardless.)
```
Two env-compat fixes are already applied in `src/kv_gym/vendor/gsm8k.py` (dataset id
`openai/gsm8k`, not the bare `gsm8k` the newer `datasets` rejects). `tensorboard` must be
installed (SB3 errors without it when `tensorboard_log` is set).

## Cost / ops
Cost-sensitive (edu credits). Always stop the VM when not actively training — see the autonomous
watcher pattern used previously (`kvgym_overnight_watch.sh` in the parent repo dir): polls for a
done-marker or crash, downloads `runs/<name>/`, and ALWAYS stops the VM.

## Files
- `src/kv_gym/eval_core.py` — shared online-eviction core + `valid_action_mask` (window).
- `src/kv_gym/env.py` — `SharedKVVecEnv`; window in `action_masks()`; shaping modes in `step_wait`.
- `src/kv_gym/probe.py` — `EvalProbeCallback` (fixed-probe monitoring + `evict_generated_frac`).
- `scripts/{train,eval,plot_curves}.py`; `configs/run_{none,recency,terminal}.yaml`; `METHOD.md`.
- `results_overnight/` — the collapse run's curves + evals (model zips excluded; reproducible).

---

## Phase 2: Batched env, cold-start diagnosis, and per-example budget attempt

*This section documents the work done in the sessions following the overnight run fixes above.*

### Architecture shift: BatchedSharedKVVecEnv + EpisodeMaskablePPO

The old `SharedKVVecEnv` ran `n_envs = n_layers = 28` environments (one per transformer layer),
with one forward pass per env step. Each "episode" spanned all 28 layers' steps simultaneously —
but SB3 thought it was running 28 independent envs. This worked but was wasteful.

The new architecture (`src/kv_gym/batched_env.py`) runs `N=2` parallel *problems* in one batched
forward pass, so `n_envs = N × n_layers = 56`. Key classes:

- **`BatchedSharedKVVecEnv`**: Wraps N problems, runs one forward pass for all N×n_layers envs.
  Holds a `FreeGrowthCache` to pre-seed KV caches from stored free-growth trajectories (avoids
  re-running the prompt + free-growth phase from scratch on every episode).
- **`EpisodeMaskablePPO`**: Custom MaskablePPO subclass with episode-level rollout collection.
  `repeats_per_problem=5` keeps the same 2 problems for 5 consecutive rollouts before rotating.
- **`BudgetCurriculumCallback`**: Anneals `budget_min`, `budget_max`, and `_eviction_k` linearly
  over `curriculum_fraction × total_timesteps` steps. Called at the start of each rollout.
- **`FreeGrowthCache`** (`src/kv_gym/free_growth_cache.py`): Disk-backed cache of tokenized
  free-growth trajectories keyed by example index. Stores the token IDs generated during the
  "free growth" phase (unconstrained generation before the eviction window). Only populated for
  examples where EOS did NOT appear during free-growth (see cold-start section below).

### The cold-start problem (root cause of 100% truncation)

**All experiments up to per_example_budget_v1 showed 100% episode truncation, zero correctness
reward, and `explained_variance=NaN`.** Here is the exact root cause:

**The free-growth skip logic** (`batched_env.py`, `_reset_episode`):
```python
if self.eos_id in fg_tokens:
    continue   # discard this episode and retry with a new example
```
When `budget` is set high enough that the LM finishes naturally (EOS appears during free-growth),
the episode is SILENTLY DISCARDED. This means:

- **The training set is structurally biased toward hard examples.** If EOS appears during
  free-growth → skip. Only examples where EOS does NOT appear survive into training.
- **The FreeGrowthCache contains ONLY hard examples.** Its 694 entries are examples where the LM
  needed > budget tokens to finish. Median `cached_len = 422` tokens generated after the prompt
  before being cut off.
- **Hard examples = gen_len >> max_new_tokens.** GSM8K on Qwen-1.5B has a long tail. The cached
  examples require 800+ tokens to finish. With `max_new_tokens=800`, these episodes ALWAYS
  truncate during RL, regardless of eviction quality.

**Result:** budget=any → all episodes truncate → correctness=0% → EV=NaN → no learning.

### Experiments that failed (all suffer the cold-start problem)

| Run name | What we tried | Outcome |
|---|---|---|
| `corr_reward_v1` | Entropy reward insurance, basic training | 100% truncation, EV=NaN, killed |
| `repeat_v1` | `repeats_per_problem=5`, curriculum on budget | Best: 10.5% retention at 1.42M steps, still 100% truncation |
| `repeat_entropy_v1` | repeat + entropy combined | Same, killed |
| `easy_start_v1` | `budget_min_start=590` (high budget → easy start) | 100% truncation (skip logic still fires!), killed at 280k steps |

Note: `easy_start_v1` was an attempt to start with a very high budget so EOS would appear during
the RL phase. It failed because the skip logic fires BEFORE the RL phase — during free-growth.
The budget would need to be ≫ the model's natural gen_len (800+) for EOS to appear during RL,
but `max_new_tokens=800` cuts all episodes there anyway.

### Per-example budget calibration attempt (per_example_budget_v1)

**Idea:** Instead of a shared budget, calibrate per-example:
```
base_budget[i] = T[i] + cached_len[i]
ep.budget[i]   = base_budget[i] - eviction_k     (applied at runtime)
```

With `eviction_k=20` (easy start):
```
n_needed = ep.budget + 1 - T = cached_len - 20 + 1 = cached_len - 19
```
Since `n_needed < cached_len`, the cache lookup is always a full hit → the first `n_needed`
cached tokens have no EOS (guaranteed by cache construction) → episode NOT skipped by the
skip logic.

The K-curriculum (`eviction_k_start=20 → eviction_k_end=150`) was designed to be the "lower
the budget over time" mechanism: start easy (20 eviction steps required) and grow harder.

**Why it still failed** (100% truncation after 1 rollout, training crashed):

The free-growth skip is solved — cached examples are no longer skipped. But:

1. **gen_len >> max_new_tokens for ALL cached examples.** By construction, the 694 cached examples
   have `gen_len > cached_len ≈ 422`. With `K=20`: free-growth uses `cached_len - 19 ≈ 403`
   tokens. The RL phase then has at most `max_new_tokens - T - (cached_len-19) ≈ 800 - 91 - 403
   = 306` remaining token slots. If `gen_len > T + 403 + 306 = 800`, EOS never appears.
   Since all cached examples have `gen_len > 422` (they were cached BECAUSE they're hard), and
   the median is likely 500-700+, truncation is nearly universal.
2. **Random eviction destroys the problem statement.** With only `n_sinks=4 + n_recent=32 = 36`
   protected slots, and prompts averaging ~91 tokens, roughly 82 of 91 prompt tokens are
   evictable. Random eviction destroys the problem before the model can answer.

**The training run:** 1 rollout, then tmux session died (likely VM preemption — GCP preemptible
instance). Stats: `ep_rew_mean=-1.571`, `ep_len=348`, `correctness=0%`, `truncation=100%`.

### The correct next step: find easy examples via full-cache inference

The fundamental fix is to identify examples where the model CAN finish within `max_new_tokens`
under ideal conditions (no eviction, full KV cache). These are the examples currently SKIPPED
by the free-growth logic.

**Script to write: `scripts/find_easy_examples.py`**

```python
# Run greedy full-cache inference on all 1000 training examples.
# No eviction, do_sample=False, max_new_tokens=600.
# Record gen_len for each example.
# Save: {"42": 187, "137": 312, ...}  (example_idx → gen_len)
```

**Filter:** Keep examples with `gen_len < 400`. These are guaranteed to finish during the RL
phase even if the agent makes suboptimal eviction decisions (leaving some budget slack).

**Expected yield:** On GSM8K / Qwen-1.5B, roughly 100-300 examples complete in < 400 tokens.
These are the genuinely easy ones. The ~700 "hard" ones have long chain-of-thought answers.

**Then:** Use `gen_len` to set `base_budget[i] = T[i] + gen_len[i] + buffer` (e.g. buffer=50).
With `eviction_k` curriculum: start at K=gen_len[i] (zero headroom) → anneal down (more
compression required). This guarantees EOS appears during RL for the calibrated examples.

**Alternative shortcut:** grep the existing 694 cached entries for those with `cached_len < 300`
— these are likely examples where the model nearly finished during free-growth (EOS appeared
close to the free-growth cutoff). They're still "hard" (EOS not in cache), but gen_len is
closer to the budget. However, this is a less reliable proxy than full-cache inference.

### New files and code changes

**`src/kv_gym/batched_env.py`** (major new file):
- `BatchedSharedKVVecEnv.__init__`: accepts `per_example_base_budgets: dict[int,int] | None`
  and `eviction_k: int`. These enable per-example budget calibration.
- `_reset_episode`: selects budget from `per_example_base_budgets[i] - eviction_k` if available,
  otherwise falls back to the shared `budget_min/budget_max` range.
- `_eviction_k` attribute is annealed by `BudgetCurriculumCallback` at rollout start.

**`scripts/train.py`** (modified):
- `BudgetCurriculumCallback.__init__`: now accepts `budget_max_start/end` and
  `eviction_k_start/end` in addition to `budget_min_start/end`.
- Per-example budget loading: reads `per_example_budgets_path` from config, filters the example
  list to only calibrated indices, remaps indices to 0-based, passes to env.

**`scripts/compute_per_example_budgets.py`** (new):
- Reads FreeGrowthCache + tokenizes prompts to compute `base_budget[i] = T + cached_len`.
- Filters by `min_cached_len=150` (ensures full cache hit even at `eviction_k_end=150`) and
  budget range `[350, 750]`.
- Output: `per_example_budgets.json` — values are T + cached_len (NO K subtracted; K applied
  at runtime). On 694 cached entries: 610 pass filters.

**`configs/per_example_budget_v1.yaml`** (new):
- Full config for the calibrated training run. See file for all hyperparameters.

### VM deployment state

VM: `kv-none-v2`, zone `us-west4-a`. **Not a git repo** — files are SCP'd manually.

```bash
# SSH via IAP (no external IP needed):
gcloud compute ssh kv-none-v2 --zone us-west4-a --tunnel-through-iap

# Deploy code changes from local repo:
# (IAP tunnel supports only 1 connection at a time — do SCP sequentially)
for f in src/kv_gym/batched_env.py scripts/train.py scripts/compute_per_example_budgets.py \
          configs/per_example_budget_v1.yaml; do
    gcloud compute scp --zone us-west4-a --tunnel-through-iap \
        "$f" kv-none-v2:~/kv-eviction-gym/"$f"
done

# Launch training (ALWAYS mkdir the run dir before tmux to fix tee log issue):
mkdir -p ~/kv-eviction-gym/runs/per_example_budget_v1
tmux new-session -d -s train4 'cd ~/kv-eviction-gym && \
    export VIRTUAL_ENV=$PWD/.venv HF_HOME=$HOME/.cache/huggingface && \
    .venv/bin/python scripts/train.py \
        --config configs/per_example_budget_v1.yaml \
        --run-name per_example_budget_v1 \
    2>&1 | tee runs/per_example_budget_v1/train.log'

# Monitor:
tmux attach -t train4
tail -f ~/kv-eviction-gym/runs/per_example_budget_v1/train.log

# ALWAYS stop the VM when done (edu credits):
gcloud compute instances stop kv-none-v2 --zone us-west4-a
```

Files already on the VM:
- `~/kv-eviction-gym/per_example_budgets.json` — 610 calibrated base budgets
- `~/.kv_eviction_cache/qwen-1.5b/` — 694 cached free-growth trajectories

### Recommended next actions (in order)

1. **Write `scripts/find_easy_examples.py`**: Run greedy full-cache inference, record gen_len per
   example, save JSON. Filter to `gen_len < 400`. These are your real training set.

2. **Generate new `per_example_budgets.json`** using find_easy_examples output:
   `base_budget[i] = T[i] + gen_len[i] + 50` (the +50 gives budget slack even with bad eviction).
   K-curriculum: `eviction_k_start = gen_len_median` (need some evictions to earn the token
   budget), `eviction_k_end = gen_len_median + 100` (harder compression).

3. **Protect prompt tokens from eviction.** The current action space evicts any non-sink/recent
   token, including the 91-token problem prompt. Add `n_prompt = T` protected tokens at the head
   (after the sinks) to the valid_action_mask, so only GENERATED tokens are evictable. This is
   the sensible inductive bias: preserve the question, compress the intermediate work.

4. **Restart training** with the new budget file and prompt protection. With gen_len < 400
   examples and proper budget calibration, expect correctness reward to fire in the first few
   rollouts.

5. **If correctness still doesn't fire** after 5 rollouts: check `ep_len` distribution. If
   episodes are hitting `max_new_tokens` before EOS, the budget math is still off — print
   `base_budget`, `eviction_k`, `T`, `cached_len` per episode to debug.

### Key invariants to maintain

- `min_cached_len ≥ eviction_k_end`: guarantees full cache hit even at hardest curriculum stage.
- `base_budget[i] - eviction_k_end ≥ T[i]`: budget can't go below prompt length.
- `max_len ≥ max(base_budget) + 1`: model's position embedding limit.
- `max_new_tokens ≥ max(base_budget) - T_min`: total generation cap must exceed max budget.

---

## Phase 3: Chat template fix — cold-start SOLVED (2026-06-29)

*Full write-up in `CHAT_TEMPLATE_FIX.md` (Ana Paula's analysis). This section summarises the
outcome and documents the active overnight run.*

### Root cause: plain-text prompts to an Instruct model

Every run up to this point fed raw GSM8K text (`format_gsm8k`) directly to
`Qwen2.5-1.5B-**Instruct**`. Instruct models are fine-tuned to generate inside a chat frame and
emit EOS only when they see `<|im_end|>` — which only appears in the correct chat template. Without
it, the model generates 900+ tokens of repetition and never emits EOS. Evidence:

- Plain-text: 0/300 episodes finish (all truncated).
- Chat template (`format_gsm8k_chat`): 5/5 episodes finish in a quick sanity check.

**Per-example budgets and `find_easy_examples.py` are no longer needed.** With the chat template,
gen_len ≈ 100–400 tokens for essentially all GSM8K examples. `budget=200` leaves ≈ 90 tokens of
free-growth followed by a ≈ 110–310-step RL phase, during which EOS appears naturally.

### Fix applied

`src/kv_gym/vendor/prompts.py` — new function `format_gsm8k_chat`:
```python
def format_gsm8k_chat(tokenizer, example):
    raw, max_new = format_gsm8k(example)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": raw}],
        tokenize=False, add_generation_prompt=True,
    )
    return text, max_new
```
`src/kv_gym/batched_env.py` now imports and calls `format_gsm8k_chat`.

**Free-growth cache invalidated.** The 694 entries in `~/.kv_eviction_cache/qwen-1.5b/` were built
with the plain-text format and must NOT be reused. Delete them before starting new runs (or set
`free_growth_cache_dir` to a fresh path). The cache will be rebuilt automatically from the first
training run. After the smoke run, 83 new entries were built.

### Smoke run results (`chat_smoke_v1`, 120k steps, 2026-06-29)

Config: `configs/chat_smoke_v1.yaml` — budget=200 fixed, no curriculum, `protect_prompt=true`,
1000 examples, `repeats_per_problem=2`, `probe_n=4`.

| Metric | Rollout 1 | Final (122k steps) |
|---|---|---|
| truncation_rate | 0% | 0% |
| correct | 100% | 100% |
| ep_rew_mean | 0.789 | 0.468 |
| ep_len_mean | 121 | 169 |
| correct_learned | — | 0.750 (= random) |
| correct_full | — | 1.000 |
| evict_generated_frac | — | 1.00 (expected with protect_prompt) |
| explained_variance | — | −0.521 (value fn just starting) |

Conclusion: training loop works end-to-end. `correct_learned = correct_random` at 120k steps is
expected — the value function needs more steps to separate signal from noise.
Models saved at `runs/chat_smoke_v1/` on VM `kv-chat-v1`.

### Active overnight run: `chat_full_v1`

VM: **`kv-chat-v1`**, zone `us-west1-a`, g2-standard-8 (L4 GPU), **non-preemptible (STANDARD)**.

Config: `configs/chat_full_v1.yaml`. Key differences from smoke run:
- `total_timesteps: 5_000_000` (~5M steps, estimated 10–14h)
- `budget_min_start=200 → budget_min_end=120` over first 80% of training (compression curriculum)
- `budget_max=200` (fixed ceiling; shared budget sampled from `[budget_min, 200]` each rollout)
- `repeats_per_problem=5` (more on-policy signal per problem before rotating)
- `probe_n=32`, `probe_every_n_rollouts=15` (stable eval with less overhead)

Run launched 2026-06-29 ~14:25 UTC-3. Session: `tmux attach -t chat_full_v1`.
Log: `~/kv-eviction-gym/runs/chat_full_v1/train.log`.

**Note:** run_name is auto-generated as a timestamp (e.g. `20260629_142502`), NOT `chat_full_v1`.
The log and models live in `runs/<timestamp>/`, not `runs/chat_full_v1/`.

```bash
# Monitor:
gcloud compute ssh kv-chat-v1 --zone=us-west1-a
tmux attach -t chat_full_v1
# or:
tail -f ~/kv-eviction-gym/runs/<timestamp>/train.log

# Download results when done:
gcloud compute scp --recurse kv-chat-v1:~/kv-eviction-gym/runs/<timestamp>/ \
    ./runs/chat_full_v1_results/ --zone=us-west1-a

# ALWAYS stop VM when done:
gcloud compute instances stop kv-chat-v1 --zone=us-west1-a
```

### What to look for in results

- `correct_learned > correct_random (0.75)` on the probe → policy learning selective eviction
- `evict_generated_frac` staying at 1.0 is CORRECT (prompt protected, only generated tokens evictable)
- `evict_mean_pos_frac` drifting below 0.77 → policy evicting older generated tokens (good)
- `explained_variance` rising above 0 → value function learning the reward structure
- If `correct_learned` never separates from `correct_random` by 1M steps → consider extending
  `repeats_per_problem` or adding attention shaping (`shaping_mode: per_step_recency`)

### Reward structure (for reference)

At each RL step: agent picks one KV slot to evict (from generated tokens only, excluding first
`n_sinks=4` and last `n_recent=32`). At episode end (EOS emitted):
- `+1.0` if answer extraction finds the correct number, `0.0` if wrong
- `−length_penalty_weight × ep_len / max_len` (discourages padding / stalling)
- `−truncation_penalty` if EOS never appeared (shouldn't happen with chat template)
- `+entropy_reward_weight × H(π)` per step (exploration)

All rewards propagated via GAE with `γ=λ=1.0` (undiscounted, full-return). The policy must learn
that early eviction choices causally affect whether the model can correctly answer 100–300 steps later.
