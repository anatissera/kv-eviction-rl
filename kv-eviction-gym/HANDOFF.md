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
