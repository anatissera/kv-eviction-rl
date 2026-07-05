# 01 · Building the environment and first rewards (collapse to evict-recent)

**Report section:** §4.3 "Building the environment and first experiments".
**Code:** `kv-eviction-gym/src/kv_gym/` (our own environment, independent of Apple's code).

## Question

Can sequential eviction (1 token evicted per decode step, `Discrete` + action masking,
sb3-contrib's MaskablePPO) be trained on a frozen Qwen2.5-1.5B-Instruct on GSM8K? And
which reward signal should be used: terminal correctness only (`none`), a dense recency
reward (`per_step_recency`), or dense attention shaping (`per_step`)?

## Setup

| component | value |
|---|---|
| model | frozen Qwen2.5-1.5B-Instruct (28 layers, GQA: 12 Q-heads, 2 KV-heads) |
| environment | `SharedKVVecEnv` (later `BatchedSharedKVVecEnv`): one sub-env per layer, shared policy |
| eviction | direct slicing of transformers' `DynamicCache` + `position_ids` update (RoPE) |
| algorithm | MaskablePPO, Monte Carlo returns (gamma=1, lambda=1) |
| main run | overnight 2026-06-21, 520k steps, budget 128/256, `shaping_mode=per_step`, 24-example probe |

Earlier design iterations (documented in the report):
- The first version created 28x12=336 sub-envs (one per Q-head); with GQA only 28x2=56
  physical caches exist, and the decision is made at the layer level (the 2 KV-heads share
  the cache's sequence dimension).
- Episodes where the model emitted EOS before exceeding the budget produced spurious
  transitions (actions with no effect); the environment detects and discards them before
  they reach PPO.

## Result: collapse to evicting what was just generated

The overnight run collapsed to evicting its own recent tokens:

| metric | value |
|---|---|
| `evict_mean_pos_frac` (probe) | 0.45 → 0.91 (evicts recent tokens) |
| `evict_attn_percentile` | → 0.0 |
| eval n=30, budget=200: learned | 0.033 (final) / 0.067 (best_probe), the WORST of all |
| full / kv_norm / random / attn_oracle / streaming | 0.467 / 0.133 / 0.100 / 0.100 / 0.067 |

Two diagnoses:
1. **The attention proxy is useless on GSM8K**: `attn_oracle == random` (0.100) < `kv_norm`
   (0.133). Evicting the least-attended prompt tokens does not beat chance: the reasoning
   chain needs recently generated tokens, which the prompt-attention signal ignores.
2. **Per-step shaping amplified a latent bug**: importance was defined only over PROMPT
   tokens, so evicting a generated token had zero cost at every step, while correctness is
   terminal. The dense bias dominated correctness and produced the collapse.

## Fixes that remained in the code

1. **Hard sink + recency window** (`eval_core.valid_action_mask`): the policy cannot evict
   the first `n_sinks` nor the last `n_recent` slots (StreamingLLM/H2O idea). It is also
   applied to the baselines so the comparison is fair.
2. **`none` mode as the default** (pure correctness + fast `sdpa` backend);
   `per_step_recency` as the repaired dense shaping; `terminal` and `per_step` remain
   selectable as references.
3. **`evict_generated_frac`** metric in the probe: makes the collapse visible at a glance.
4. Always evaluate `best_probe_model`, not `final_model` (the final one may be collapsed).

## What the report says

§4.3: the main result of this stage was identifying the failure mode common to every
reward variant (collapse to discarding essential information) and stabilizing it with the
sink protection + recent window, leaving the environment ready for the comparative
experiments. The choice of which reward to use is picked up again in §3.2 of the report
and in docs [09](09-dense-kl-at-scale.md) and [12](12-capstone-passkey.md).

## Raw data

- `kv-eviction-gym/results_overnight/` (curves + evals of the collapsed run, config included).
- `kv-eviction-gym/results_recency_long/` (early run with the recency reward).
- `kv-eviction-gym/runs/none/` (`none` run with curves and checkpoints).

## Status

Superseded as a numeric result: these runs predate the chat-template fix
([02](02-chat-template-fix.md)), which invalidates all pre-2026-06-29 numbers. The design
fixes (sink/recency window, spurious-transition discarding, collapse metric) remain in
the current code.
