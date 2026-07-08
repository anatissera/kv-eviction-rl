# 08 · E6/E7: long generations and the exploration limit

**Report section:** §4.11 "Long generations and the exploration limit".
**Configs:** `kv-eviction-gym/configs/e6_long.yaml`, `e6b_long192.yaml`, `e6c_long256.yaml`,
`e7_repeat.yaml`.

## Question

The wide eval ([05](05-wide-eval-regime.md)) showed that the learnable headroom appears
in long generations under cache pressure. With the environment moved to that arena, can
online PPO with a terminal reward learn anything there?

## Arena calibration (E6, E6b, E6c)

GSM8K filtered with `min_answer_words=60` (2208 examples, mean generation ~285 tokens):

- **E6 (budget 160): OVERSHOT regime.** Train correctness 0.000 on every rollout,
  truncation 0.8 → 1.0: under that much pressure the model solves nothing, constant
  reward, zero signal (the mirror image of the benign regime).
- **budget 192**: also overdone.
- **E6c / long-gen wide eval (budget 256, n=128):** full=0.680, **random=0.227,
  kv_norm=0.211**. `full - random = 0.45`: huge margin, the task remains solvable, and
  **kv_norm does not even beat chance**. The regime knob's useful window is narrow: the
  task must remain solvable under eviction but the strategy has to matter.

## E7: high repetitions + regret baseline

Exposure analysis: each episode costs ~4900 timesteps (~174 steps x 28 layers), so a
"2M-step" run completes ~287 episodes ≈ 57 examples x 5 passes. With a 0/1 terminal
reward, 5 samples per example give SE ≈ 0.32 against an effect of ~0.05: the signal sits
6x below the noise floor. E7's levers:

1. Pool of 32 long-gen examples, `repeats_per_problem: 50`, 8M timesteps
   (~50 exposures per example).
2. **Per-example regret baseline** (`per_example_baseline: true` in
   `batched_env._terminal_reward`): the terminal reward is centered with the same
   example's running mean, removing the between-example difficulty variance (±0.45).
3. Filtered held-out probe (n=16).

## E7 result: the exploration cliff

- Probes: **learned = kv_norm = random = 0.000** (full = 0.875). The arena is maximally
  discriminative (any probe > 0 would be real learning) and still nothing.
- Training found correct episodes only by chance (~4 in 235 rollouts) and PPO never
  turned them into a policy. Finding a consistent sequence of ~350 evictions (with ~220
  options per step, x28 layers) is beyond the reach of online exploration from scratch.
- **Verdict:** neither sample count (50x) nor variance reduction (regret) touches the
  real bottleneck: the terminal reward is too sparse to guide exploration in this regime.
  It is the load-bearing negative that motivates the dense reward
  ([09](09-dense-kl-at-scale.md)).

Methodological note: the oracle bound is NOT measurable in this arena: capturing
attention requires `eager`, and under eager the eviction damage disappears in long-gen
(full - kv_norm = 0.000 there, vs 0.47 under sdpa; fifth sighting of a backend regime
shift). Data in `ab_results/oracle_longgen_results.jsonl`.

## What the report says

§4.11: budgets 160/192 overshot the useful point; in the calibrated configuration the
marked exploration problem appears; "the bottleneck seems to be in the reward signal, too
sparse to guide effective exploration in this regime".

## Raw data

- `kv-eviction-gym/ab_results/s_long_{learning,probe}_curve.csv` (E6, budget 160)
- `kv-eviction-gym/ab_results/s_long192_learning_curve.csv` (E6b)
- `kv-eviction-gym/ab_results/longgen_probe_curve.csv`, `longgen_labels.csv` (long-gen wide eval)
- `kv-eviction-gym/ab_results/s_e7_{learning,probe}_curve.csv` (E7)
- `kv-eviction-gym/ab_results/pool_screen.jsonl` (pool solvability screening)

## Status

Valid: it defines the long-gen arena (used by E8/E9) and establishes the terminal
reward's exploration limit.
