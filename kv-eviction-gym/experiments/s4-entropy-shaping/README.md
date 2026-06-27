# S4 — Information-theoretic per-step reward shaping for KV-eviction

Living document for the S4 experiment. Status, hyperparameters, and results are
updated here as runs progress.

## Motivation

RL training uses `gamma=1, gae_lambda=1` (Monte Carlo): the terminal reward
(correctness ± length/truncation penalties) propagates **identically to every
step and every one of the 28 layers**. A catastrophic eviction at step 3 and a
harmless one at step 200 receive the same credit → weak temporal credit
assignment. The existing attention-shaping (`shaping_mode=per_step`) tried to
localize credit but (a) needs `eager` attention (slow), (b) uses a proxy
(attention overlap), (c) treated generated tokens as "free" → recency collapse.

**S4** replaces the proxy with the **causal** signal: the real effect of each
eviction on the LLM's next-token distribution. At every decode step we compare
the output with the evicted cache vs a never-evicted (full) shadow cache:

```
r_step[b] = − kl_weight · clip( damage[b], 0, kl_clip )
  exact : damage = KL( p_full ‖ p_evict )   # causal effect of this step's evictions
  proxy : damage = H( p_evict )             # reference-free output entropy (cheap, biased)
```

Dense, per-step, label-free, and works with `sdpa` (no attention capture). It is
**added on top of the unchanged baseline reward**, so the only difference vs the
`none` baseline is this shaping term.

## Design notes

- Implemented in `src/kv_gym/batched_env.py` (the training path,
  `BatchedSharedKVVecEnv`). The single-env `env.py` path is unused in training.
- **Shadow cache** (`exact`): at the moment eviction begins (`cache_size=budget+1`)
  we clone the episode's KV as `past_kv_full` (per `EpisodeState`), grown but
  never evicted. One extra batch=1 forward per episode per step (~2–2.5× decode
  compute). `proxy` needs no shadow cache and no extra forward (≈ baseline speed).
- **Layer-shared credit**: the KL is one scalar per episode/step (the LLM output
  is joint over layers), broadcast to all 28 layer-slots. It localizes credit in
  TIME but not per-layer — a known limitation; per-layer attribution is future work.
- Caches diverge in length across episodes after independent resets, so the full
  forward is done **per episode** (N=2 batch=1 forwards), not batched.

## Arms

| Arm | Config | Reward | Cost | Run name |
|---|---|---|---|---|
| baseline `none` | `configs/corr_reward.yaml` | correctness + length/trunc | 1× | `none_v12` (reused, already running to 5M) |
| **S4-exact** | `experiments/s4-entropy-shaping/configs/run_s4_kl.yaml` | + per-step KL-to-full | ~2–2.5× | `s4_kl_v1` |
| **S4-proxy** | `experiments/s4-entropy-shaping/configs/run_s4_selfent.yaml` | + per-step self-entropy | ~1× | `s4_selfent_v1` |

## Hyperparameters

Identical to `none_v12` for comparability — only the `kl_*` block differs:
budget curriculum 400→256 (frac 0.75), n_sinks=4, n_recent=32, hidden=64,
n_epochs=4, batch=512, n_steps=650, clip=0.2, ent_coef=0.01, gamma=1, lambda=1,
length_penalty=0.5, trunc_penalty=1.0, n_parallel=2, sdpa, **total_timesteps=5M**.
New: `kl_shaping=true`, `kl_mode={exact|proxy}`, `kl_weight=0.05` (calibrate),
`kl_clip=5.0`.

`kl_weight` is calibrated with a short dry-run that logs mean/summed per-step
damage; target summed per-episode shaping ≈ 0.3–0.5 (sub-dominant to correctness ±1).

## Comparability

All arms see the **identical data stream**: `n_examples=1000`, `seed=0`, same 32
left-out probe examples, same example ordering, same 5M-timestep horizon. Compared
at matched timesteps. The baseline arm reuses `none_v12` (no rerun).

## How to reproduce

```bash
# unit tests (fast, no GPU)
pytest tests/test_s4_kl.py -q

# smoke (a couple of minutes on the L4)
python scripts/train.py --config experiments/s4-entropy-shaping/configs/smoke_s4.yaml --run-name s4_smoke

# full arms (run with the glibc-mmap fix already in train.py; use a watchdog)
python scripts/train.py --config experiments/s4-entropy-shaping/configs/run_s4_kl.yaml      --run-name s4_kl_v1
python scripts/train.py --config experiments/s4-entropy-shaping/configs/run_s4_selfent.yaml --run-name s4_selfent_v1

# compare retention curves
python experiments/s4-entropy-shaping/compare_probe.py \
  --runs runs/none_v12:baseline runs/s4_kl_v1:S4-exact runs/s4_selfent_v1:S4-proxy
```

## Metrics & what we want to see

- **H1 (primary):** `retention` (probe_curve.csv) of S4-exact ≥ baseline at
  matched timesteps, ideally rising earlier/higher.
- **H2 (collapse):** S4-exact keeps `evict_generated_frac` / `evict_sink_frac`
  healthy (no pure-recency collapse); `kl_step_mean` (learning_curve.csv)
  decreases over training (policy learns low-damage evictions).
- **H3 (mechanism):** learned-policy mean per-step KL < random/kv_norm at equal budget.
- **proxy vs causal:** S4-proxy ≈ baseline but S4-exact > baseline → the causal
  signal matters. S4-proxy ≈ S4-exact → the cheap proxy suffices (useful finding).
- **Null result is informative:** S4-exact ≈ baseline → bottleneck is policy
  capacity (`PerTokenMLP`, no cross-token), not credit assignment → next step is
  architecture, not reward.

## Results

_(updated as runs progress)_

| Run | timesteps | final retention | best retention | final kl_step_mean | notes |
|---|---|---|---|---|---|
| none_v12 (baseline) | — | — | — | — | reused |
| s4_kl_v1 | — | — | — | — | |
| s4_selfent_v1 | — | — | — | — | |
