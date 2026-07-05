# 04 · Scaled runs: rich features, warm-start and cross-token attention (2M steps)

**Report section:** §4.7 "Scaling the system: features, warm-start and attention".
**Figure:** `fig2_scaled_traj.png`.
**Configs:** `kv-eviction-gym/configs/e1_rich.yaml`, `e3_warm.yaml`, `e4_attn.yaml`.
**Driver:** `experiments/phase2-capacity/run_scaled.sh`, comparison via `compare.py`.

## Question

Of the hypotheses the screen ([03](03-capacity-screen.md)) left alive, which one beats
`kv_norm` at scale? Three arms, each isolating one hypothesis:

- **s_rich** (rich features + `kvz`): representation (D1+D2).
- **s_warm** (BC toward kv_norm, then RL): cold-start / exploration (D4).
- **s_attn** (rich + cross-token attention policy `PerTokenAttention`): architecture (D3).

## Setup

| component | value |
|---|---|
| scale | 2M timesteps, 1000 train examples (seed 0), held-out probe of 32 |
| environment | GSM8K, fixed budget 256, same anchors per arm |
| metric | paired contrast `correct_learned - correct_kv_norm` within each arm |

## Results

| arm | paired mean | last-5 | max | reading |
|---|---|---|---|---|
| s_rich | **-0.044** | -0.050 | +0.094 | PARITY: rich+kvz took the policy from ~random UP TO kv_norm, not beyond |
| s_warm | **-0.022** | -0.019 | +0.031 | the clone worked (first probe with learned >= kv_norm) but 2M of PPO does not push it beyond; D4 closed: exploration was NOT the wall |
| s_attn | **-0.111** | -0.125 | -0.062 | never at kv_norm's level; the screen's +0.19 did not transfer; D3 closed negative |

Notes:
- With `kvz` as a feature, the easiest thing to learn is "being kv_norm", and it stalls there.
- s_warm starts AT kv_norm (the BC took) and RL even degrades it mid-run before returning
  to parity.
- Comparability caveat: the three arms ran on different software stacks
  (torch/transformers change the generations); only within-arm paired gaps are valid.
  The decisive cross-arm number is the wide eval ([05](05-wide-eval-regime.md)).
- Infra bug found and fixed along the way: checkpoints were never saved
  (`save_freq` counts wall-steps and the batched environment does 56 timesteps per
  wall-step), so every spot preemption restarted from zero. Fix: `checkpoint_freq=900`.
  Also an SB3 resume bug that added the timestep budget on every resume (fix `8decafd`).

## What the report says

§4.7: informative features closed the representation gap (real progress: from ~random to
kv_norm) but do not push beyond it; the warm-start isolates and discards the cold-start
hypothesis; the attention variant behaves worst ("complexity PPO fails to exploit").
Figure 2 shows the three trajectories oscillating around or below 0.

## Raw data

- `kv-eviction-gym/ab_results/s_{rich,warm,attn}_{learning,probe}_curve.csv`
- `kv-eviction-gym/ab_results/scaled_comparison.txt`, `scaled_retention.png`
- Final checkpoints (untracked, in local `ab_results/checkpoints/`): s_rich, s_warm, s_golden.
- Report figure: `../imgs/fig2_scaled_traj.png`.

## Status

Valid and central to the report: closes D1/D2 (partial: they reach parity), D4 (negative)
and D3 (negative). The explanation of WHY everything ends at parity arrives in two parts:
the regime ([05](05-wide-eval-regime.md)) and per-layer credit ([10](10-per-layer-credit.md)).
