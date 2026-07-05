# 03 · E0: capacity screen (which variant CAN beat kv_norm?)

**Report sections:** §4.5 "Why the policy did not beat kv_norm" and §4.6 "A first
capacity screen". **Figure:** `fig1_screen.png`.
**Configs:** `kv-eviction-gym/experiments/phase2-capacity/screen_configs/e0_*.yaml` (5 variants).
**Driver:** `experiments/phase2-capacity/run_screen.sh`, verdict via `screen_verdict.py`.

## Prior hypotheses (the 4 deficiencies, code-level analysis)

With the environment working and the chat template fixed, the scaled policy sat at parity
with `kv_norm`. Reading the code surfaced four separable deficiencies:

| # | deficiency | code evidence | consequence |
|---|---|---|---|
| D1 | norm blindness: the policy applies LayerNorm to K and V as its first operation | `policy.py` | cannot represent `kv_norm`'s exact signal (the norm) |
| D2 | position blindness: no explicit feature, only RoPE inside K | `features.py` | a per-token post-LayerNorm MLP does not decode position/recency |
| D3 | no cross-token reasoning: independent decision per token | `policy.py` | cannot represent redundancy across positions |
| D4 | cold-start: RL explores from random init | flat curves | may never find kv_norm's basin |

D1 alone is enough to explain "the policy performs like random": the most predictive
feature of the task is destroyed at the input.

## Screen setup

Before spending GPU-days scaling, each variant trains and evaluates on the SAME
16 examples (`probe_on_train`): a pure **capacity** (overfit) test. ~1h per variant.
On those 16 examples: `full`=0.81, `kv_norm`=`random`=0.06 (eviction is catastrophic for
the heuristics; there is a lot of headroom).

## Result

| variant | what it isolates | best learned | vs kv_norm | reading |
|---|---|---|---|---|
| `e0_baseline` (K‖V, MLP) | nothing | 0.06 | +0.00 | cannot beat kv_norm (norm-blind) |
| `e0_rich` (rich features) | representation D1+D2 | 0.25 | **+0.19** | beats it 4x at the peak, but spiky (0 ↔ 0.25) |
| `e0_attn` (rich + attention) | architecture D3 | 0.25 | **+0.19** | same, spiky, rising at cutoff |
| `e0_rich_s4` (rich + dense KL) | reward on a capable policy | 0.125 | +0.06 | < rich alone: the dense reward does NOT help here |
| `e0_rich_warm` (rich + BC) | exploration D4 | failed | n/a | BC bug (fixed later, see below) |

**Verdict:** representation (D1/D2) is directionally the bottleneck; rich and attention
can beat kv_norm in overfit, the baseline cannot. Adding the dense reward to an already
expressive policy did not improve (+0.06 < +0.19). The signal is noisy and spiky.

## The warm-start fix (two bugs, important for [04](04-scaled-runs.md))

- **Bug A (representability):** the rich features provided `kz` and `vz` standardized
  separately, so the policy could not reconstruct `argmin(‖K‖+‖V‖)`. Fix: the **`kvz`**
  feature = standardized(‖K‖+‖V‖ averaged over heads) = kv_norm's EXACT signal. With it
  the policy can represent the heuristic (`logit = -kvz`) and re-weight K vs V to try to
  beat it. Total: 9 extra columns (2H+5).
- **Bug B (optimization):** BC used cross-entropy to the argmin over ~220 slots with
  batches of 56: extremely high variance. Fix: dense MSE of the actor's logits toward
  `-kvz*3` on all valid slots (stable; argmax = kv_norm's choice).

Test that proves it: `tests/test_rich_features.py` (`test_kz_recovers_kv_norm_ordering`,
`test_train_eval_obs_identical`).

## What the report says

§4.6: the screen said what was worth scaling and what was not. Two observations: the
signal was spiky (jumps between 0 and 0.25), and the dense reward did not help even an
already capable policy (rich features + dense got +0.06, below rich features alone),
which at the time weakened the proposed contribution.

## Raw data

- `kv-eviction-gym/ab_results/e0_{baseline,rich,rich_s4,rich_warm,attn}_{learning,probe}_curve.csv`
- `kv-eviction-gym/ab_results/screen_verdict.txt`
- Figura del informe: `informe/Figures/1. Imgs/fig1_screen.png` (regenerable con `informe/figures.py`).

## Status

Valid as a capacity test (overfit on 16 examples). Generalization is measured in
[04](04-scaled-runs.md): the screen's +0.19 peak did NOT transfer to scale.
