# 12 · Capstone: our online method on a signal-bearing dataset (E11 + KVP reference)

**Report section:** §4.16 "Our method on a signal-bearing dataset".
**Figure:** `fig10_capstone.png`.
**Configs:** `kv-eviction-gym/configs/e11_klA.yaml`, `e11_klB.yaml`, `e11_klC.yaml`,
`e11_klC_seed1.yaml`. **Offline reference:** `scripts/passkey_ranker.py`.

## Question

Having located a dataset where learned eviction has margin ([11](11-dataset-causality.md)),
can OUR method (sequential online PPO) exploit that signal, or is the literature's
offline training required?

## Reference: Apple's offline recipe (KVP) on passkey [POSITIVE, not our contribution]

`passkey_ranker.py` reimplements the KVP recipe: per-layer future-attention rankers
trained OFFLINE on 120 traces (features = K/V + scale/position columns, label = future
attention), evaluated AS an eviction policy on held-out data. Budget 176 (the +0.43
regime).

| arm | seed 0 (n=40) | seed 1 (n=60) |
|---|---|---|
| full (ceiling) | 0.725 | ~ |
| oracle | 0.675 | 0.883 |
| **learned ranker** | **0.775** | **0.883** |
| random | 0.450 | ~ |
| kv_norm | 0.325 | ~ |

**Combined learned - kv_norm: +0.46 (49 wins / 3 losses over 100, p < 1e-8), replicated
across 2 independent seeds.** It confirms the arena has signal exploitable by SOME
method. The report presents this as a reference (it is Apple's method, not our
contribution).

## Our method: E11, online PPO with the causal dense reward

The first online attempt with a terminal reward (E10, see [11](11-dataset-causality.md))
touched perfect policies but oscillated without converging: a STABILITY problem. E11
trains with the causal dense reward (KL to the shadow cache, the report's final proposal,
§3.2) and sweeps PPO hyperparameters. Common setup: passkey, budget 300, forced
generation up to 300 tokens, `kl_weight=0.05`, `kl_clip=5.0`, 3M steps. Only the
hyperparameters change:

| config | lr | ent_coef | n_epochs | probe mean | 1st half | 2nd half | max |
|---|---|---|---|---|---|---|---|
| e11_klA | 1e-4 | 0.005 | 4 | 0.375 | 0.435 | 0.318 | 0.81 |
| e11_klB | 5e-5 | 0.003 | 4 | 0.217 | 0.188 | 0.246 | 0.67 |
| **e11_klC** | **1e-4** | **0.01** | **10** | **0.509** | **0.359** | **0.659** | **1.00** |

**The result (klC):** in the first half the policy oscillates between 0 and kv_norm
(mean ~0.37); in the second half the oscillation's centre shifts upward (mean ~0.66,
above kv_norm ~0.56) and the best probe reaches a perfect policy (1.0). For the first
time in the whole project the policy spends most of its time ABOVE the heuristic,
something that happened neither on GSM8K nor on passkey with the terminal reward alone.

## Limitations (explicit in the report)

1. **Hyperparameter sensitivity:** of the three configs, only klC shows the shift (more
   optimization epochs per rollout seem to stabilize).
2. **A single seed:** a replica was launched (`e11_klC_seed1`, partial data in
   `data/e11_klC_seed1_*.csv`) but the published evidence rests on seed 0.
3. It is not stable convergence: it keeps oscillating, with drops to kv_norm or below.
   The improvement is a shift of the average behavior.

## What the report says

§4.16 and the conclusion: the evidence suggests that the proposed online reformulation,
which sat systematically at parity on GSM8K, can beat the heuristic when the dataset
offers learnable signal, without the literature's offline supervision, although
confirming it robustly (more seeds, longer runs, PPO stabilization techniques) remains
future work.

## Raw data

- `kv-eviction-gym/experiments/phase3-dataset-causality/data/e11_kl{A,B,C}_{learning,probe}.csv`,
  `e11_klC_seed1_{learning,probe}.csv`
- Ranker de referencia: `data/passkey_ranker_seed{0,1}_summary.json`,
  `ab_results/passkey_ranker_summary.json`
- Figuras: `informe/Figures/1. Imgs/fig10_capstone.png`,
  `experiments/phase3-dataset-causality/plots/fig5_learned_ranker_wins.png`

## Status

Valid with the caveats above (1 seed, hyperparameter-sensitive, no stable convergence).
It is the report's constructive closing. The follow-up stability sweep is
[13](13-e12-stability.md).
