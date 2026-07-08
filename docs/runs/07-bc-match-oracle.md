# 07 · E5 Golden-BC: future attention is unpredictable from the present

**Report section:** §4.10 "Future attention is unpredictable from the present".
**Config:** `kv-eviction-gym/configs/e5_golden.yaml`. **Traces:** `scripts/trace_gen.py`.

## Question

The natural step (and the literature's recipe, ForesightKV) after measuring the oracle
([06](06-future-attention-oracle.md)): distill it. Can a policy that only observes the
present (K, V and derived features) learn by behavior cloning to imitate the
future-attention oracle's decision?

## Setup

- Full-cache traces of ~400 train examples (eager attention to capture the scores), ~35 min.
- Dense BC of the actor's logits toward the future-attention score, 3000 steps, with the
  same dense-MSE recipe that successfully cloned kv_norm ([03](03-capacity-screen.md)).
- Metric: `match_oracle` = fraction of steps where the policy's action matches the
  oracle's. Then a short PPO on top (run `s_golden`).

## Result

- **BC never converged: `match_oracle` stayed between 0.018 and 0.036** over the 3000
  steps (loss 9.0 → 8.4, flat). Barely ~8x over chance (1/220 slots). In contrast, the
  SAME recipe cloning kv_norm reaches match ≈ 1.0: the procedure is stable when the
  signal is accessible from the present.
- Conclusion: **the oracle's +7pp margin is made of information that does NOT exist in
  the present observation**. The problem is partial observability with respect to that
  criterion. (The literature dodges this by adding attention-history features that our
  `sdpa` environment does not expose, and even then it needs pairwise ranking schemes and
  per-head scorers.)
- The `s_golden` run (failed BC + RL) showed +0.074 on ITS probe, but it was a slice
  artifact: its probe [400:432] has kv_norm=0.06 (catastrophic eviction), a hostile
  screen-like regime. On the shared wide slice it gave **-0.070** with truncation 0.29:
  worse than kv_norm. The second time GSM8K's slice heterogeneity almost fooled us
  (a methodological trap also documented in [03](03-capacity-screen.md) and §4.1 of the
  report).

## What the report says

§4.10: behavior cloning does not converge on `match_oracle` (0.018-0.036 vs ~1.0 when
cloning kv_norm), so the +7pp margin is information not contained in the present state.
This turns the problem into one of partial observability and explains why the literature
resorts to offline supervision with attention-history features.

## Raw data

- `kv-eviction-gym/ab_results/s_golden_{learning,probe}_curve.csv`
- wide2 (the eval that unmasks the artifact): `ab_results/wide2_probe_curve.csv`

## Status

Valid and load-bearing: it is the cleanest claim of all of phase 2 (future information is
not predictable from present K/V on GSM8K). The phase-3 complement is
`rank_predictability` ([11](11-dataset-causality.md)): an offline ranker CAN predict
future utility on GSM8K but mostly via recency, not content.
