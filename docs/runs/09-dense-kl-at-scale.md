# 09 · E8: the causal dense reward (KL to the shadow cache) at scale

**Report section:** §4.12 "The causal dense reward at scale" (the reward is defined in
§3.2 "Reward design").
**Configs:** `kv-eviction-gym/configs/e8_s4.yaml` (MLP), `e8_attn_s4.yaml` (attention),
`e8_smoke.yaml`. Implementation: `kl_shaping` in `src/kv_gym/batched_env.py`.

## The reward (the report's final proposal)

At every generation step, the next-token distribution with the evicted cache is compared
against that of a "shadow" cache that never evicts:

```
r_step = -w * clip( KL(p_full || p_evict), 0, c )      with w=0.03-0.05, c=5.0
```

Dense, per-step, causal, label-free and compatible with the fast `sdpa` backend (it only
needs output probabilities, not attention weights). Cost: ~2-2.5x the decode (one extra
shadow forward per episode per step).

## Prior history: the first S4 A/B was INVALID (methodological lesson)

The original A/B (control vs S4, 5M steps) gave an apparent +0.183 retention gain for S4.
It was an artifact: the two arms ran on different VMs with different code and DIFFERENT
32-example probes. Proof: the policy-independent baselines differed across arms
(`kv_norm` 0.625 vs 0.750), impossible in a comparable eval. Under the honest paired
metric both arms gave ≈ -0.037: S4 ≈ control and neither beat kv_norm. This is where the
report's golden rule (§4.1) comes from: only paired contrasts within a single run, and if
the anchors do not match, the comparison is void. (The original S4 experiment design and
its hyperparameters remain in the history: `experiments/s4-entropy-shaping/`.)

## E8: the fair test (signal-bearing arena + capable policy)

The first genuinely favorable scenario for the dense reward: rich features
([04](04-scaled-runs.md)) + the long-gen arena with 45pp of headroom
([08](08-longgen-exploration.md)), 50 repetitions per example + regret baseline. Two
arms: MLP and the attention policy.

## Result: parity, but the most informative null

| arm | probes | paired learned - kv_norm |
|---|---|---|
| s_e8 (MLP + dense KL) | 9 | **+0.021 ± 0.014** (n.s.) |
| s_e8attn (attention + dense KL) | 12 | **+0.016 ± 0.018** (n.s.) |

**Mechanistic finding:** `kl_step_mean` DROPS steadily (0.109 → 0.073 in s_e8): the dense
reward does get optimized. But probe correctness never follows it. **Minimizing the KL
divergence of the next-token distribution is a proxy that decouples from correctness**
under the truncation cliff: the policy keeps distributions close even when the evictions
have already broken the reasoning chain (`evict_generated_frac` 0.5-0.99 in the probes,
trunc ~0.45).

## What the report says

§4.12: "the result was parity again... The per-step KL divergence does decrease... but
that improvement never translates into higher correctness. The reward stops being aligned
with the final objective." The underlying structural explanation (per-layer credit) is in
[10](10-per-layer-credit.md).

## Raw data

- `kv-eviction-gym/ab_results/s_e8_{learning,probe}_curve.csv`, `s_e8attn_{learning,probe}_curve.csv`
- Original invalid A/B: `ab_results/control_{learning,probe}_curve.csv`,
  `treat_{learning,probe}_curve.csv`, `ab_retention.png`, `compare_ab.py`

## Status

Valid. The causal dense reward reappears as the piece that DOES help in the passkey arena
([12](12-capstone-passkey.md)): same shaping, signal-bearing dataset.
