# 05 · Wide eval (n=128) and the regime insight

**Report section:** §4.8 "Wide eval and regime analysis".
**Figures:** `fig3_wide_eval.png`, `fig5_regime.png`.
**Script:** `kv-eviction-gym/scripts/wide_eval.py`.

## Question

The three scaled arms lived on different stacks: what is the clean comparison? The three
final checkpoints were evaluated on 128 new, disjoint examples (slice [1032:1160] of the
seeded shuffle), on ONE VM with the same anchors.

## Result: the decisive table

| arm | learned | paired vs kv_norm | trunc |
|---|---|---|---|
| s_rich_final | 0.695 | **-0.016** (~2 examples) | 0.03 |
| s_warm_final | 0.688 | **-0.023** (~3 examples) | 0.21 |
| s_attn_final | 0.648 | **-0.063** | 0.01 |
| s_attn_best | 0.633 | **-0.078** | 0.02 |
| *kv_norm* | *0.711* | (baseline) | |
| *random* | *0.703* | *-0.008 vs kv_norm* | |
| *full* | *0.742* | *ceiling* | |

- The per-token arms (rich, warm) sit at **parity within noise** (1 example = 0.0078).
- Attention is genuinely worse (also below random): D3's negative replicates outside the
  probe.

## The regime insight (the phase's most important finding)

**`full - random = 0.039`**: with budget=256 and these prompt lengths, evicting at random
costs only ~4pp versus not evicting, and `kv_norm` captures ~20% of that tiny margin
(+0.008 over random). There is almost nothing for a learned policy to gain in this
regime: the terminal reward is nearly flat as a function of the policy (PPO with no
gradient), and even a perfect oracle would gain ~3pp.

This is bounded by the **training environment's architecture**: the batched environment
pre-allocates `budget+1` shared slots and discards any example whose prompt is longer
than the budget, so the budget can never go below the prompt length (~232). The
environment cannot express aggressive compression. The literature's wins (KVP,
ForesightKV) live at 50% budgets and long contexts where eviction really hurts.

The contrast with the long-generation regime (same GSM8K filtered to long reasoning
chains, see [08](08-longgen-exploration.md)): there `full - random = 0.45` and `kv_norm`
does not even beat chance. Report figure 5 (`fig5_regime.png`) shows both regimes.

Also, the fifth sighting of a stack-driven "regime shift": on wide2 (another stack) the
same 128 examples give `full - random = 0.156`. This is why the report insists: only
paired contrasts within a single run are comparable.

## What the report says

§4.8: "This experiment ended up being the most important discovery of the whole phase,
and not because of the policies but because of the regime". It defines the
"low-difficulty regime" and motivates moving the environment to long generations before
trying more variants.

## Raw data

- `kv-eviction-gym/ab_results/wide_eval_probe_curve.csv`, `wide_eval_labels.csv`
- `kv-eviction-gym/ab_results/wide2_probe_curve.csv`, `wide2_labels.csv` (replication on another stack)
- Report figures: `fig3_wide_eval.png`, `fig5_regime.png`.

## Status

Valid; it is the definitive number for the unfiltered-GSM8K phase and the motivation for
the arena change. The wide eval of the long-gen regime is in [08](08-longgen-exploration.md).
