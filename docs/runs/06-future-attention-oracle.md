# 06 · Future-attention oracle: the learnable margin exists (+7pp) and it is future information

**Report section:** §4.9 "Policy with future information".
**Figure:** `fig4_oracle.png`.
**Script:** `kv-eviction-gym/scripts/oracle_eval.py`.

## Question

If the low-difficulty regime has little headroom, how much can one improve over kv_norm
IN PRINCIPLE? We build an ideal eviction with future information: over the full
generation trace, at every step the token with the lowest maximum future attention is
discarded (a decision only possible with access to the future; it is a retrospective
upper bound, not a policy attainable online).

## Setup

128 examples from the wide slice, all arms paired per example under ONE model instance
with `eager` attention (needed to read the attention weights). Arms: `full`, `kv_norm`,
`attn_cur` (evict the one with the lowest PRESENT accumulated attention, H2O family),
`oracle_fut` (lowest maximum FUTURE attention), `random`.

## Result

| arm | acc | paired vs kv_norm |
|---|---|---|
| **oracle_fut** | **0.516** | **+0.070 ± 0.025 (2.8 sigma; 10W-1L, McNemar p ≈ 0.01)** |
| full (no eviction) | 0.492 | +0.047 |
| attn_cur (present) | 0.461 | +0.016 (noise) |
| kv_norm | 0.445 | (baseline) |

Three findings:
1. **The learnable margin exists (~+7pp) and it is specifically future information.**
   Present attention does not beat kv_norm. This coherently explains the whole phase: no
   online policy (learned or heuristic) sees the future, so they all tie at kv_norm's
   level, which turns out to be the effective online ceiling on GSM8K.
2. **The oracle beats even the full cache** (0.516 > 0.492): a good eviction does not
   just preserve information, it can improve reasoning by discarding the right tokens.
3. **Another backend regime shift:** full = 0.49 under `eager` vs 0.74 under `sdpa` on
   the same examples (eager computes QK^T in bf16; sdpa accumulates in fp32). Paired gaps
   within the same backend remain valid; never compare raw numbers across backends.

H2O note: this experiment is also the measurement of the H2O heuristic (accumulated
attention) cited in §2.3 of the report: it does not significantly beat kv_norm on GSM8K.

## What the report says

§4.9: the margin exists but depends on future information; the supervised ceiling sits
~7pp above kv_norm. It foreshadows that this turned out to be dataset-specific: on
passkey ([11](11-dataset-causality.md)) kv_norm stops being a ceiling and falls below
chance.

## Raw data

- `kv-eviction-gym/ab_results/oracle_results.jsonl` (per example, the 5 paired arms).
- `kv-eviction-gym/ab_results/oracle_longgen_results.jsonl` (long-gen attempt: under
  eager the eviction damage disappears in that arena, the bound is unmeasurable there;
  see [08](08-longgen-exploration.md)).
- Report figure: `fig4_oracle.png`.

## Status

Valid (with the eager-backend caveat). It is the upper bound against which the whole
phase-3 dataset analysis is defined.
