# 11 · Phase 3: the null result was the DATASET (passkey, HotpotQA, compression)

**Report section:** §4.15 "The dataset as the limiting factor".
**Report figures:** `fig8_dataset_spectrum.png`, `fig9_margin_vs_compression.png`.
**Scripts:** `kv-eviction-gym/scripts/eval_passkey.py`, `eval_prefill_compress.py`,
`rank_predictability.py`. **Own plots:** `experiments/phase3-dataset-causality/make_plots.py`.

## Question

Are the GSM8K nulls due to the online training method or to the dataset? Comparing our
setup with KVP's surfaces the structural difference (table in §4.15 of the report): in
RULER the filler dominates the prompt and future utility is distinguishable BY CONTENT
before generating; in GSM8K nothing is spare and what matters is GENERATED during
decoding.

## Setup: the passkey arena

A reduced version of RULER at our scale: filler text + a secret code buried at random
depth + forced generation (count to 40/80) so the needle comes under real eviction
pressure (~200-370 evictions). Budget 176 (later 128), n=96. Same model, same code, same
paired oracle arms ([06](06-future-attention-oracle.md)).

## Headline result

| arm | GSM8K (n=128) | passkey (n=96) |
|---|---|---|
| full (ceiling) | 0.49 | 0.85 |
| oracle (future attention) | 0.52 | 0.80 |
| attn_cur (present) | 0.46 | 0.40 |
| kv_norm | 0.45 | 0.38 |
| random | ~ | 0.49 |
| **oracle - kv_norm (paired)** | **+0.07** | **+0.43 (7.3 sigma; 45W/4L/47T)** |

Two structural facts:
- **kv_norm falls BELOW random on passkey** (0.375 < 0.490): keeping higher-norm tokens
  is actively harmful when what matters is a content-distinguishable needle. It is the
  signature of the RULER regime reported by Apple's paper, reproduced with our code.
- **Only future information wins**: attn_cur does not beat kv_norm in any regime.

## Confirmation on real data (HotpotQA-distractor) and the compression law

`eval_prefill_compress.py` (one-shot prefill compression, SnapKV style), n=96:

| dataset | compression | oracle - kv_norm |
|---|---|---|
| HotpotQA | 4-5x (budget 256) | **+0.094 (z=2.4, 12W/3L)** |
| HotpotQA | 8-10x (budget 128) | **+0.188 (z=4.4, 19W/1L)** |
| passkey | budget 176 | +0.427 |
| passkey | budget 128 | **+0.917 (z=32.3)**, kv_norm collapses to 0.04, oracle 0.96 |

The learnable margin **grows monotonically with compression aggressiveness** on real and
synthetic data: under hard compression kv_norm collapses to random's level while the
oracle holds near full. The three datasets form a spectrum ordered by how
content-distinguishable the information to preserve is: GSM8K (~0) < HotpotQA (small but
real, grows with compression) < passkey (large).

## E10: online PPO on passkey (causality control)

`configs/e10_passkey_rl.yaml` (+seed1, +warm): the SAME MaskablePPO + rich-features
pipeline that stayed flat on GSM8K, training on passkey (budget 300, forced generation,
terminal reward). Result (seed1, 28 probes): probe accuracy oscillates across the WHOLE
range [0.06, 1.00], mean 0.53, std 0.26 (kv_norm=0.688). **It reaches perfect eviction
policies (1.0), something that never happened on GSM8K (std ~0.05 pinned at kv_norm),
but it does not converge** and it also collapses; the warm-start collapses as soon as RL
starts. Causal reading: the signal exists and is exploitable; the bottleneck moved to the
stability of online training with a sparse reward. It motivates the E11 sweep with the
dense reward ([12](12-capstone-passkey.md)).

## Offline predictability control (recency caveat)

`rank_predictability.py` on GSM8K traces: an offline ranker achieves Spearman +0.89
predicting future utility (vs -0.01 for the kv_norm proxy), BUT a large part of it is
RECENCY (spearman(position, utility) = +0.57): it predicts "recent gets attended", which
streaming heuristics already exploit. The clean content test is the ranker on passkey
([12](12-capstone-passkey.md)).

## Signal-bearing dataset checklist (derived from the contrast)

1. effective compression >= 3-4x; 2. a large fraction of useless tokens; 3. utility
distinguishable by content (retrieval, not dense reasoning); 4. short generation (avoids
the truncation cliff); 5. automatically verifiable answer.

## Raw data

In `kv-eviction-gym/experiments/phase3-dataset-causality/data/`:
`gsm8k_oracle.jsonl`, `passkey_oracle.jsonl`, `passkey_b128_results.jsonl`,
`hotpot_results.jsonl`, `hotpot_b128_results.jsonl`, `e10_*_{learning,probe}.csv`,
`gsm8k_ranker_summary.json`. Also `ab_results/passkey_results.jsonl` and
`ab_results/rank_pred_summary.json`. Own figures in `plots/fig1..fig7`
(regenerate with `python make_plots.py`).

## Status

Valid; it is the central finding of the work (§4.15 and the report's conclusion). Honest
caveat: columns measured under eager attention; only compare paired gaps within the same
backend.
