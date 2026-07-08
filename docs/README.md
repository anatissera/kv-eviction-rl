# Experiment documentation

One document per experiment, in the order they appear in the report (section 4,
"Development and Results"; the LaTeX report is delivered separately and does not live in
this repo). Every doc follows the same format: question/hypothesis, setup, results (tables
with the measured numbers), what the report says, pointers to the raw data, and status
(valid / superseded / invalid and why). The report figures live in [`imgs/`](imgs/).

- [`METHOD.md`](METHOD.md): the design of the environment and the policy (what the project
  does, how, and what it explicitly does not do).

## Metric convention

The main metric of the whole project is the **paired contrast**
`correct_learned - correct_kv_norm` on the same probe and with the same anchors
(`full` = ceiling, `kv_norm` = the heuristic to beat, `random` = floor). Raw numbers
depend on the software stack and the data subset, so **only paired gaps within a single
run are comparable** (the report documents five backend/stack "regime shifts").

## Run index

| doc | experiment | report | figure | configs / scripts | raw data |
|---|---|---|---|---|---|
| [00](runs/00-kvp-repro.md) | KVP (Apple) repro with Qwen2-1.5B, RLOO vs PPO | §4.2 | `learning_curves.png` | `ml-learning-to-evict/run_experiment.sh` | gitignored (regenerable) |
| [01](runs/01-env-first-rewards.md) | Sequential environment + first rewards (collapse) | §4.3 | | `configs/run_none.yaml` | `results_overnight/`, `results_recency_long/`, `runs/none/` |
| [02](runs/02-chat-template-fix.md) | Root cause: chat template (+ budget floor) | §4.4 | | `src/kv_gym/vendor/prompts.py`, `tests/test_chat_template.py` | evidence table in the doc |
| [03](runs/03-capacity-screen.md) | E0: capacity screen (D1-D4) | §4.5-4.6 | fig1 | `experiments/phase2-capacity/screen_configs/` | `ab_results/e0_*` |
| [04](runs/04-scaled-runs.md) | Scaled runs: rich / warm / attn (2M) | §4.7 | fig2 | `configs/e1_rich, e3_warm, e4_attn` | `ab_results/s_{rich,warm,attn}_*` |
| [05](runs/05-wide-eval-regime.md) | Wide eval n=128 + regime insight | §4.8 | fig3, fig5 | `scripts/wide_eval.py` | `ab_results/wide_eval_*`, `wide2_*` |
| [06](runs/06-future-attention-oracle.md) | Future-attention oracle (+7pp) | §4.9 | fig4 | `scripts/oracle_eval.py` | `ab_results/oracle_results.jsonl` |
| [07](runs/07-bc-match-oracle.md) | E5: BC to the oracle does not converge (match 2-4%) | §4.10 | | `configs/e5_golden.yaml`, `scripts/trace_gen.py` | `ab_results/s_golden_*` |
| [08](runs/08-longgen-exploration.md) | E6/E7: long-gen and the exploration limit | §4.11 | | `configs/e6*, e7_repeat` | `ab_results/s_long*`, `s_e7_*`, `longgen_*` |
| [09](runs/09-dense-kl-at-scale.md) | E8: causal dense reward (KL) at scale | §4.12 (def. §3.2) | | `configs/e8_s4, e8_attn_s4` | `ab_results/s_e8*` |
| [10](runs/10-per-layer-credit.md) | Per-layer credit root cause + E9 per-layer | §4.13-4.14 | | `configs/e9_perlayer*` | `ab_results/s_e9*` |
| [11](runs/11-dataset-causality.md) | Phase 3: the null was the dataset (passkey/HotpotQA) | §4.15 | fig8, fig9 | `scripts/eval_passkey.py`, `eval_prefill_compress.py` | `experiments/phase3-dataset-causality/data/` |
| [12](runs/12-capstone-passkey.md) | Capstone: E11 online + offline KVP reference | §4.16 | fig10 | `configs/e11_kl*`, `scripts/passkey_ranker.py` | `phase3.../data/e11_*`, `passkey_ranker_*` |
| [13](runs/13-e12-stability.md) | E12: 48h stability sweep (seeds, epochs, PPO knobs) | §4.16 | fig11-14 | `configs/e12_*`, `experiments/phase4-stability/` | `experiments/phase4-stability/data/` |

Paths are relative to `kv-eviction-gym/` unless noted. The report's `figN` figures live in
[`imgs/`](imgs/) and are regenerated with `scripts/figures.py` and `scripts/figures_e12.py`
(they read the raw data from `kv-eviction-gym/`); the phase-3-specific figures in
`kv-eviction-gym/experiments/phase3-dataset-causality/plots/` are regenerated with
`make_plots.py` in that directory.

## The story in four lines

1. We reformulate KV-cache eviction as sequential decisions trainable with standard
   MaskablePPO (00-02) and scale it on GSM8K.
2. Every variant (representation, warm-start, architecture, dense reward, per-layer
   credit) ends at PARITY with `kv_norm` (03-10); the effective online ceiling is the
   heuristic, and the margin that exists (+7pp) is unobservable future information (06-07).
3. The cause was not the method but the DATASET: in retrieval (passkey/HotpotQA) the
   oracle margin is +0.43/+0.19 and grows with compression; on GSM8K it is +0.07 (11).
4. In the signal-bearing arena, the offline reference (KVP) wins +0.46, and our online
   method with the causal dense reward shows the first sustained shift above `kv_norm`,
   touching the perfect policy, though without stable convergence yet (12); a 48h
   stability sweep isolates n_epochs as the driver and maps what does not help (13).
