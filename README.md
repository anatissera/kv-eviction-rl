# Sequential KV-cache eviction with PPO

Final project on **learned KV-cache eviction** for a frozen LLM
(Qwen2.5-1.5B-Instruct): we reformulate the decision of which token to discard as a
**sequence of discrete actions** (one per decode step, with action masking), which makes
the problem expressible as a Gymnasium environment and trainable with standard
**MaskablePPO**, in contrast with the one-shot ranking of Apple's KVP method (which is
static: a single decision over the prefilled cache, a 1-step environment).

**Main result:** on GSM8K, the whole ladder of variants (rich features, warm-start,
cross-token attention, causal dense reward, per-layer credit) ends at parity with the
`kv_norm` heuristic. Using a future-attention oracle we show this is a property of the
**dataset**, not of the method: the learnable margin is +0.07 on GSM8K versus +0.43 on
synthetic retrieval (passkey) and +0.09/+0.19 on HotpotQA, and it grows with compression
aggressiveness. In the signal-bearing arena, our online formulation with the causal dense
reward shows the first sustained shift above the heuristic, touching the perfect eviction
policy (though without stable convergence); a follow-up 48h stability sweep isolates the
number of optimization epochs per rollout as the driver of that shift.

**El informe completo:** [`informe/main.pdf`](informe/main.pdf).

## Repo map

```
report/                       The LaTeX report + compiled PDF + figure scripts
                              (figures.py regenerates the figures from the raw data).
docs/                         Per-experiment documentation:
  README.md                     index experiment <-> report section <-> data
  METHOD.md                     design of the environment and the policy
  runs/00..12                   one doc per experiment, with results tables
kv-eviction-gym/              OUR CONTRIBUTION: the sequential eviction environment
                              (Gymnasium + SB3 MaskablePPO), policies, training/eval
                              scripts, configs for every experiment, and the raw
                              results data (ab_results/, experiments/).
ml-learning-to-evict/         APPLE's repo (KVP), our starting point. Apple's code under
                              its own licence + ~10 files of ours for the Qwen2-1.5B
                              reproduction (see the PROVENANCE block in its README).
```

## Attribution

- `ml-learning-to-evict/` is a copy of [`apple/ml-learning-to-evict`](https://github.com/apple/ml-learning-to-evict)
  (Apple Sample Code License). Apple's files keep their copyright header; the ones we
  added carry the "Added for this project" header and are listed in the PROVENANCE block
  of [`ml-learning-to-evict/README.md`](ml-learning-to-evict/README.md).
- `kv-eviction-gym/` is **our own, independent code** (zero imports from or copies of
  Apple's code). The only connections: KVP's general idea as inspiration (comment in
  `src/kv_gym/policy.py`) and the deliberate reimplementation of the KVP recipe in
  `scripts/passkey_ranker.py`, used as a reference point in the report. The utilities in
  `src/kv_gym/vendor/` come from a previous repo of ours, not from Apple.

## The experiments (in the report's order)

| doc | experiment | report |
|---|---|---|
| [00](docs/runs/00-kvp-repro.md) | KVP reproduction (RULER, RLOO vs PPO) | §4.2 |
| [01](docs/runs/01-env-first-rewards.md) | Sequential environment + first rewards | §4.3 |
| [02](docs/runs/02-chat-template-fix.md) | The chat-template root cause | §4.4 |
| [03](docs/runs/03-capacity-screen.md) | E0: capacity screen | §4.5-4.6 |
| [04](docs/runs/04-scaled-runs.md) | Scaled runs: rich / warm-start / attention | §4.7 |
| [05](docs/runs/05-wide-eval-regime.md) | Wide eval + regime insight | §4.8 |
| [06](docs/runs/06-future-attention-oracle.md) | Future-attention oracle (+7pp) | §4.9 |
| [07](docs/runs/07-bc-match-oracle.md) | E5: future attention is not predictable from the present | §4.10 |
| [08](docs/runs/08-longgen-exploration.md) | E6/E7: long-gen and the exploration limit | §4.11 |
| [09](docs/runs/09-dense-kl-at-scale.md) | E8: causal dense reward at scale | §4.12 |
| [10](docs/runs/10-per-layer-credit.md) | Per-layer credit + E9 | §4.13-4.14 |
| [11](docs/runs/11-dataset-causality.md) | The null was the dataset (passkey/HotpotQA) | §4.15 |
| [12](docs/runs/12-capstone-passkey.md) | Capstone: E11 online in the signal-bearing arena | §4.16 |
| [13](docs/runs/13-e12-stability.md) | E12: 48h stability sweep | §4.16 |

Metric convention: paired contrast `correct_learned - correct_kv_norm` with
`full`/`kv_norm`/`random` anchors per run; gaps are only compared within a single run
(details in [`docs/README.md`](docs/README.md)).

## Quickstart

```bash
# Train our environment (GPU; local smoke on CPU with configs/quickstart.yaml)
cd kv-eviction-gym
uv sync
uv run python scripts/train.py --config configs/run_none.yaml --run-name my_run
```

Full instructions (VM setup, eval, tests): [`kv-eviction-gym/README.md`](kv-eviction-gym/README.md).
The KVP repro runs with `ml-learning-to-evict/run_experiment.sh` (see its README).

### Regenerating the report figures

```bash
python informe/figures.py                                        # figs 1-5, 8-10 del informe
python kv-eviction-gym/experiments/phase3-dataset-causality/make_plots.py   # figs propias de fase 3
```

All of them read the versioned raw data in `kv-eviction-gym/ab_results/` and
`kv-eviction-gym/experiments/*/data/`.

## Authors

Ana Paula Tissera and Alex Bodner. Universidad de San Andrés, 2026.
