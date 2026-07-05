# KV-Eviction Gym

Reinforcement learning agent that learns to evict tokens from a transformer's KV cache
**during generation** while preserving answer correctness on GSM8K.

The policy (MaskablePPO + per-token MLP) makes one eviction decision per layer per
decode step, keeping the cache at a fixed budget throughout generation. See
[../docs/METHOD.md](../docs/METHOD.md) for the full design and
[../docs/README.md](../docs/README.md) for the per-experiment documentation
(one doc per report experiment, with result tables and raw-data pointers).

---

## Setup

**Requirements:** Python ≥ 3.10, CUDA GPU (L4/T4/A100 recommended), [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd kv-eviction-gym
uv sync          # creates .venv and installs all dependencies incl. tensorboard
```

Dependencies are declared in `pyproject.toml` and include: `torch`, `transformers`,
`stable-baselines3`, `sb3-contrib`, `tensorboard`, `datasets`, `gymnasium`.

> **Note:** always use `uv sync` (not `pip install`). Plain pip will miss
> `tensorboard` and other indirect deps declared in `pyproject.toml`.

---

## Training on a GPU VM (one command)

```bash
HF_TOKEN=<your_token> bash setup_and_train.sh --run-name my_run
```

`setup_and_train.sh` handles the full setup automatically:
1. Installs `uv` if missing
2. Runs `uv sync` to create/update the venv
3. Checks CUDA availability
4. Launches `scripts/train.py` with `configs/run_none.yaml`

Optional flags:
```bash
# Resume from a checkpoint
HF_TOKEN=... bash setup_and_train.sh --run-name resumed --resume-from runs/my_run/best_model.zip

# Use a different config
HF_TOKEN=... bash setup_and_train.sh --config configs/quickstart.yaml --run-name test
```

---

## Training only (manual)

```bash
uv run python scripts/train.py --config configs/run_none.yaml --run-name my_run
```

Key config options (`configs/run_none.yaml`):

| Field | Default | Notes |
|-------|---------|-------|
| `model_name` | `qwen-1.5b` | Model key; defaults loaded from `configs/model_defaults.json` |
| `n_examples` | `1000` | GSM8K training examples (cycled) |
| `probe_n` | `32` | Left-out training examples for periodic eval (0 to disable) |
| `budget_min` / `budget_max` | `256` / `600` | Cache capacity range (with optional curriculum `budget_min_start/end`) |
| `max_new_tokens` | `800` | Max decode steps per episode |
| `n_recent` / `n_sinks` | `32` / `4` | Recency window and attention sinks, protected from eviction |
| `n_parallel` | `2` | Episodes decoded together in one forward pass (n_envs = 2 x 28 layers = 56) |
| `length_penalty_weight` | `0.5` | Dense signal: `-weight x steps/max_new_tokens` |
| `truncation_penalty` | `1.0` | Hard penalty when episode hits token cap without EOS |
| `total_timesteps` | `5000000` | SB3 timestep budget (counts x56 envs per wall-step) |
| `shaping_mode` | `none` | `none` = pure correctness (fastest); see also `kl_shaping` below |
| `rich_features` | (E1+) | Scale-aware feature columns incl. `kvz` (the exact kv_norm signal) |
| `kl_shaping` / `kl_weight` / `kl_clip` | (E8+) | Dense causal reward: per-step KL to a never-evicted shadow cache |
| `per_layer_reward` | (E9+) | Per-layer dense reward (marginal hidden-state divergence) |
| `dataset` | `gsm8k` | `passkey` selects the synthetic retrieval arena (phase 3) |

The experiment configs used in the report live in `configs/` (`e1_rich`, `e3_warm`,
`e4_attn`, `e5_golden`, `e6*`, `e7_repeat`, `e8*`, `e9*`, `e10*`, `e11*`); each maps to a
doc in [`../docs/runs/`](../docs/runs/).

All outputs go to `runs/<run_name>/`:

```
runs/<run_name>/
  config.yaml            # copy of the config used
  best_model.zip         # checkpoint with highest mean episode reward
  final_model.zip        # checkpoint at end of training
  checkpoints/           # periodic saves
  learning_curve.csv     # timestep, ep_rew_mean, correctness_rate, truncation_rate, …
  probe_curve.csv        # periodic held-out eval (correctness, retention, gen_frac, …)
  probe_anchors.pkl      # cached baselines, reused on resume, skip recompute
  tb/                    # TensorBoard event files
```

**Monitoring during training**: the callback prints per rollout:

```
t=   14,672  rew=0.3142  correct=28.6%  align=0.412  ep_time=12.3s  episodes=4  trunc=100.0%  seen=4 (max 1x)  pass=0
t=   29,344  rew=0.3301  correct=31.4%  align=0.438  ep_time=11.8s  episodes=4  trunc=75.0%   seen=8 (max 1x)  pass=0
...
```

Or open TensorBoard:

```bash
tensorboard --logdir runs/my_run/tb
```

---

## Evaluation only

```bash
python scripts/eval.py \
  --model  runs/my_run/best_model \
  --config configs/run_none.yaml \
  --budget 180 \
  --n      100 \
  --output runs/my_run/eval_results.json
```

Runs six strategies online (one eviction per decode step, hard budget constraint):

| Strategy | Description |
|----------|-------------|
| `full` | No eviction (upper bound) |
| `learned` | Trained PPO policy |
| `streaming` | Attention sinks (first `--n-sinks` tokens) + most recent |
| `attn_layer` | Per-layer attention oracle (requires eager attention) |
| `kv_norm` | Per-layer: evict lowest current `‖K‖+‖V‖` norm |
| `random` | Uniform random eviction |

Also reports **correlation**: mean attention-rank percentile of the tokens the PPO
chose to evict (0.0 = always evicts least-attended token; 0.5 = random).

`attn_layer` and correlation require `attn_implementation='eager'`
(set automatically when `use_attention_shaping: true`). They are silently
skipped otherwise.

---

## Plot learning curves (standalone)

```bash
python scripts/plot_curves.py --run runs/my_run --window 10
```

Saves `runs/my_run/learning_curve.png` with three panels: episode reward,
correctness rate + alignment, and episode length.

---

## Quick local test

Use `configs/quickstart.yaml` to verify the loop runs end-to-end on CPU/MPS
in a few minutes (3 examples, small budget, 300k steps):

```bash
uv run python scripts/train.py --config configs/quickstart.yaml --run-name quickstart
```

---

## Experiment / analysis scripts

| Script | What it does | Doc |
|---|---|---|
| `scripts/wide_eval.py` | Wide paired eval (n=128, one stack, shared anchors) | [05](../docs/runs/05-wide-eval-regimen.md) |
| `scripts/oracle_eval.py` | Future-attention oracle vs heuristics (paired, eager) | [06](../docs/runs/06-oraculo-atencion-futura.md) |
| `scripts/trace_gen.py` | Full-cache traces with attention capture (for BC/rankers) | [07](../docs/runs/07-bc-match-oracle.md) |
| `scripts/screen_pool.py` | Full-cache solvability screening of a training pool | [08](../docs/runs/08-longgen-exploracion.md) |
| `scripts/eval_passkey.py` | Passkey retrieval arena (oracle vs heuristics) | [11](../docs/runs/11-dataset-causalidad.md) |
| `scripts/eval_prefill_compress.py` | HotpotQA prefill-compression arena (SnapKV-style) | [11](../docs/runs/11-dataset-causalidad.md) |
| `scripts/rank_predictability.py` | Offline ranker predictability control (GSM8K traces) | [11](../docs/runs/11-dataset-causalidad.md) |
| `scripts/passkey_ranker.py` | KVP offline recipe end-to-end on passkey (reference) | [12](../docs/runs/12-capstone-passkey.md) |
| `experiments/phase2-capacity/` | E0 screen + scaled-run drivers, `compare.py` | [03](../docs/runs/03-screen-capacidad.md)-[04](../docs/runs/04-runs-a-escala.md) |
| `experiments/phase3-dataset-causality/` | Phase-3 data, plots (`make_plots.py`), E11 keeper | [11](../docs/runs/11-dataset-causalidad.md)-[12](../docs/runs/12-capstone-passkey.md) |

---

## Tests

```bash
python -m pytest tests/ -q
```
