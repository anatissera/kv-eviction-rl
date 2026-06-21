# KV-Eviction Gym

Reinforcement learning agent that learns to evict tokens from a transformer's KV cache
**during generation** while preserving answer correctness on GSM8K.

The policy (MaskablePPO + per-token MLP) makes one eviction decision per layer per
decode step, keeping the cache at a fixed budget throughout generation. See
[METHOD.md](METHOD.md) for the full design.

---

## Setup

```bash
cd kv-eviction-gym
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2. On a GPU server with CUDA, install
`flash-attn` separately if you want to disable attention shaping and use
`flash_attention_2` for faster training.

---

## Full pipeline (recommended)

```bash
bash run.sh [run_name] [config] [eval_budget] [eval_n]
```

This runs four steps in sequence:

| Step | What happens |
|------|-------------|
| 1 | `pip install -e .` |
| 2 | Train MaskablePPO — saves best model, periodic checkpoints, learning curve CSV, TensorBoard events |
| 3 | Plot learning curves → `runs/<name>/learning_curve.png` |
| 4 | Evaluate on 100 GSM8K test examples → `runs/<name>/eval_results.json` |

**Examples:**

```bash
# Defaults: timestamped run name, configs/train.yaml, budget=180, n=100
bash run.sh

# Named run on the server
bash run.sh server_run1

# Custom config and eval budget
bash run.sh ablation_no_shaping configs/train_no_shaping.yaml 180 100
```

All outputs go to `runs/<run_name>/`:

```
runs/<run_name>/
  config.yaml            # copy of the config used
  best_model.zip         # checkpoint with highest mean episode reward
  final_model.zip        # checkpoint at end of training
  checkpoints/           # periodic saves (every 50k steps)
  learning_curve.csv     # timestep, ep_rew_mean, correctness_rate, alignment_mean
  learning_curve.png     # plot of the above
  tb/                    # TensorBoard event files
  eval_results.json      # per-example and summary eval results
```

---

## Training only

```bash
python scripts/train.py --config configs/train.yaml --run-name my_run
```

Key config options in `configs/train.yaml`:

| Field | Default | Notes |
|-------|---------|-------|
| `model_name` | `qwen-1.5b` | Model key; defaults loaded from `configs/model_defaults.json` |
| `n_examples` | `200` | GSM8K training examples |
| `budget_min` | `128` | Minimum cache capacity (tokens) |
| `budget_max` | `400` | Maximum cache capacity (tokens) |
| `total_timesteps` | `10000000` | ~1,000 episodes (SB3 counts ×28 envs per step) |
| `use_attention_shaping` | `true` | Dense reward shaping via reference generate; set `false` to halve training time |
| `attention_weight` | `0.3` | Mix: `0` = pure correctness, `1` = pure attention alignment |
| `hidden` | `64` | Policy MLP hidden size |

**Monitoring during training** — the callback prints per rollout:

```
t=   14,672  rew=0.3142  correct=28.6%  align=0.412
t=   29,344  rew=0.3301  correct=31.4%  align=0.438
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
  --config configs/train.yaml \
  --budget 180 \
  --n      100 \
  --output runs/my_run/eval_results.json
```

Runs six strategies online (one eviction per decode step, hard budget constraint):

| Strategy | Description |
|----------|-------------|
| `full` | No eviction — upper bound |
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
bash run.sh quickstart configs/quickstart.yaml 128 10
```

---

## Tests

```bash
python -m pytest tests/ -q
```
