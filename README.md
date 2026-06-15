# Learning to Evict from Key-Value Cache

[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-1f6feb?logo=openreview)](https://openreview.net/forum?id=0OevIlRMYN)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10238-b31b1b.svg)](https://arxiv.org/abs/2602.10238)

This software project accompanies the research paper:
_Learning to Evict from Key-Value Cache_
by _Luca Moschella, Laura Manduchi, Ozan Sener_.

![](assets/teaser.png)


We introduce KV Policy (KVP), a framework of lightweight per-head RL agents that learn to rank KV cache entries by their predicted future utility, enabling adaptive eviction without modifying the underlying LLM or adding inference overhead.

## Getting Started

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.template .env  # edit with your paths
```

Download model weights:

```bash
uv run tune download Qwen/Qwen2.5-7B-Instruct \
    --output-dir models/Qwen2.5-7B-Instruct
```

## Dataset Generation

Extract Q, K, V activations from the RULER dataset:

```bash
uv run tune run src/kvcompression/entrypoints/datagen/unroll_and_store.py \
    --config configs/preprocess_ruler.yaml \
    device=cuda:0 total_num_chunks=1 current_chunk=0
```

For parallel processing across multiple GPUs, follow the commented-out chunking pattern in `configs/preprocess_ruler.yaml`.

Copy the pre-generated train/val splits to the output directory:

```bash
cp assets/ruler_split/*.txt \
    data/gqa_safetensors/simonjegou_ruler/Qwen2.5-7B-Instruct/temperature_0.00/
```

## Training

Qwen2.5-7B-Instruct uses Grouped Query Attention with 28 layers and 4 KV heads, requiring 112 agents (one per layer-head pair).

Train a single agent:

```bash
uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_single_head.yaml distributed=False
```

Override layer and head:

```bash
uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_single_head.yaml \
    target_layer_idx=15 kv_head_idx=2 distributed=False
```

Each agent is trained with DDP across 8 GPUs. To train all 112 agents, follow the commented-out sweep pattern in `configs/train_single_head.yaml`.

Trained agents are saved to `agents/grouped/<sweep_name>/layer_<NNNNNN>/kv_head_<NNN>/`.

## Inference

After training, generate a config composite for the trained agents:

```bash
uv run python scripts/orchestration/agent_registry.py list-sweeps
uv run python scripts/orchestration/agent_registry.py discover <sweep_name>
uv run python scripts/orchestration/assignment_generator.py write-composites <sweep_name>
```

See [`notebooks/inference_demo.ipynb`](notebooks/inference_demo.ipynb) for a demo of generation with learned KV cache compression.

---

## Qwen2-1.5B Experiment (PPO vs RLOO)

This branch extends the original paper by (1) running on the smaller **Qwen2-1.5B-Instruct** model and (2) adding **PPO** as an alternative to the paper's RLOO algorithm.

### Prerequisites

Requires Python >= 3.10, [uv](https://docs.astral.sh/uv/), and a CUDA GPU with ≥ 8 GB VRAM.

```bash
uv sync
cp .env.template .env   # set PROJECT_ROOT and KVCOMPRESSION_DATA_ROOT
```

Download model weights:

```bash
uv run tune download Qwen/Qwen2-1.5B-Instruct \
    --output-dir models/Qwen2-1.5B-Instruct --ignore-patterns "original/*"
```

If `nvidia-smi` fails on your machine, load the kernel module first:

```bash
sudo modprobe nvidia nvidia_uvm
```

### Run the full pipeline (data gen → train → eval)

```bash
./run_experiment.sh
```

This script runs three phases in sequence:
1. **Data generation** — loads Qwen2-1.5B, captures Q/K/V activations from ~100 RULER examples, writes safetensors to `$KVCOMPRESSION_DATA_ROOT`
2. **Training** — trains one RLOO agent per `(layer, head)` pair specified by `LAYERS`/`HEADS`
3. **Evaluation** — compares learned agents vs `RandomPress` heuristic baseline

Key tunables (pass as env vars):

```bash
NUM_EXAMPLES=100   # how many RULER samples to generate
LAYERS="0 1 2"     # which layers to train agents for (default: 0 only)
HEADS="0 1"        # Qwen2-1.5B has 2 KV heads per layer
SWEEP=qwen1b_validation
DEVICE=cuda:0
./run_experiment.sh
```

For the full 56-agent model (all 28 layers × 2 heads):

```bash
LAYERS="$(seq 0 27)" HEADS="0 1" ./run_experiment.sh
```

### Train with PPO instead of RLOO

```bash
uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_qwen1b_ppo.yaml \
    distributed=False device=cuda dtype=bf16 \
    target_layer_idx=0 kv_head_idx=0 \
    'loader.train.batch_size=16' 'loader.eval.batch_size=8' \
    'loader.train.dataloader_num_workers=2' \
    'training.num_epochs_or_steps=200' 'training.eval_interval=50'
```

### Compare RLOO vs PPO learning curves

After running both algorithms, generate the side-by-side comparison plot:

```bash
python scripts/plot_rloo_vs_ppo.py   # auto-discovers latest checkpoints
# or point explicitly:
python scripts/plot_rloo_vs_ppo.py \
    --ppo  agents/grouped/qwen1b_ppo_validation/.../checkpoints/000000000200.pth \
    --rloo agents/grouped/qwen1b_rloo_cpu/.../checkpoints/000000000200.pth \
    --out  learning_curves.png
```

### Quick local test (Mac CPU, no GPU needed)

Proves the code runs without real data or a GPU:

```bash
bash run_ppo_local.sh          # PPO,  50 steps, layer 0 head 0
# or RLOO:
tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_qwen1b_rloo_cpu.yaml distributed=False
```

---

## Citation

If you find our work useful, please cite the following paper:

```bibtex
@inproceedings{
    moschella2026learningtoevict,
    title={Learning to Evict from Key-Value Cache},
    author={Luca Moschella and Laura Manduchi and Ozan Sener},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026},
    url={https://openreview.net/forum?id=0OevIlRMYN}
}
```

## Acknowledgements

Our codebase is built using multiple open-source contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details.

## License

Please check out the repository [LICENSE](LICENSE) before using the provided code.
