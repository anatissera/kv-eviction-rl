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
