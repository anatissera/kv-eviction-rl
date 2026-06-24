# PLAN — End-to-end KVP experiment with Qwen2-1.5B-Instruct

Goal: run Apple's KVP ("Learning to Evict from Key-Value Cache") pipeline end-to-end with a
small model — **Qwen2-1.5B-Instruct** instead of the paper's Qwen2.5-7B-Instruct — to validate
the full flow: **data generation → agent training → evaluation vs a heuristic baseline**.

This plan reflects the actual code in this repo (entrypoints run through torchtune's
`tune run`, the LLM is built by a **torchtune model builder** rather than HuggingFace
`transformers`, and agent training is fully decoupled from the LLM).

> Environment note for this machine: the NVIDIA driver (580) + CUDA toolkit are installed but
> the kernel module is **not loaded** (`nvidia-smi` fails). Activate it with
> `sudo modprobe nvidia nvidia_uvm` before running. The local GPU is a **4 GB** RTX 3050 Mobile
> (below the 8 GB target) — data generation will likely need a smaller context on 4 GB (see
> "Anticipated problems"). All artifacts below are prepared to run on any single CUDA GPU.

---

## 1. Data generation

**Entrypoint:** `src/kvcompression/entrypoints/datagen/unroll_and_store.py` (torchtune recipe).

**What it does:** loads the LLM (torchtune) + RULER from HuggingFace, greedily generates a
continuation per example, then runs one full forward with an attention hook that captures the
per-layer **Q/K/V** tensors and writes them to disk for offline agent training.

**Command (driven by `run_experiment.sh`):**
```bash
uv run tune run src/kvcompression/entrypoints/datagen/unroll_and_store.py \
    --config configs/preprocess_qwen1b.yaml \
    device=cuda:0 total_num_chunks=<N> current_chunk=0
```
`ChunkSampler` has no max-samples knob, so to cap at ~100 examples the script computes
`total_num_chunks = ceil(len(RULER) / 100)` and takes `current_chunk=0`.

**Dataset:** `simonjegou/ruler`, config `"4096"` (the **smallest** RULER bucket — there is no
1024/2048 subset). The tokenizer `max_seq_len: 2048` in the config truncates each prompt to
≤2048 tokens to honor the target and keep the footprint small.

**Output layout** (under the gitignored `data/`, set by `KVCOMPRESSION_DATA_ROOT`):
```
data/gqa_safetensors/simonjegou_ruler/Qwen2-1.5B-Instruct/temperature_0.00/
  sample_NNNNNNN/
    all_text.txt  generated_text.txt
    all_tokens.safetensors  prompt_ntokens.safetensors
    layer_NNNNNN/
      input_pos.safetensors
      kv_head_NNN/attention_tensors.safetensors   # {k:[S,128], v:[S,128], q_group:[6,S,128]}
  train.txt  val.txt        # copied from assets/ruler_split/ by run_experiment.sh
```

**Estimates (100 examples, Qwen2-1.5B bf16, ≤2048 ctx + 128 gen):**
- Time: ~5–15 min on an 8 GB+ GPU (≈3–9 s/sample).
- Disk: a few hundred MB (all 28 layers × 2 heads of Q/K/V per sample).
- VRAM peak ≈ **6–7 GB**: ~3.1 GB weights + KV cache allocated at the model's full
  `max_seq_len` (32768) + the dense extraction forward. **On 4 GB this will likely OOM** —
  reduce `tokenizer.max_seq_len` and `max_new_tokens` (e.g. 1024 / 64).

---

## 2. Training

**Entrypoint:** `src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py`.

**Key fact:** training **never loads the LLM**. It reads the stored Q/K/V (`QKVDataset`) for a
single `(target_layer_idx, kv_head_idx)` and trains a small MLP policy
(`KVSamplingGroupedQueryAgent`) with an RLOO + Gumbel trainer and an attention-based reward
(`FutureAttentionAucNormalizedReward`) computed by an `Oracle`. It is lightweight (CPU-feasible).

**Command (per agent):**
```bash
uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config configs/train_qwen1b.yaml distributed=False \
    target_layer_idx=<L> kv_head_idx=<H> sweep_name=qwen1b_validation
```

**Reads:** `data/gqa_safetensors/simonjegou_ruler/Qwen2-1.5B-Instruct/temperature_0.00` +
`train.txt`/`val.txt`.

**Produces:** checkpoints under (gitignored `agents/`):
```
agents/grouped/<sweep_name>/layer_NNNNNN/kv_head_NNN/
  best_ckpt.pth                # best-by-eval_reward (used by eval / SamplerAgentPress)
  checkpoints/<step>.pth       # periodic resume checkpoints
```
Each `.pth` stores `model_state_dict`, `ema_model_state_dict`, optimizer/scheduler, and the
full `config` (so the agent architecture is reconstructable from the checkpoint alone).

**Full model = 28 layers × 2 KV heads = 56 agents.** `run_experiment.sh` trains a configurable
subset (default: `LAYERS=0 HEADS="0 1"`, i.e. 2 agents) for a fast validation; set
`LAYERS="$(seq 0 27)"` for the full sweep.

**Estimates (per agent, 200 steps, batch 16):** ~1–5 min, <1 GB VRAM. Full 56-agent sweep,
run sequentially, ≈1–4 h. (The paper trains these across 8 GPUs.)

---

## 3. Model compatibility (Qwen2-1.5B-Instruct)

**HuggingFace model ID:** `Qwen/Qwen2-1.5B-Instruct` (verified config):

| Property | Qwen2.5-7B (repo default) | **Qwen2-1.5B (this run)** |
|---|---|---|
| torchtune builder | `qwen2_5.qwen2_5_7b_instruct` | **`qwen2.qwen2_1_5b`** |
| hidden size | 3584 | 1536 |
| layers | 28 | 28 |
| attention heads | 28 | 12 |
| **KV heads** | 4 | **2** |
| **head_dim** | 128 | **128 (unchanged)** |
| GQA group size (`num_queries_per_group`) | 7 | **6** |
| **# agents** (layers × KV heads) | 112 | **56** |
| weights | 4 shards | single `model.safetensors` |

**Hardcoded assumptions in the repo and how they're handled:**
- The LLM is chosen by config (`model._component_`), so swapping to Qwen2-1.5B is a **config
  change, not a code change**. `configs/preprocess_qwen1b.yaml` sets the `qwen2_1_5b` builder,
  `qwen2_tokenizer`, single-shard `checkpoint_files`, and `model_type: QWEN2`.
- The **agent `head_dim` stays 128** (same as 7B), so the agent architecture is unchanged.
  `num_queries_per_group` is set to 6 for correctness but is only consumed when
  `query_aggregation="learned"` (the default config uses neither queries nor that path).
- `src/kvcompression/data/kvpress_dataset.py` imports `Qwen2_5Tokenizer` **only as a type
  hint**; the real tokenizer is injected and `tokenize_messages` exists on the qwen2 tokenizer
  too — no change needed. (Qwen2 and Qwen2.5 share the BPE vocab/merges.)

**Minimum required changes: config-only.** No model/source patches are needed.

---

## 4. Environment

**Requirements:** Python ≥3.10, [`uv`](https://docs.astral.sh/uv/), a CUDA GPU with an active
driver. Dependencies are pinned in `pyproject.toml` / `uv.lock` (torch 2.10 cu128, torchtune
0.6.1, transformers 4.55.4, flash-attn 2.8.3, triton, datasets, safetensors, wandb, s3fs …).

**Install & weights:**
```bash
uv sync                                   # builds the env (flash-attn CUDA build is skipped)
cp .env.template .env                      # or use the provided .env (local paths, no S3)
uv run tune download Qwen/Qwen2-1.5B-Instruct \
    --output-dir models/Qwen2-1.5B-Instruct --ignore-patterns "original/*"
```

**`.env` (provided):** `KVCOMPRESSION_DATA_ROOT=data`, `KVCOMPRESSION_MODEL_ID=Qwen/Qwen2-1.5B-Instruct`,
`AWS_EC2_METADATA_DISABLED=true` (so the credential-less S3 upload fails instantly instead of
hanging), `DISABLE_TORCH_COMPILER=1`, `WANDB_MODE=disabled`. `PROJECT_ROOT` is derived from the
git root automatically.

**GPU memory:** target 8 GB. Data generation is the bottleneck (~6–7 GB at 2048 ctx); training
and eval are comfortable under 8 GB. For the 4 GB local GPU, shrink `tokenizer.max_seq_len`
and `max_new_tokens`.

**Run everything:**
```bash
sudo modprobe nvidia nvidia_uvm     # only if nvidia-smi currently fails
./run_experiment.sh                 # tunables: NUM_EXAMPLES, LAYERS, HEADS, CACHE_SIZE, ...
```

---

## 5. Evaluation

The repo has **no standalone CLI evaluator and no StreamingLLM/H2O baselines**. The available
heuristic press is **`RandomPress`**; the learned strategy is **`SamplerAgentPress`**; the
intended qualitative demo is `notebooks/inference_demo.ipynb`. Generation with compression goes
through `generate_with_compression` + `KVCompressor` (verified by
`tests/test_generate_with_compression.py`).

`scripts/eval_vs_baseline.py` (added here) does a small quantitative comparison at a fixed KV
budget, decoding each prompt three ways and reporting token-match vs the uncompressed reference:
1. **reference** — `DummyCompressionStrategy` (no eviction; equals plain greedy decoding).
2. **random** — global `RandomPress` heuristic baseline.
3. **learned** — per-(layer,head) `SamplerAgentPress` composite, **only if all 56 agents exist
   locally**; otherwise skipped with a message. A meaningful whole-model learned evaluation
   requires the full sweep (or the S3/registry → composite → notebook route).

---

## 6. Anticipated problems

- **GPU driver not loaded** (current state): `nvidia-smi` fails → `sudo modprobe nvidia nvidia_uvm`.
- **4 GB VRAM** (local): data generation may OOM at 2048 ctx. Mitigate via `tokenizer.max_seq_len`
  (e.g. 1024) and `max_new_tokens` (e.g. 64), or run data-gen on an 8 GB+ machine.
- **`flash-attn` install**: the lockfile sets `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`, so `uv sync`
  installs it without compiling CUDA. torchtune uses SDPA, so flash-attn is not on the hot path.
- **S3 upload**: data-gen and training try to upload to `remote_dir` and delete locals on success.
  With no credentials the upload **fails fast** (logged, harmless) and **locals are preserved** —
  `AWS_EC2_METADATA_DISABLED=true` avoids IMDS hangs. The trainer requires a non-empty
  `remote_dir`, satisfied by the placeholder `s3://kvp-local-noupload`.
- **RULER min context is 4096**: no 1024/2048 subset exists, so we truncate via the tokenizer.
  Truncation can cut RULER "needles," so task scores aren't meaningful — this run validates the
  **pipeline**, not RULER accuracy.
- **Distributed defaults**: `train_single_head.yaml` defaults `distributed: True`; the new
  `train_qwen1b.yaml` sets `distributed: False` for single-process local runs.
- **Learned whole-model eval** needs all 56 agents; the default 2-agent run will skip the learned
  path. Expand `LAYERS`/`HEADS` to train the full sweep.
- **`torch.compile`/triton**: disabled via `DISABLE_TORCH_COMPILER=1` and `compile_agent: False`
  to reduce startup cost and avoid inductor/triton issues on the validation run.
```

---

## Files added/changed for this experiment
- `configs/preprocess_qwen1b.yaml` — Qwen2-1.5B data-generation config.
- `configs/train_qwen1b.yaml` — single-agent training config (short, single-GPU).
- `scripts/eval_vs_baseline.py` — local eval: learned agents vs RandomPress vs uncompressed.
- `run_experiment.sh` — orchestrates data-gen → training → eval with timed summary.
- `.env` — local paths, no S3 (gitignored).
- `.gitignore` — added `/output` (logs, eval JSON, summaries).
