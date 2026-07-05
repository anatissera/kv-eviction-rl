#!/bin/bash
# Added for this project (UdeSA RL final). Not part of Apple's original ml-learning-to-evict release.
#
# Quick local PPO experiment (Mac CPU): proves the agent learns before GPU run.
#
# Prerequisites (same as run_experiment.sh):
#   1. .env file with PROJECT_ROOT and KVCOMPRESSION_DATA_ROOT
#   2. Preprocessing already done (configs/preprocess_qwen1b.yaml), so the
#      safetensors activations exist under $KVCOMPRESSION_DATA_ROOT.
#   3. uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#
# Usage:
#   bash run_ppo_local.sh                           # train layer 0, head 0
#   bash run_ppo_local.sh --layer 5 --head 1       # train a different head
#   bash run_ppo_local.sh --steps 100              # more steps (default: 50)
#
set -euo pipefail

# ── defaults ────────────────────────────────────────────────────────────────
LAYER=0
HEAD=0
STEPS=50
CONFIG="configs/train_qwen1b_ppo.yaml"

# ── arg parsing ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --layer) LAYER="$2"; shift 2;;
    --head)  HEAD="$2";  shift 2;;
    --steps) STEPS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# ── env ─────────────────────────────────────────────────────────────────────
[ -f .env ] || { echo "ERROR: .env missing. Copy from .env.template."; exit 1; }
set -a; source .env; set +a

echo "=========================================="
echo " PPO local training (CPU, $STEPS steps)"
echo " Layer: $LAYER  Head: $HEAD"
echo " Config: $CONFIG"
echo "=========================================="

uv run tune run \
    src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \
    --config "$CONFIG" \
    distributed=False \
    device=cpu \
    dtype=fp32 \
    target_layer_idx="$LAYER" \
    kv_head_idx="$HEAD" \
    "training.num_epochs_or_steps=$STEPS"

echo ""
echo "Done. Trained agent saved under: agents/grouped/qwen1b_ppo_validation/"
echo ""
echo "To run on the GPU server (SSH), use the GPU overrides:"
echo "  uv run tune run src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py \\"
echo "      --config configs/train_qwen1b_ppo.yaml \\"
echo "      distributed=False device=cuda dtype=bf16 \\"
echo "      'loader.train.batch_size=16' 'loader.eval.batch_size=8' \\"
echo "      'loader.train.dataloader_num_workers=2' 'loader.eval.dataloader_num_workers=2' \\"
echo "      'loader.train.prefetch_factor=2' 'loader.eval.prefetch_factor=2' \\"
echo "      'trainer.ema_burnin_steps=20' 'trainer.gumbel_sampling_chunk_size=16' \\"
echo "      'training.num_epochs_or_steps=200' 'training.eval_interval=50'"
