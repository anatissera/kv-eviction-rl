#!/usr/bin/env bash
# setup_and_train.sh — install dependencies and launch PPO training.
#
# Usage (on the GPU VM):
#   HF_TOKEN=<your_token> bash setup_and_train.sh [--run-name <name>] [--resume-from <ckpt>]
#
# Environment variables:
#   HF_TOKEN     — HuggingFace token for model download (required on first run)
#   HF_HOME      — HuggingFace cache dir (default: ~/.cache/huggingface)
#   CONFIG       — path to training config (default: configs/run_none.yaml)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/run_none.yaml}"
RUN_NAME=""
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)    RUN_NAME="$2";    shift 2 ;;
        --resume-from) RESUME_FROM="$2"; shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── 1. Install uv if missing ──────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[setup] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
echo "[setup] uv $(uv --version)"

# ── 2. Create / sync venv ────────────────────────────────────────────────────
echo "[setup] Syncing venv..."
uv sync

# ── 3. Verify CUDA ───────────────────────────────────────────────────────────
.venv/bin/python - <<'EOF'
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[setup] GPU: {name}  ({mem:.0f} GB)")
else:
    print("[setup] WARNING: CUDA not available — training will be very slow on CPU")
EOF

# ── 4. HuggingFace auth ───────────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] HF_TOKEN set — authenticated downloads enabled"
else
    echo "[setup] WARNING: HF_TOKEN not set — may fail on gated models"
fi

# ── 5. Launch training ────────────────────────────────────────────────────────
TRAIN_ARGS=(--config "$CONFIG")
[[ -n "$RUN_NAME"    ]] && TRAIN_ARGS+=(--run-name "$RUN_NAME")
[[ -n "$RESUME_FROM" ]] && TRAIN_ARGS+=(--resume-from "$RESUME_FROM")

echo "[setup] Starting training: ${TRAIN_ARGS[*]}"
echo "--------------------------------------------------------"
exec .venv/bin/python scripts/train.py "${TRAIN_ARGS[@]}"
