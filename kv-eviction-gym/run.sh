#!/usr/bin/env bash
# run.sh — end-to-end: env setup → training → learning curve plot → evaluation
#
# Usage:
#   bash run.sh                              # defaults
#   bash run.sh my_run configs/run_none.yaml    # named run, custom config
#   bash run.sh my_run configs/run_none.yaml 180 100   # + eval budget, n examples
#
# Arguments (all optional, positional):
#   $1  RUN_NAME  — label for runs/<name>/ directory  (default: timestamp)
#   $2  CONFIG    — training config yaml               (default: configs/run_none.yaml)
#   $3  BUDGET    — tokens to keep at eval             (default: 180)
#   $4  EVAL_N    — number of test examples for eval   (default: 100)

set -euo pipefail

RUN_NAME="${1:-run_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${2:-configs/run_none.yaml}"
BUDGET="${3:-180}"
EVAL_N="${4:-100}"
RESUME_FROM="${5:-}"

RUN_DIR="runs/${RUN_NAME}"
BEST_MODEL="${RUN_DIR}/best_model"
EVAL_OUT="${RUN_DIR}/eval_results.json"

echo "========================================"
echo "  KV-eviction training + eval pipeline"
echo "========================================"
echo "  run name : ${RUN_NAME}"
echo "  config   : ${CONFIG}"
echo "  budget   : ${BUDGET}"
echo "  eval n   : ${EVAL_N}"
echo "  run dir  : ${RUN_DIR}"
echo ""

# ── 1. Environment setup ─────────────────────────────────────────────────────
echo "[1/4] Setting up Python environment..."

# Prefer uv if available (faster), fall back to pip
if command -v uv &>/dev/null; then
    uv pip install -e . --quiet
else
    pip install -e . --quiet
fi

echo "      Done."
echo ""

# ── 2. Training ───────────────────────────────────────────────────────────────
echo "[2/4] Training MaskablePPO..."
echo "      Outputs → ${RUN_DIR}/"
echo ""

RESUME_ARG=""
if [ -n "${RESUME_FROM}" ]; then
    RESUME_ARG="--resume-from ${RESUME_FROM}"
fi

python scripts/train.py \
    --config   "${CONFIG}" \
    --run-name "${RUN_NAME}" \
    ${RESUME_ARG}

echo ""
echo "      Training complete."
echo "      Best model : ${BEST_MODEL}.zip"
echo ""

# ── 3. Learning curve plot ────────────────────────────────────────────────────
echo "[3/4] Plotting learning curves..."

python scripts/plot_curves.py \
    --run    "${RUN_DIR}" \
    --window 10

echo ""

# ── 4. Evaluation ─────────────────────────────────────────────────────────────
echo "[4/4] Evaluating on GSM8K test set (n=${EVAL_N}, budget=${BUDGET})..."
echo ""

python scripts/eval.py \
    --model  "${BEST_MODEL}" \
    --config "${CONFIG}" \
    --budget "${BUDGET}" \
    --n      "${EVAL_N}" \
    --output "${EVAL_OUT}"

echo ""
echo "========================================"
echo "  Pipeline complete."
echo "========================================"
echo "  Run directory  : ${RUN_DIR}/"
echo "  Best model     : ${BEST_MODEL}.zip"
echo "  Learning curve : ${RUN_DIR}/learning_curve.png"
echo "  Eval results   : ${EVAL_OUT}"
echo "  TensorBoard    : tensorboard --logdir ${RUN_DIR}/tb"
echo "========================================"
