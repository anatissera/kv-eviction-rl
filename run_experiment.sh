#!/bin/bash
#
# End-to-end KVP pipeline-validation run with Qwen2-1.5B-Instruct:
#   data generation (unroll_and_store) -> agent training -> eval vs heuristic baseline.
#
# Designed for a single GPU. Requires the NVIDIA driver to be active
# (e.g. `sudo modprobe nvidia nvidia_uvm` if nvidia-smi fails), `uv`, and model weights at
# models/Qwen2-1.5B-Instruct/ (see PLAN.md). Tunables via environment variables below.
#
set -euo pipefail

# --------------------------- configuration ---------------------------------
NUM_EXAMPLES="${NUM_EXAMPLES:-100}"     # data-gen samples (capped via total_num_chunks)
SWEEP="${SWEEP:-qwen1b_validation}"
LAYERS="${LAYERS:-0}"                    # agents to train; space-separated. Full model: $(seq 0 27)
HEADS="${HEADS:-0 1}"                    # Qwen2-1.5B has 2 KV heads
EVAL_SAMPLES="${EVAL_SAMPLES:-1}"
CACHE_SIZE="${CACHE_SIZE:-256}"          # KV tokens kept per head during eval
EVAL_GEN="${EVAL_GEN:-64}"              # tokens generated per eval sample
DEVICE="${DEVICE:-cuda:0}"

PREP_CONFIG="configs/preprocess_qwen1b.yaml"
TRAIN_CONFIG="configs/train_qwen1b.yaml"
DATAGEN_ENTRY="src/kvcompression/entrypoints/datagen/unroll_and_store.py"
TRAIN_ENTRY="src/kvcompression/entrypoints/rl/train_agent_sampler_distributed.py"

LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"

# --------------------------- helpers ---------------------------------------
log()  { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
fail() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2; exit 1; }
CURRENT_STEP="startup"
trap 'fail "step \"$CURRENT_STEP\" failed (line $LINENO). See logs in $LOG_DIR/."' ERR

# Load local env so we know KVCOMPRESSION_DATA_ROOT etc. in the shell.
[ -f .env ] || fail "Missing .env (copy from .env.template). See PLAN.md."
set -a; # shellcheck disable=SC1091
source .env; set +a
DATA_DIR="${KVCOMPRESSION_DATA_ROOT}/gqa_safetensors/simonjegou_ruler/Qwen2-1.5B-Instruct/temperature_0.00"

# --------------------------- prechecks -------------------------------------
CURRENT_STEP="precheck: GPU"
log "===== Precheck: GPU ====="
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found."
if ! nvidia-smi >/dev/null 2>&1; then
  fail "GPU not reachable. The NVIDIA kernel module is likely not loaded. Try: sudo modprobe nvidia nvidia_uvm"
fi
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv

CURRENT_STEP="precheck: dependencies"
log "===== Precheck: dependencies ====="
command -v uv >/dev/null 2>&1 || fail "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
uv run python -c "import kvcompression, torchtune, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || fail "Environment not ready. Run: uv sync"
[ -f models/Qwen2-1.5B-Instruct/model.safetensors ] \
  || fail "Model weights missing. Run: uv run tune download Qwen/Qwen2-1.5B-Instruct --output-dir models/Qwen2-1.5B-Instruct --ignore-patterns 'original/*'"
[ -f assets/ruler_split/train.txt ] && [ -f assets/ruler_split/val.txt ] \
  || fail "Missing assets/ruler_split/{train,val}.txt"

# --------------------------- Phase 1: data generation ----------------------
CURRENT_STEP="data generation"
log "===== Phase 1: data generation (~${NUM_EXAMPLES} examples) ====="
t0=$(date +%s)
log "Counting RULER examples to compute chunking..."
DATASET_LEN=$(uv run python -c "from datasets import load_dataset; print(len(load_dataset('simonjegou/ruler', data_dir='4096', split='test')))")
CHUNKS=$(( (DATASET_LEN + NUM_EXAMPLES - 1) / NUM_EXAMPLES ))
[ "$CHUNKS" -lt 1 ] && CHUNKS=1
log "RULER test size=$DATASET_LEN -> total_num_chunks=$CHUNKS, current_chunk=0 (~$NUM_EXAMPLES samples)"
uv run tune run "$DATAGEN_ENTRY" \
  --config "$PREP_CONFIG" \
  device="$DEVICE" total_num_chunks="$CHUNKS" current_chunk=0 \
  2>&1 | tee "$LOG_DIR/datagen.log"
t1=$(date +%s); DATAGEN_SECS=$((t1 - t0))

[ -d "$DATA_DIR" ] || fail "Expected data dir not found: $DATA_DIR"
log "Copying RULER train/val splits into the data directory..."
cp assets/ruler_split/*.txt "$DATA_DIR/"
N_SAMPLES=$(find "$DATA_DIR" -maxdepth 1 -type d -name 'sample_*' | wc -l)
log "Data generation produced $N_SAMPLES sample(s) in ${DATAGEN_SECS}s."

# --------------------------- Phase 2: training -----------------------------
CURRENT_STEP="training"
log "===== Phase 2: training agents (sweep=$SWEEP; layers='$LAYERS' heads='$HEADS') ====="
t0=$(date +%s)
N_AGENTS=0
for L in $LAYERS; do
  for H in $HEADS; do
    CURRENT_STEP="training layer $L head $H"
    log "--- training agent layer=$L head=$H ---"
    uv run tune run "$TRAIN_ENTRY" \
      --config "$TRAIN_CONFIG" distributed=False \
      device="$DEVICE" target_layer_idx="$L" kv_head_idx="$H" sweep_name="$SWEEP" \
      2>&1 | tee "$LOG_DIR/train_L${L}_H${H}.log"
    N_AGENTS=$((N_AGENTS + 1))
  done
done
t1=$(date +%s); TRAIN_SECS=$((t1 - t0))
log "Trained $N_AGENTS agent(s) in ${TRAIN_SECS}s."

# Best-effort final-loss extraction from the last training log.
LAST_TRAIN_LOG=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1 || true)
FINAL_LOSS="(see $LOG_DIR/train_*.log)"
if [ -n "${LAST_TRAIN_LOG:-}" ]; then
  EXTRACTED=$(grep -oE 'total_loss[=: ]+[-0-9.eE]+' "$LAST_TRAIN_LOG" | tail -1 || true)
  [ -n "$EXTRACTED" ] && FINAL_LOSS="$EXTRACTED (from $(basename "$LAST_TRAIN_LOG"))"
fi

# --------------------------- Phase 3: evaluation ---------------------------
CURRENT_STEP="evaluation"
log "===== Phase 3: evaluation vs RandomPress baseline ====="
t0=$(date +%s)
uv run python scripts/eval_vs_baseline.py \
  --preprocess-config "$PREP_CONFIG" \
  --sweep-name "$SWEEP" \
  --num-samples "$EVAL_SAMPLES" \
  --cache-size "$CACHE_SIZE" \
  --max-new-tokens "$EVAL_GEN" \
  --device "$DEVICE" \
  2>&1 | tee "$LOG_DIR/eval.log"
t1=$(date +%s); EVAL_SECS=$((t1 - t0))

# --------------------------- summary ---------------------------------------
CURRENT_STEP="summary"
log "============================================================"
log "EXPERIMENT SUMMARY"
log "  Model:            Qwen2-1.5B-Instruct"
log "  Data generation:  ${N_SAMPLES} samples in ${DATAGEN_SECS}s"
log "  Training:         ${N_AGENTS} agent(s) in ${TRAIN_SECS}s"
log "  Final train loss: ${FINAL_LOSS}"
log "  Evaluation:       ${EVAL_SECS}s"
if [ -f output/eval/eval_results.json ]; then
  log "  Eval results (token-match vs uncompressed reference):"
  uv run python -c "import json;d=json.load(open('output/eval/eval_results.json'));print('    RandomPress baseline:', d['random_baseline_mean']);print('    Learned KVP agents : ', d['learned_mean'])" || true
  log "  Full eval JSON: output/eval/eval_results.json"
fi
log "  Logs: $LOG_DIR/"
log "============================================================"
log "Done."
