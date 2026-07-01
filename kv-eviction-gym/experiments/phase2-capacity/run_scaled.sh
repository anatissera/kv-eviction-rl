#!/bin/bash
# Scaled-run driver (VM-side) with CHECKPOINT-RESUME for spot resilience.
# Unlike run_screen/run_phase2 (per-arm markers only → a preempt restarts the arm
# from 0), this resumes each arm from its latest checkpoint, so a 2M run loses only
# the steps since the last checkpoint (CheckpointCallback saves every checkpoint_freq).
#   Usage:  bash run_scaled.sh <cfg-path>:<run-name> [<cfg-path>:<run-name> ...]
# e.g.      bash run_scaled.sh configs/e1_rich.yaml:s_e1_rich
set -u
cd ~/repo || exit 1
source .venv/bin/activate
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
mkdir -p ~/scaled
DL=~/scaled/driver.log
echo "==== scaled driver (re)start $(date -u) : $* ====" >> "$DL"

if [ ! -f ~/scaled/cache.cleared ]; then
  rm -f ~/.kv_eviction_cache/* 2>/dev/null; touch ~/scaled/cache.cleared
  echo "cleared kv cache $(date -u)" >> "$DL"
fi

latest_ckpt() {  # echo path of highest-step ckpt_*.zip for run $1, or empty
  ls -1 ~/repo/runs/"$1"/checkpoints/ckpt_*_steps.zip 2>/dev/null \
    | sed -E 's/.*ckpt_([0-9]+)_steps\.zip/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-
}

for spec in "$@"; do
  cfg=${spec%%:*}; run=${spec##*:}
  [ -f ~/scaled/$run.done ] && continue
  ck=$(latest_ckpt "$run")
  if [ -n "$ck" ]; then
    echo "RESUME $run from $ck $(date -u)" >> "$DL"
    python scripts/train.py --config "$cfg" --run-name "$run" --resume-from "$ck" \
        >> ~/scaled/$run.log 2>&1 && touch ~/scaled/$run.done
  else
    echo "START $run ($cfg) $(date -u)" >> "$DL"
    python scripts/train.py --config "$cfg" --run-name "$run" \
        > ~/scaled/$run.log 2>&1 && touch ~/scaled/$run.done
  fi
  echo "END $run rc=$? $(date -u)" >> "$DL"
done

alldone=1
for spec in "$@"; do run=${spec##*:}; [ -f ~/scaled/$run.done ] || alldone=0; done
if [ $alldone -eq 1 ]; then touch ~/scaled/ALL_DONE; echo "==== ALL DONE $(date -u) ====" >> "$DL"; fi
