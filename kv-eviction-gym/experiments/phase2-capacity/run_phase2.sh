#!/bin/bash
# Phase 2 driver (VM-side, restart-safe for spot preemption).
# Runs one or more arms sequentially; per-arm marker files survive a preempt so a
# restart only redoes the in-progress arm. Usage (on the VM):
#     bash run_phase2.sh e1_rich:e1_rich e2_rich_s4:e2_rich_s4
# where each arg is  <config-basename>:<run-name>.
# With ONE VM, pass both arms (they chain). With TWO VMs, launch one arm on each.
set -u
cd ~/repo || exit 1
source .venv/bin/activate
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
mkdir -p ~/phase2
DL=~/phase2/driver.log
echo "==== phase2 driver (re)start $(date -u) : args=$* ====" >> "$DL"

# clear stale raw-format free-growth cache ONCE (guard against restart re-wipe).
if [ ! -f ~/phase2/cache.cleared ]; then
  rm -f ~/.kv_eviction_cache/* 2>/dev/null; touch ~/phase2/cache.cleared
  echo "cleared kv cache $(date -u)" >> "$DL"
fi

for spec in "$@"; do
  cfg=${spec%%:*}; run=${spec##*:}
  if [ ! -f ~/phase2/$run.done ]; then
    echo "START $run (configs/$cfg.yaml) $(date -u)" >> "$DL"
    python scripts/train.py --config configs/$cfg.yaml --run-name $run \
        > ~/phase2/$run.log 2>&1 && touch ~/phase2/$run.done
    echo "END $run rc=$? $(date -u)" >> "$DL"
  fi
done

# ALL_DONE only if every requested arm finished
alldone=1
for spec in "$@"; do run=${spec##*:}; [ -f ~/phase2/$run.done ] || alldone=0; done
if [ $alldone -eq 1 ]; then touch ~/phase2/ALL_DONE; echo "==== ALL DONE $(date -u) ====" >> "$DL"; fi
