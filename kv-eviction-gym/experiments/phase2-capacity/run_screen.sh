#!/bin/bash
# E0 capacity screen driver (VM-side, restart-safe for spot preemption).
# Runs the named screen variants sequentially; per-variant marker files survive a
# preempt so a restart only redoes the in-progress one. Split the variant list
# across two VMs to parallelize.
#   Usage:  bash run_screen.sh e0_baseline e0_rich e0_rich_s4 e0_rich_warm e0_attn
set -u
cd ~/repo || exit 1
source .venv/bin/activate
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=1048576
CFGDIR=experiments/phase2-capacity/screen_configs
mkdir -p ~/screen
DL=~/screen/driver.log
echo "==== screen driver (re)start $(date -u) : $* ====" >> "$DL"

if [ ! -f ~/screen/cache.cleared ]; then
  rm -f ~/.kv_eviction_cache/* 2>/dev/null; touch ~/screen/cache.cleared
  echo "cleared kv cache $(date -u)" >> "$DL"
fi

for v in "$@"; do
  if [ ! -f ~/screen/$v.done ]; then
    echo "START $v $(date -u)" >> "$DL"
    python scripts/train.py --config $CFGDIR/$v.yaml --run-name $v \
        > ~/screen/$v.log 2>&1 && touch ~/screen/$v.done
    echo "END $v rc=$? $(date -u)" >> "$DL"
  fi
done

alldone=1
for v in "$@"; do [ -f ~/screen/$v.done ] || alldone=0; done
if [ $alldone -eq 1 ]; then touch ~/screen/ALL_DONE; echo "==== ALL DONE $(date -u) ====" >> "$DL"; fi
