#!/bin/bash
# E7 queue + watch on kvp-ab (SPOT):
#   Phase 0: wait until eval_e6c finishes (its watcher downloads
#            longgen_probe_curve.csv locally and stops the VM).
#   Phase 1: start VM, launch s_e7 via run_scaled.sh (ckpt-resume).
#   Phase 2: every poll: DOWNLOAD live curves locally (user wants running
#            results), handle preemption/proc-death, ALL_DONE → final
#            download + STOP VM.
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
SPEC="configs/e7_repeat.yaml:s_e7"
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_e7.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }
scpdl(){ G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap "$VM:$1" "$2" 2>/dev/null; }
launch(){ sshc "rm -f ~/scaled/ALL_DONE; tmux kill-session -t e7 2>/dev/null; \
  tmux new-session -d -s e7 'bash ~/run_scaled.sh $SPEC'"; }

echo "================ e7 watcher start $(date -u) ================" >> "$LOG"

# Phase 0 — wait for eval_e6c artifact (downloaded by its own watcher)
while [ ! -f "$OUT/longgen_probe_curve.csv" ]; do
  echo "$(date -u) [e7] queued behind eval_e6c" >> "$LOG"
  sleep 300
done
echo "$(date -u) [e7] eval_e6c done -> starting VM + launching s_e7" >> "$LOG"

# Phase 1 — start VM (retry: spot may stockout briefly) + launch
for i in $(seq 1 24); do
  ST=$(vmstatus)
  if [ "$ST" = "RUNNING" ]; then break; fi
  G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null && break
  sleep 300
done
sleep 45
for i in 1 2 3 4 5 6; do launch && break; sleep 60; done
echo "S_E7_LAUNCHED"
echo "$(date -u) [e7] s_e7 launched" >> "$LOG"

# Phase 2 — watch + live-download curves
while true; do
  sleep 300
  R=$(sshc "echo SSHOK; test -f ~/scaled/ALL_DONE && echo DONE || echo RUNNING; \
    ps -eo args | grep -c 'scripts/train.p[y]'; \
    tail -1 ~/repo/runs/s_e7/learning_curve.csv 2>/dev/null | cut -d, -f1,4")
  if echo "$R" | grep -q SSHOK; then
    ST=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p); TSC=$(echo "$R" | sed -n 4p)
    echo "$(date -u) [e7] $ST procs=$NP ts,corr=$TSC" >> "$LOG"
    scpdl '~/repo/runs/s_e7/probe_curve.csv'    "$OUT/s_e7_probe_curve.csv"
    scpdl '~/repo/runs/s_e7/learning_curve.csv' "$OUT/s_e7_learning_curve.csv"
    if [ "$ST" = "DONE" ]; then
      for t in 1 2 3; do G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break; sleep 20; done
      echo "E7_ALL_DONE"
      exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) [e7] no proc -> relaunch (resumes)" >> "$LOG"
      launch
    fi
  else
    S=$(vmstatus)
    echo "$(date -u) [e7] SSH failed vm=$S" >> "$LOG"
    if [ "$S" = "TERMINATED" ] || [ "$S" = "STOPPED" ]; then
      if G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45; launch
      fi
    fi
  fi
done
