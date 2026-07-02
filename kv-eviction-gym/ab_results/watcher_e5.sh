#!/bin/bash
# Watcher for the E5 chain (trace_gen → s_golden BC+RL) on kvp-ab (SPOT).
# On preemption: restart VM + relaunch ~/run_e5.sh (both stages resume).
# On ~/scaled/ALL_DONE: download s_golden curves + BC log excerpt, STOP VM.
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_e5.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }

echo "================ e5 watcher start $(date -u) ================" >> "$LOG"
while true; do
  R=$(sshc "echo SSHOK; test -f ~/scaled/ALL_DONE && echo DONE || echo RUNNING; \
    test -f ~/scaled/TRACES_DONE && echo TRACES_OK || ls ~/repo/traces 2>/dev/null | wc -l; \
    pgrep -fc 'trace_gen|scripts/train.p[y]' || true; \
    tail -1 ~/repo/runs/s_golden/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    grep -o 'match_oracle=[0-9.]*' ~/scaled/s_golden.log 2>/dev/null | tail -1")
  if echo "$R" | grep -q SSHOK; then
    STATE=$(echo "$R" | sed -n 2p); TR=$(echo "$R" | sed -n 3p); NP=$(echo "$R" | sed -n 4p)
    TS=$(echo "$R" | sed -n 5p); MO=$(echo "$R" | sed -n 6p)
    echo "$(date -u) state=$STATE traces=$TR procs=$NP golden_ts=$TS $MO" >> "$LOG"
    if [ "$STATE" = "DONE" ]; then
      G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        $VM:'~/repo/runs/s_golden/probe_curve.csv' "$OUT/s_golden_probe_curve.csv" 2>/dev/null
      G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        $VM:'~/repo/runs/s_golden/learning_curve.csv' "$OUT/s_golden_learning_curve.csv" 2>/dev/null
      G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        $VM:'~/scaled/s_golden.log' "$OUT/s_golden_run.log" 2>/dev/null
      for try in 1 2 3; do
        G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break
        sleep 20
      done
      echo "E5_ALL_DONE"
      exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) no proc & not done -> relaunch e5 chain (resumes)" >> "$LOG"
      sshc "tmux kill-session -t e5 2>/dev/null; tmux new-session -d -s e5 'bash ~/run_e5.sh'"
    fi
  else
    ST=$(vmstatus)
    echo "$(date -u) SSH failed vm_status=$ST" >> "$LOG"
    if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
      if G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45
        sshc "tmux kill-session -t e5 2>/dev/null; tmux new-session -d -s e5 'bash ~/run_e5.sh'"
      fi
    fi
  fi
  sleep 300
done
