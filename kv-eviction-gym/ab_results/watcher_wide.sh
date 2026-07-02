#!/bin/bash
# Watcher for the wide eval on kv-chat-v1: waits for ~/scaled/WIDE_DONE,
# downloads runs/wide_eval/{probe_curve,labels}.csv, STOPS the VM.
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
G(){ gcloud --account=atissera@udesa.edu.ar "$@"; }
sshc(){ G compute ssh kv-chat-v1 --project=proyecto-final-425415 --zone=us-west1-a \
  --tunnel-through-iap --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
FAILS=0
while true; do
  R=$(sshc "echo SSHOK; test -f ~/scaled/WIDE_DONE && echo DONE || echo RUNNING; grep -c 'rc=' ~/scaled/wide_eval.log 2>/dev/null")
  if ! echo "$R" | grep -q SSHOK; then
    FAILS=$((FAILS+1))
    if [ $FAILS -ge 3 ]; then echo "WIDE_EVAL_SSH_LOST x3 — investigate"; exit 1; fi
  else
    FAILS=0
    echo "$(date -u) $(echo "$R" | sed -n 2p) arms_finished=$(echo "$R" | sed -n 3p)" >> "$OUT/watcher_wide.log"
    if echo "$R" | sed -n 2p | grep -q DONE; then
      for f in probe_curve.csv labels.csv; do
        G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
          kv-chat-v1:~/repo/runs/wide_eval/$f "$OUT/wide_eval_$f" 2>/dev/null
      done
      G compute instances stop kv-chat-v1 --project=proyecto-final-425415 --zone=us-west1-a --quiet 2>/dev/null && echo VM_STOPPED
      echo WIDE_EVAL_DONE
      exit 0
    fi
  fi
  sleep 300
done
