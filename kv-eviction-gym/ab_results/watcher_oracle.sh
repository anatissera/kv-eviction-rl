#!/bin/bash
# Watcher for the oracle-gap eval on kv-chat-v1: waits for ~/scaled/ORACLE_DONE,
# downloads oracle_results.jsonl + log, STOPS the VM. Relaunches tmux if the
# process dies (resume-safe: the script skips done examples).
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
G(){ gcloud --account=atissera@udesa.edu.ar "$@"; }
sshc(){ G compute ssh kv-chat-v1 --project=proyecto-final-425415 --zone=us-west1-a \
  --tunnel-through-iap --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
FAILS=0
while true; do
  R=$(sshc "echo SSHOK; test -f ~/scaled/ORACLE_DONE && echo DONE || echo RUNNING; \
    wc -l < ~/repo/runs/oracle_eval/oracle_results.jsonl 2>/dev/null; \
    pgrep -fc 'scripts/oracle_eval' || true")
  if ! echo "$R" | grep -q SSHOK; then
    FAILS=$((FAILS+1))
    if [ $FAILS -ge 3 ]; then echo "ORACLE_SSH_LOST x3 — investigate"; exit 1; fi
  else
    FAILS=0
    STATE=$(echo "$R" | sed -n 2p); NLINES=$(echo "$R" | sed -n 3p); NP=$(echo "$R" | sed -n 4p)
    echo "$(date -u) state=$STATE examples_done=$NLINES procs=$NP" >> "$OUT/watcher_oracle.log"
    if [ "$STATE" = "DONE" ]; then
      G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
        kv-chat-v1:~/repo/runs/oracle_eval/oracle_results.jsonl "$OUT/oracle_results.jsonl" 2>/dev/null
      G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
        kv-chat-v1:'~/scaled/oracle.log' "$OUT/oracle_run.log" 2>/dev/null
      G compute instances stop kv-chat-v1 --project=proyecto-final-425415 --zone=us-west1-a --quiet 2>/dev/null && echo VM_STOPPED
      echo ORACLE_DONE
      exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) proc dead, not done -> relaunch (resumes)" >> "$OUT/watcher_oracle.log"
      sshc "tmux kill-session -t oracle 2>/dev/null; tmux new-session -d -s oracle 'bash ~/run_oracle.sh'"
    fi
  fi
  sleep 300
done
