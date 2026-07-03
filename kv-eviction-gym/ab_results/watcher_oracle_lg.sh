#!/bin/bash
# Watcher: long-gen oracle bound on kv-chat-v1 (on-demand, no preemption risk).
# Live-downloads the per-example JSONL every poll; ORACLE_LG_DONE → final
# download + STOP VM. Relaunches tmux if the process dies (resume-safe JSONL).
ACC=atissera@udesa.edu.ar
PROJ=proyecto-final-425415
VM=kv-chat-v1
ZONE=us-west1-a
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_oracle_lg.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
scpdl(){ G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap "$VM:$1" "$2" 2>/dev/null; }
echo "================ oracle_lg watcher start $(date -u) ================" >> "$LOG"
FAILS=0
while true; do
  sleep 300
  R=$(sshc "echo SSHOK; test -f ~/scaled/ORACLE_LG_DONE && echo DONE || echo RUNNING; \
    wc -l < ~/repo/runs/oracle_longgen/oracle_results.jsonl 2>/dev/null; \
    ps -eo args | grep -c 'scripts/oracle_eva[l]'")
  if ! echo "$R" | grep -q SSHOK; then
    FAILS=$((FAILS+1)); echo "$(date -u) [orclg] SSH fail x$FAILS" >> "$LOG"
    [ $FAILS -ge 5 ] && { echo "ORACLE_LG_SSH_LOST"; exit 1; }
    continue
  fi
  FAILS=0
  ST=$(echo "$R" | sed -n 2p); NL=$(echo "$R" | sed -n 3p); NP=$(echo "$R" | sed -n 4p)
  echo "$(date -u) [orclg] $ST examples=$NL procs=$NP" >> "$LOG"
  scpdl '~/repo/runs/oracle_longgen/oracle_results.jsonl' "$OUT/oracle_longgen_results.jsonl"
  if [ "$ST" = "DONE" ]; then
    for t in 1 2 3; do G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break; sleep 20; done
    echo "ORACLE_LG_DONE"
    exit 0
  fi
  if [ "$NP" = "0" ]; then
    echo "$(date -u) [orclg] no proc -> relaunch (resumes)" >> "$LOG"
    sshc "tmux kill-session -t orclg 2>/dev/null; tmux new-session -d -s orclg 'bash ~/run_oracle_longgen.sh'"
  fi
done
