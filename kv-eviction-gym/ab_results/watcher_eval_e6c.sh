#!/bin/bash
# Watcher: s_rich eval on the filtered long-gen wide slice (kvp-ab, SPOT).
# Preemption → restart + relaunch (wide_eval anchors are cached in
# runs/wide_eval_longgen/probe_anchors.pkl → resume is cheap).
# EVAL_E6C_DONE → download results, STOP VM.
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_eval_e6c.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }
echo "================ eval_e6c watcher start $(date -u) ================" >> "$LOG"
while true; do
  sleep 300
  R=$(sshc "echo SSHOK; test -f ~/scaled/EVAL_E6C_DONE && echo DONE || echo RUNNING; \
    pgrep -fc 'wide_eval' || true; tail -1 ~/scaled/eval_e6c.log 2>/dev/null | head -c 120")
  if echo "$R" | grep -q SSHOK; then
    ST=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p)
    echo "$(date -u) [eval_e6c] $ST procs=$NP :: $(echo "$R" | sed -n 4p)" >> "$LOG"
    if [ "$ST" = "DONE" ]; then
      G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        kvp-ab:'~/repo/runs/wide_eval_longgen/probe_curve.csv' "$OUT/longgen_probe_curve.csv" 2>/dev/null
      G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        kvp-ab:'~/repo/runs/wide_eval_longgen/labels.csv' "$OUT/longgen_labels.csv" 2>/dev/null
      for t in 1 2 3; do G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break; sleep 20; done
      echo "EVAL_E6C_DONE"
      exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) [eval_e6c] no proc -> relaunch (anchors cached)" >> "$LOG"
      sshc "tmux kill-session -t evale6c 2>/dev/null; tmux new-session -d -s evale6c 'bash ~/run_eval_e6c.sh'"
    fi
  else
    S=$(vmstatus)
    echo "$(date -u) [eval_e6c] SSH failed vm=$S" >> "$LOG"
    if [ "$S" = "TERMINATED" ] || [ "$S" = "STOPPED" ]; then
      if G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45
        sshc "tmux kill-session -t evale6c 2>/dev/null; tmux new-session -d -s evale6c 'bash ~/run_eval_e6c.sh'"
      fi
    fi
  fi
done
