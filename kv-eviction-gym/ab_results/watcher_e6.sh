#!/bin/bash
# Chained watcher on kvp-ab (SPOT):
#   Phase 1: wait for WIDE2_DONE (s_golden/s_rich paired wide eval) → download
#            wide2 results + s_golden checkpoint → LAUNCH s_long (E6) directly
#            (no stop/start cycle).
#   Phase 2: watch s_long like watcher_scaled (preemption → restart+resume;
#            proc death → relaunch; ALL_DONE → download curves, STOP VM).
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
SPEC="configs/e6_long.yaml:s_long"
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_e6.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }
scpdl(){ G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap "$VM:$1" "$2" 2>/dev/null; }

echo "================ e6 watcher start $(date -u) ================" >> "$LOG"

# ---- Phase 1: wide2 ----
while true; do
  R=$(sshc "echo SSHOK; test -f ~/scaled/WIDE2_DONE && echo DONE || echo RUNNING; tail -1 ~/scaled/wide2.log 2>/dev/null | head -c 120")
  if echo "$R" | grep -q SSHOK; then
    ST=$(echo "$R" | sed -n 2p)
    echo "$(date -u) [wide2] $ST :: $(echo "$R" | sed -n 3p)" >> "$LOG"
    [ "$ST" = "DONE" ] && break
  else
    S=$(vmstatus)
    echo "$(date -u) [wide2] SSH failed vm=$S" >> "$LOG"
    if [ "$S" = "TERMINATED" ]; then
      G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null && sleep 45 && \
        sshc "tmux has-session -t wide2 2>/dev/null || tmux new-session -d -s wide2 'bash ~/run_wide2.sh'"
    fi
  fi
  sleep 300
done
scpdl '~/repo/runs/wide_eval_kvpab/probe_curve.csv' "$OUT/wide2_probe_curve.csv"
scpdl '~/repo/runs/wide_eval_kvpab/labels.csv'      "$OUT/wide2_labels.csv"
scpdl '~/repo/runs/s_golden/final_model.zip'        "$OUT/checkpoints/s_golden_final_model.zip"
echo "WIDE2_RESULTS_DOWNLOADED"

# ---- Launch E6 s_long ----
sshc "rm -f ~/scaled/ALL_DONE ~/scaled/s_long.done; tmux kill-session -t e6 2>/dev/null; \
  tmux new-session -d -s e6 'bash ~/run_scaled.sh $SPEC'"
echo "S_LONG_LAUNCHED"
echo "$(date -u) s_long launched" >> "$LOG"

# ---- Phase 2: s_long ----
while true; do
  sleep 300
  R=$(sshc "echo SSHOK; test -f ~/scaled/ALL_DONE && echo DONE || echo RUNNING; \
    ps -eo args | grep -c 'scripts/train.p[y]'; \
    tail -1 ~/repo/runs/s_long/learning_curve.csv 2>/dev/null | cut -d, -f1")
  if echo "$R" | grep -q SSHOK; then
    ST=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p); TS=$(echo "$R" | sed -n 4p)
    echo "$(date -u) [s_long] $ST procs=$NP ts=$TS" >> "$LOG"
    if [ "$ST" = "DONE" ]; then
      scpdl '~/repo/runs/s_long/probe_curve.csv'    "$OUT/s_long_probe_curve.csv"
      scpdl '~/repo/runs/s_long/learning_curve.csv' "$OUT/s_long_learning_curve.csv"
      for try in 1 2 3; do
        G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break; sleep 20
      done
      echo "E6_ALL_DONE"
      exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) [s_long] no proc -> relaunch (resumes from ckpt)" >> "$LOG"
      sshc "tmux kill-session -t e6 2>/dev/null; tmux new-session -d -s e6 'bash ~/run_scaled.sh $SPEC'"
    fi
  else
    S=$(vmstatus)
    echo "$(date -u) [s_long] SSH failed vm=$S" >> "$LOG"
    if [ "$S" = "TERMINATED" ] || [ "$S" = "STOPPED" ]; then
      if G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45
        sshc "tmux kill-session -t e6 2>/dev/null; tmux new-session -d -s e6 'bash ~/run_scaled.sh $SPEC'"
      fi
    fi
  fi
done
