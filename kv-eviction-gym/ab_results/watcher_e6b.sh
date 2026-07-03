#!/bin/bash
# Watcher for E6b (s_long192) on kvp-ab with AUTO-ABORT:
#   - if after ≥45 min the training curve shows correctness_rate == 0 on every
#     rollout (≥5 rollouts), the regime is dead (E6 v1 failure mode) → kill the
#     run, stop the VM, report ABORT.
#   - preemption → restart + resume; proc death → relaunch.
#   - ALL_DONE → download curves, stop VM, report DONE.
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
SPEC="configs/e6b_long192.yaml:s_long192"
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/watcher_e6b.log
G(){ gcloud --account=$ACC "$@"; }
sshc(){ G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }
scpdl(){ G compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap "$VM:$1" "$2" 2>/dev/null; }
stopvm(){ for t in 1 2 3; do G compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break; sleep 20; done; }

T0=$(date +%s)
echo "================ e6b watcher start $(date -u) ================" >> "$LOG"
while true; do
  sleep 300
  R=$(sshc "echo SSHOK; test -f ~/scaled/ALL_DONE && echo DONE || echo RUNNING; \
    ps -eo args | grep -c 'scripts/train.p[y]'; \
    awk -F, 'NR>1{n++; if(\$4+0>0) pos++} END{print n\" \"pos+0}' ~/repo/runs/s_long192/learning_curve.csv 2>/dev/null")
  NOW=$(date +%s); MIN=$(( (NOW-T0)/60 ))
  if echo "$R" | grep -q SSHOK; then
    ST=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p); CURVE=$(echo "$R" | sed -n 4p)
    NROLL=$(echo "$CURVE" | cut -d' ' -f1); NPOS=$(echo "$CURVE" | cut -d' ' -f2)
    echo "$(date -u) [e6b] $ST procs=$NP rollouts=$NROLL pos_corr=$NPOS elapsed=${MIN}m" >> "$LOG"
    if [ "$ST" = "DONE" ]; then
      scpdl '~/repo/runs/s_long192/probe_curve.csv'    "$OUT/s_long192_probe_curve.csv"
      scpdl '~/repo/runs/s_long192/learning_curve.csv' "$OUT/s_long192_learning_curve.csv"
      stopvm; echo "E6B_DONE"; exit 0
    fi
    # auto-abort: ≥45 min, ≥5 rollouts logged, zero rollouts with any correct episode
    if [ "$MIN" -ge 45 ] && [ -n "$NROLL" ] && [ "$NROLL" -ge 5 ] && [ "$NPOS" = "0" ]; then
      echo "$(date -u) [e6b] ABORT: $NROLL rollouts, correctness all zero — dead regime" >> "$LOG"
      sshc "tmux kill-session -t e6b 2>/dev/null; pkill -f 'scripts/train.py.*s_long192'"
      scpdl '~/repo/runs/s_long192/learning_curve.csv' "$OUT/s_long192_learning_curve.csv"
      stopvm; echo "E6B_ABORTED_DEAD_REGIME"; exit 0
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) [e6b] no proc -> relaunch (resumes)" >> "$LOG"
      sshc "tmux kill-session -t e6b 2>/dev/null; tmux new-session -d -s e6b 'bash ~/run_scaled.sh $SPEC'"
    fi
  else
    S=$(vmstatus)
    echo "$(date -u) [e6b] SSH failed vm=$S" >> "$LOG"
    if [ "$S" = "TERMINATED" ] || [ "$S" = "STOPPED" ]; then
      if G compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45
        sshc "tmux kill-session -t e6b 2>/dev/null; tmux new-session -d -s e6b 'bash ~/run_scaled.sh $SPEC'"
      fi
    fi
  fi
done
