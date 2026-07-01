#!/bin/bash
# Clean A/B watcher — single SPOT VM, sequential control->treatment, OUR home dir.
# Preemption-aware: if the spot VM gets STOPPED (preempted), tries to restart it
# and re-launch the restart-safe driver (which skips already-finished arms via
# marker files). When ~/ab_clean/ALL_DONE appears: download both arms' curves,
# stop the VM, run the comparison. Resilient to the gcloud account flip.
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
CMP=$OUT/compare_ab.py
LOG=$OUT/watcher_clean.log
INTERVAL=300
CTRL_DIR=/home/anatissera/repo/runs/ab_control_s0
TREAT_DIR=/home/anatissera/repo/runs/ab_treat_s0

g(){ gcloud --account=$ACC "$@"; }
sshc(){ g compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ g compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }

exec >>"$LOG" 2>&1
echo "================ clean watcher start $(date -u) ================"

DONE=0; SSHFAIL=0
while [ $DONE -eq 0 ]; do
  R=$(sshc "echo SSHOK; \
    test -f ~/ab_clean/ALL_DONE && echo ALLDONE || echo RUNNING; \
    ps -eo comm,args | grep 'scripts/train.p[y]' | grep -vc tmux; \
    tail -1 ~/repo/runs/ab_control_s0/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    tail -1 ~/repo/runs/ab_treat_s0/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    ls ~/ab_clean/control.done ~/ab_clean/treat.done 2>/dev/null | tr '\n' ' '")
  if echo "$R" | grep -q SSHOK; then
    SSHFAIL=0
    STATE=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p)
    CTS=$(echo "$R" | sed -n 4p); TTS=$(echo "$R" | sed -n 5p); MK=$(echo "$R" | sed -n 6p)
    echo "$(date -u) state=$STATE procs=$NP ctrl_ts=$CTS treat_ts=$TTS markers=[$MK]"
    if [ "$STATE" = "ALLDONE" ]; then
      echo "$(date -u) ALL_DONE -> download+stop"
      for pair in "$CTRL_DIR:clean_control" "$TREAT_DIR:clean_treat"; do
        D=${pair%%:*}; P=${pair##*:}
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap $VM:$D/probe_curve.csv    "$OUT/${P}_probe_curve.csv" 2>/dev/null
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap $VM:$D/learning_curve.csv "$OUT/${P}_learning_curve.csv" 2>/dev/null
      done
      for try in 1 2 3; do
        g compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break
        echo "$(date -u) stop retry $try"; sleep 20
      done
      echo "$(date -u) VM STOPPED"
      DONE=1; break
    fi
    # alive but no train proc AND not all-done -> driver between arms or crashed; re-kick driver
    if [ "$NP" = "0" ]; then
      echo "$(date -u) no train proc & not done -> (re)launch driver"
      sshc "tmux has-session -t ab 2>/dev/null || tmux new-session -d -s ab 'bash ~/ab_driver.sh'"
    fi
  else
    SSHFAIL=$((SSHFAIL+1))
    ST=$(vmstatus)
    echo "$(date -u) SSH failed ($SSHFAIL) vm_status=$ST"
    if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
      echo "$(date -u) VM preempted -> attempt restart"
      if g compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        echo "$(date -u) restart OK -> relaunch driver (skips finished arms)"
        sleep 30
        sshc "tmux kill-session -t ab 2>/dev/null; tmux new-session -d -s ab 'bash ~/ab_driver.sh'"
      else
        echo "$(date -u) restart FAILED (stockout) -> will retry next poll"
      fi
    fi
  fi
  [ $DONE -eq 1 ] && break
  sleep $INTERVAL
done

echo "================ comparing $(date -u) ================"
python3 "$CMP" "$OUT/clean_control_probe_curve.csv" "$OUT/clean_treat_probe_curve.csv" \
  "$OUT/ab_retention_clean.png" > "$OUT/comparison_clean.txt" 2>&1
cat "$OUT/comparison_clean.txt"
echo "================ CLEAN WATCHER DONE $(date -u) ================"
