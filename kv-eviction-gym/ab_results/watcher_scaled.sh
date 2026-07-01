#!/bin/bash
# Scaled-runs watcher — preemption-aware, checkpoint-resume aware. Waits for the
# scaled driver's ALL_DONE (s_rich then s_warm), downloads both arms' curves, stops
# the VM, runs the multi-arm comparison (incl. the screen baseline for reference).
# On spot preemption: restart the VM + relaunch run_scaled.sh (which RESUMES each
# arm from its latest checkpoint).
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
SPEC="configs/e1_rich.yaml:s_rich configs/e3_warm.yaml:s_warm"
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
EXP=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase2-capacity
LOG=$OUT/watcher_scaled.log
INTERVAL=300

g(){ gcloud --account=$ACC "$@"; }
sshc(){ g compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ g compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }

exec >>"$LOG" 2>&1
echo "================ scaled watcher start $(date -u) ================"

DONE=0
while [ $DONE -eq 0 ]; do
  R=$(sshc "echo SSHOK; \
    test -f ~/scaled/ALL_DONE && echo ALLDONE || echo RUNNING; \
    ps -eo comm,args | grep 'scripts/train.p[y]' | grep -vc tmux; \
    tail -1 ~/repo/runs/s_rich/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    tail -1 ~/repo/runs/s_warm/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    tail -1 ~/scaled/driver.log 2>/dev/null")
  if echo "$R" | grep -q SSHOK; then
    STATE=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p)
    RTS=$(echo "$R" | sed -n 4p); WTS=$(echo "$R" | sed -n 5p); TAIL=$(echo "$R" | sed -n 6p)
    echo "$(date -u) state=$STATE procs=$NP rich_ts=$RTS warm_ts=$WTS :: $TAIL"
    if [ "$STATE" = "ALLDONE" ]; then
      echo "$(date -u) ALL_DONE -> download+stop"
      for r in s_rich s_warm; do
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
          $VM:~/repo/runs/$r/probe_curve.csv    "$OUT/${r}_probe_curve.csv" 2>/dev/null
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
          $VM:~/repo/runs/$r/learning_curve.csv "$OUT/${r}_learning_curve.csv" 2>/dev/null
      done
      for try in 1 2 3; do
        g compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break
        echo "$(date -u) stop retry $try"; sleep 20
      done
      echo "$(date -u) VM STOPPED"
      DONE=1; break
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) no train proc & not done -> relaunch scaled driver (resumes)"
      sshc "tmux has-session -t scaled 2>/dev/null || tmux new-session -d -s scaled 'bash ~/run_scaled.sh $SPEC'"
    fi
  else
    ST=$(vmstatus)
    echo "$(date -u) SSH failed vm_status=$ST"
    if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
      echo "$(date -u) preempted -> restart+relaunch (resume from checkpoints)"
      if g compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 30
        sshc "tmux kill-session -t scaled 2>/dev/null; tmux new-session -d -s scaled 'bash ~/run_scaled.sh $SPEC'"
      fi
    fi
  fi
  [ $DONE -eq 1 ] && break
  sleep $INTERVAL
done

echo "================ scaled comparison $(date -u) ================"
python3 "$EXP/compare.py" \
  "rich=$OUT/s_rich_probe_curve.csv" \
  "warm_start=$OUT/s_warm_probe_curve.csv" \
  --out "$OUT/scaled_retention.png" > "$OUT/scaled_comparison.txt" 2>&1
cat "$OUT/scaled_comparison.txt"
echo "================ SCALED WATCHER DONE $(date -u) ================"
