#!/bin/bash
# Watcher for s_attn (E4 attention @ scale) on kv-chat-v1 (Alex's project,
# on-demand L4 → preemption unlikely, but relaunch logic kept anyway).
# Waits for ~/scaled/ALL_DONE, downloads curves, STOPS the VM, runs compare.py
# against s_rich (+ s_warm when present).
ACC=atissera@udesa.edu.ar
PROJ=proyecto-final-425415
VM=kv-chat-v1
ZONE=us-west1-a
SPEC="configs/e4_attn.yaml:s_attn"
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
EXP=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase2-capacity
LOG=$OUT/watcher_attn.log
INTERVAL=300

g(){ gcloud --account=$ACC "$@"; }
sshc(){ g compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ g compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }

exec >>"$LOG" 2>&1
echo "================ attn watcher start $(date -u) ================"

DONE=0
while [ $DONE -eq 0 ]; do
  R=$(sshc "echo SSHOK; \
    test -f ~/scaled/ALL_DONE && echo ALLDONE || echo RUNNING; \
    ps -eo comm,args | grep 'scripts/train.p[y]' | grep -vc tmux; \
    tail -1 ~/repo/runs/s_attn/learning_curve.csv 2>/dev/null | cut -d, -f1; \
    tail -1 ~/scaled/driver.log 2>/dev/null")
  if echo "$R" | grep -q SSHOK; then
    STATE=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p)
    ATS=$(echo "$R" | sed -n 4p); TAIL=$(echo "$R" | sed -n 5p)
    echo "$(date -u) state=$STATE procs=$NP attn_ts=$ATS :: $TAIL"
    if [ "$STATE" = "ALLDONE" ]; then
      echo "$(date -u) ALL_DONE -> download+stop"
      g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        $VM:~/repo/runs/s_attn/probe_curve.csv    "$OUT/s_attn_probe_curve.csv" 2>/dev/null
      g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
        $VM:~/repo/runs/s_attn/learning_curve.csv "$OUT/s_attn_learning_curve.csv" 2>/dev/null
      for try in 1 2 3; do
        g compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break
        echo "$(date -u) stop retry $try"; sleep 20
      done
      echo "$(date -u) VM STOPPED"
      DONE=1; break
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) no train proc & not done -> relaunch driver (resumes from ckpt)"
      sshc "tmux has-session -t scaled 2>/dev/null || tmux new-session -d -s scaled 'bash ~/run_scaled_attn.sh $SPEC'"
    fi
  else
    ST=$(vmstatus)
    echo "$(date -u) SSH failed vm_status=$ST"
    if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
      echo "$(date -u) VM stopped unexpectedly -> restart+relaunch (resume from ckpt)"
      if g compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 45
        sshc "tmux kill-session -t scaled 2>/dev/null; tmux new-session -d -s scaled 'bash ~/run_scaled_attn.sh $SPEC'"
      fi
    fi
  fi
  [ $DONE -eq 1 ] && break
  sleep $INTERVAL
done

echo "================ attn comparison $(date -u) ================"
ARGS="rich=$OUT/s_rich_probe_curve.csv"
[ -f "$OUT/s_warm_probe_curve.csv" ] && ARGS="$ARGS warm_start=$OUT/s_warm_probe_curve.csv"
ARGS="$ARGS attention=$OUT/s_attn_probe_curve.csv"
python3 "$EXP/compare.py" $ARGS --out "$OUT/attn_retention.png" > "$OUT/attn_comparison.txt" 2>&1
cat "$OUT/attn_comparison.txt"
echo "================ ATTN WATCHER DONE $(date -u) ================"
