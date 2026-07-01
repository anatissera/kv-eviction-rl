#!/bin/bash
# E0 screen watcher — preemption-aware. Waits for the screen driver's ALL_DONE,
# downloads every variant's probe curve, stops the VM, runs screen_verdict.py.
# On spot preemption: restart the VM + relaunch run_screen.sh (per-variant markers
# skip finished variants).
ACC=atissera@udesa.edu.ar
PROJ=tp-final-rl-kv-eviction
VM=kvp-ab
ZONE=asia-southeast1-a
VARIANTS="e0_baseline e0_rich e0_rich_warm e0_attn e0_rich_s4"      # space-separated variant names for the relaunch
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
EXP=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase2-capacity
LOG=$OUT/watcher_screen.log
INTERVAL=300

g(){ gcloud --account=$ACC "$@"; }
sshc(){ g compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
  --ssh-flag="-o ConnectTimeout=30" --command="$1" 2>/dev/null; }
vmstatus(){ g compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" \
  --format="value(status)" 2>/dev/null; }

exec >>"$LOG" 2>&1
echo "================ screen watcher start $(date -u) VM=$VM ================"

DONE=0
while [ $DONE -eq 0 ]; do
  R=$(sshc "echo SSHOK; \
    test -f ~/screen/ALL_DONE && echo ALLDONE || echo RUNNING; \
    ps -eo comm,args | grep 'scripts/train.p[y]' | grep -vc tmux; \
    ls ~/screen/*.done 2>/dev/null | wc -l; \
    tail -2 ~/screen/driver.log 2>/dev/null | tr '\n' '|'")
  if echo "$R" | grep -q SSHOK; then
    STATE=$(echo "$R" | sed -n 2p); NP=$(echo "$R" | sed -n 3p)
    NDONE=$(echo "$R" | sed -n 4p); TAIL=$(echo "$R" | sed -n 5p)
    echo "$(date -u) state=$STATE procs=$NP done=$NDONE :: $TAIL"
    if [ "$STATE" = "ALLDONE" ]; then
      echo "$(date -u) ALL_DONE -> download+stop"
      for v in $VARIANTS; do
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
          $VM:~/repo/runs/$v/probe_curve.csv    "$OUT/${v}_probe_curve.csv" 2>/dev/null
        g compute scp --project=$PROJ --zone=$ZONE --tunnel-through-iap \
          $VM:~/repo/runs/$v/learning_curve.csv "$OUT/${v}_learning_curve.csv" 2>/dev/null
      done
      for try in 1 2 3; do
        g compute instances stop $VM --project=$PROJ --zone=$ZONE --quiet && break
        echo "$(date -u) stop retry $try"; sleep 20
      done
      echo "$(date -u) VM STOPPED"
      DONE=1; break
    fi
    if [ "$NP" = "0" ]; then
      echo "$(date -u) no train proc & not done -> relaunch screen driver"
      sshc "tmux has-session -t screen 2>/dev/null || tmux new-session -d -s screen 'bash ~/run_screen.sh $VARIANTS'"
    fi
  else
    ST=$(vmstatus)
    echo "$(date -u) SSH failed vm_status=$ST"
    if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
      echo "$(date -u) preempted -> restart+relaunch"
      if g compute instances start $VM --project=$PROJ --zone=$ZONE --quiet 2>/dev/null; then
        sleep 30
        sshc "tmux kill-session -t screen 2>/dev/null; tmux new-session -d -s screen 'bash ~/run_screen.sh $VARIANTS'"
      fi
    fi
  fi
  [ $DONE -eq 1 ] && break
  sleep $INTERVAL
done

echo "================ screen verdict $(date -u) ================"
python3 "$EXP/screen_verdict.py" "$OUT" > "$OUT/screen_verdict.txt" 2>&1
cat "$OUT/screen_verdict.txt"
echo "================ SCREEN WATCHER DONE $(date -u) ================"
