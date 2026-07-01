#!/bin/bash
# Autonomous A/B watcher: wait for both runs to finish (or crash/stall),
# download probe+learning curves, stop each VM as soon as ITS run ends, then
# run the comparison. Resilient to: transient SSH failures (SSHOK sentinel),
# the gcloud active-account flip (explicit --account on every call), and hung
# runs (stall detector forces a stop after 30 min of no progress).

ACC=atissera@udesa.edu.ar
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
CMP=$OUT/compare_ab.py
LOG=$OUT/watcher.log
INTERVAL=300          # poll every 5 min
STALL_LIMIT=6         # 6 * 5min = 30 min with no timestep progress => assume hung

# control:   kv-chat-v1  / proyecto-final-425415 / us-west1-a / Alex's home (sudo)
# treatment: kvp-s4      / tp-final-rl-kv-eviction / us-west4-a / our home
CTRL_DIR=/home/alexanderbodner/kv-eviction-gym/runs/20260629_201229
TREAT_DIR=/home/anatissera/repo/runs/chat_full_s4_kl

g() { gcloud --account=$ACC "$@"; }
sshc() { # name project zone command
  g compute ssh "$1" --project="$2" --zone="$3" --tunnel-through-iap \
    --ssh-flag="-o ConnectTimeout=30" --command="$4" 2>/dev/null
}

exec >>"$LOG" 2>&1
echo "================ watcher start $(date -u) ================"

CTRL_DONE=0; TREAT_DONE=0
CTRL_PREV=""; CTRL_STALL=0
TREAT_PREV=""; TREAT_STALL=0

while [ $CTRL_DONE -eq 0 ] || [ $TREAT_DONE -eq 0 ]; do

  # ---------------- CONTROL ----------------
  if [ $CTRL_DONE -eq 0 ]; then
    # [2] trick: regex 'chat_full_v[2]' matches the real cmdline but NOT this ssh
    # command (which contains the literal '[2]'). Exclude tmux so the session
    # wrapper (whose cmdline mirrors the inner command) isn't counted as alive.
    R=$(sshc kv-chat-v1 proyecto-final-425415 us-west1-a \
      "echo SSHOK; ps -eo comm,args | grep 'chat_full_v[2].yaml' | grep -vc tmux; sudo tail -1 $CTRL_DIR/learning_curve.csv 2>/dev/null | cut -d, -f1")
    if echo "$R" | grep -q SSHOK; then
      NP=$(echo "$R" | sed -n 2p); TS=$(echo "$R" | sed -n 3p)
      echo "$(date -u) control: procs=$NP ts=$TS"
      if [ "$TS" = "$CTRL_PREV" ] && [ -n "$TS" ]; then CTRL_STALL=$((CTRL_STALL+1)); else CTRL_STALL=0; fi
      CTRL_PREV=$TS
      if [ "$NP" = "0" ] || [ $CTRL_STALL -ge $STALL_LIMIT ]; then
        echo "$(date -u) control ENDED (procs=$NP stall=$CTRL_STALL) -> download+stop"
        sshc kv-chat-v1 proyecto-final-425415 us-west1-a \
          "sudo cp $CTRL_DIR/probe_curve.csv /tmp/ctrl_probe.csv 2>/dev/null; sudo cp $CTRL_DIR/learning_curve.csv /tmp/ctrl_lc.csv 2>/dev/null; sudo chmod 644 /tmp/ctrl_*.csv 2>/dev/null"
        g compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap kv-chat-v1:/tmp/ctrl_probe.csv "$OUT/control_probe_curve.csv" 2>/dev/null
        g compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap kv-chat-v1:/tmp/ctrl_lc.csv "$OUT/control_learning_curve.csv" 2>/dev/null
        for try in 1 2 3; do
          g compute instances stop kv-chat-v1 --project=proyecto-final-425415 --zone=us-west1-a --quiet && break
          echo "$(date -u) control stop retry $try"; sleep 20
        done
        echo "$(date -u) control STOPPED"
        CTRL_DONE=1
      fi
    else
      echo "$(date -u) control: SSH failed (transient), will retry"
    fi
  fi

  # ---------------- TREATMENT ----------------
  if [ $TREAT_DONE -eq 0 ]; then
    R=$(sshc kvp-s4 tp-final-rl-kv-eviction us-west4-a \
      "echo SSHOK; ps -eo comm,args | grep 'chat_full_s4_k[l].yaml' | grep -vc tmux; tail -1 $TREAT_DIR/learning_curve.csv 2>/dev/null | cut -d, -f1")
    if echo "$R" | grep -q SSHOK; then
      NP=$(echo "$R" | sed -n 2p); TS=$(echo "$R" | sed -n 3p)
      echo "$(date -u) treatment: procs=$NP ts=$TS"
      if [ "$TS" = "$TREAT_PREV" ] && [ -n "$TS" ]; then TREAT_STALL=$((TREAT_STALL+1)); else TREAT_STALL=0; fi
      TREAT_PREV=$TS
      if [ "$NP" = "0" ] || [ $TREAT_STALL -ge $STALL_LIMIT ]; then
        echo "$(date -u) treatment ENDED (procs=$NP stall=$TREAT_STALL) -> download+stop"
        g compute scp --project=tp-final-rl-kv-eviction --zone=us-west4-a --tunnel-through-iap kvp-s4:$TREAT_DIR/probe_curve.csv "$OUT/treat_probe_curve.csv" 2>/dev/null
        g compute scp --project=tp-final-rl-kv-eviction --zone=us-west4-a --tunnel-through-iap kvp-s4:$TREAT_DIR/learning_curve.csv "$OUT/treat_learning_curve.csv" 2>/dev/null
        for try in 1 2 3; do
          g compute instances stop kvp-s4 --project=tp-final-rl-kv-eviction --zone=us-west4-a --quiet && break
          echo "$(date -u) treatment stop retry $try"; sleep 20
        done
        echo "$(date -u) treatment STOPPED"
        TREAT_DONE=1
      fi
    else
      echo "$(date -u) treatment: SSH failed (transient), will retry"
    fi
  fi

  [ $CTRL_DONE -eq 1 ] && [ $TREAT_DONE -eq 1 ] && break
  sleep $INTERVAL
done

echo "================ both ended, comparing $(date -u) ================"
python3 "$CMP" "$OUT/control_probe_curve.csv" "$OUT/treat_probe_curve.csv" "$OUT/ab_retention.png" > "$OUT/comparison.txt" 2>&1
cat "$OUT/comparison.txt"
echo "================ WATCHER DONE $(date -u) ================"
