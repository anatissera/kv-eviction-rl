#!/bin/bash
# Phase-3 keeper: cron every 10 min. Keeps the 3 dataset-causality experiments
# alive (auto-relaunch on preemption / proc death), live-downloads curves+results
# into experiments/phase3-dataset-causality/data/. NEVER stops a VM (user wants
# them left on). flock guards against overlap.
PH=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase3-dataset-causality
DATA=$PH/data
LOG=$PH/keeper.log
exec 9>"$PH/.keeper.lock"; flock -n 9 || exit 0
G(){ gcloud --account=atissera@udesa.edu.ar "$@"; }

# --- E10 seed0 on kv-none-v2 (on-demand L4, IAP) ---
kvnone(){
  local ST=$(G compute instances describe kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --format="value(status)" 2>/dev/null)
  [ "$ST" = "RUNNING" ] || { G compute instances start kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --quiet 2>/dev/null; echo "$(date -u) kv-none-v2 start" >>"$LOG"; return; }
  local R=$(G compute ssh kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="pgrep -fc 'python scripts/train.py'; test -f ~/repo/runs/s_e10/final_model.zip && echo DONE" 2>/dev/null)
  local NP=$(echo "$R"|sed -n 1p)
  echo "$R"|grep -q DONE && return
  G compute scp --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap kv-none-v2:'~/repo/runs/s_e10/probe_curve.csv' "$DATA/e10_seed0_probe.csv" 2>/dev/null
  G compute scp --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap kv-none-v2:'~/repo/runs/s_e10/learning_curve.csv' "$DATA/e10_seed0_learning.csv" 2>/dev/null
  if [ "$NP" = "0" ]; then
    G compute ssh kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="cd ~/repo && tmux kill-session -t e10 2>/dev/null; tmux new-session -d -s e10 'source .venv/bin/activate; export PYTHONPATH=~/repo/src; export MALLOC_MMAP_THRESHOLD_=1048576; python scripts/train.py --config configs/e10_passkey_rl.yaml --run-name s_e10 >> ~/e10.log 2>&1'" 2>/dev/null
    echo "$(date -u) kv-none-v2 relaunch E10 seed0" >>"$LOG"
  fi
}

# --- E10 seed1 on simcot-t4 (on-demand T4, DIRECT ssh, no IAP) ---
simcot(){
  local ST=$(G compute instances describe simcot-t4 --project=tp-final-nlp --zone=us-central1-a --format="value(status)" 2>/dev/null)
  [ "$ST" = "RUNNING" ] || { G compute instances start simcot-t4 --project=tp-final-nlp --zone=us-central1-a --quiet 2>/dev/null; echo "$(date -u) simcot start" >>"$LOG"; return; }
  local R=$(G compute ssh simcot-t4 --project=tp-final-nlp --zone=us-central1-a --command="pgrep -fc 'python scripts/train.py'; test -f ~/repo/runs/s_e10_seed1/final_model.zip && echo DONE" 2>/dev/null | grep -vE 'Warning|numpy|tcp|iap|Recommend|troubleshoot')
  local NP=$(echo "$R"|sed -n 1p)
  echo "$R"|grep -q DONE && return
  G compute scp --project=tp-final-nlp --zone=us-central1-a simcot-t4:'~/repo/runs/s_e10_seed1/probe_curve.csv' "$DATA/e10_seed1_probe.csv" 2>/dev/null
  G compute scp --project=tp-final-nlp --zone=us-central1-a simcot-t4:'~/repo/runs/s_e10_seed1/learning_curve.csv' "$DATA/e10_seed1_learning.csv" 2>/dev/null
  if [ "$NP" = "0" ]; then
    G compute ssh simcot-t4 --project=tp-final-nlp --zone=us-central1-a --command="cd ~/repo && tmux kill-session -t e10 2>/dev/null; tmux new-session -d -s e10 'bash ~/run_e10.sh'" 2>/dev/null
    echo "$(date -u) simcot relaunch E10 seed1" >>"$LOG"
  fi
}

# --- passkey_ranker on kvp-ab (SPOT L4, IAP, revive on preempt) ---
kvpab(){
  local ST=$(G compute instances describe kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --format="value(status)" 2>/dev/null)
  if [ "$ST" != "RUNNING" ]; then
    G compute instances start kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --quiet 2>/dev/null && \
    { sleep 40; G compute ssh kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=30" --command="cd ~/repo && test -f runs/passkey_ranker_v2/summary.json || tmux new-session -d -s pk 'source .venv/bin/activate; export PYTHONPATH=~/repo/src; export MALLOC_MMAP_THRESHOLD_=1048576; python scripts/passkey_ranker.py --n 160 --n-train 120 --budget 176 --out-dir runs/passkey_ranker_v2 >> ~/pkranker2.log 2>&1'" 2>/dev/null; }
    echo "$(date -u) kvp-ab revive+relaunch pk" >>"$LOG"; return
  fi
  local R=$(G compute ssh kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="pgrep -fc passkey_ranker; test -f ~/repo/runs/passkey_ranker_v2/summary.json && echo DONE" 2>/dev/null)
  local NP=$(echo "$R"|sed -n 1p)
  if echo "$R"|grep -q DONE; then
    G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap kvp-ab:'~/repo/runs/passkey_ranker_v2/summary.json' "$DATA/passkey_ranker_summary.json" 2>/dev/null
    G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap kvp-ab:'~/pkranker2.log' "$DATA/passkey_ranker.log" 2>/dev/null
    return
  fi
  if [ "$NP" = "0" ]; then
    G compute ssh kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="cd ~/repo && tmux new-session -d -s pk 'source .venv/bin/activate; export PYTHONPATH=~/repo/src; export MALLOC_MMAP_THRESHOLD_=1048576; python scripts/passkey_ranker.py --n 160 --n-train 120 --budget 176 --out-dir runs/passkey_ranker_v2 >> ~/pkranker2.log 2>&1'" 2>/dev/null
    echo "$(date -u) kvp-ab relaunch pk" >>"$LOG"
  fi
}

kvnone; simcot; kvpab
echo "$(date -u) keeper cycle done" >>"$LOG"
