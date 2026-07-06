#!/bin/bash
# Phase-3 keeper: cron every 10 min. Keeps the E11 online-PPO sweep (dense causal
# KL reward on passkey, 3 hyperparameter configs) alive across preemption/proc
# death, live-downloads probe curves into data/. NEVER stops a VM. flock guards
# overlap.
#   A = s_e11_klC_seed1  (kv-none-v2, LR=1e-4 ent=0.005 n_epochs=4)
#   B = s_e11_klB  (kvp-ab SPOT, LR=5e-5 ent=0.003 n_epochs=4)
#   C = s_e11_klC  (simcot-t4,   LR=1e-4 ent=0.01  n_epochs=10)
PH=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase3-dataset-causality
DATA=$PH/data
LOG=$PH/keeper.log
exec 9>"$PH/.keeper.lock"; flock -n 9 || exit 0
G(){ gcloud --account=atissera@udesa.edu.ar "$@"; }
RUNCMD='cd ~/repo && tmux kill-session -t RS 2>/dev/null; tmux new-session -d -s RS "source .venv/bin/activate; export PYTHONPATH=~/repo/src; export MALLOC_MMAP_THRESHOLD_=1048576; python scripts/train.py --config configs/CFG --run-name RUN >> ~/RUN.log 2>&1"'

# --- config A on kv-none-v2 (on-demand L4, IAP) ---
kvnone(){
  local ST=$(G compute instances describe kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --format="value(status)" 2>/dev/null)
  [ "$ST" = "RUNNING" ] || { G compute instances start kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --quiet 2>/dev/null; echo "$(date -u) kv-none-v2 start" >>"$LOG"; return; }
  G compute scp --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap kv-none-v2:'~/repo/runs/s_e11_klC_seed1/probe_curve.csv' "$DATA/e11_klC_seed1_probe.csv" 2>/dev/null
  G compute scp --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap kv-none-v2:'~/repo/runs/s_e11_klC_seed1/learning_curve.csv' "$DATA/e11_klC_seed1_learning.csv" 2>/dev/null
  local NP=$(G compute ssh kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="pgrep -fc 'e11_klC_seed1'" 2>/dev/null | tr -dc 0-9)
  if [ "${NP:-0}" = "0" ]; then
    local c=${RUNCMD//RS/klA}; c=${c//CFG/e11_klC_seed1.yaml}; c=${c//RUN/s_e11_klC_seed1}
    G compute ssh kv-none-v2 --project=proyecto-final-425415 --zone=us-west4-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="$c" 2>/dev/null
    echo "$(date -u) kv-none-v2 relaunch klA" >>"$LOG"
  fi
}

# --- config C on simcot-t4 (on-demand T4, DIRECT ssh, no IAP) ---
simcot(){
  local ST=$(G compute instances describe simcot-t4 --project=tp-final-nlp --zone=us-central1-a --format="value(status)" 2>/dev/null)
  [ "$ST" = "RUNNING" ] || { G compute instances start simcot-t4 --project=tp-final-nlp --zone=us-central1-a --quiet 2>/dev/null; echo "$(date -u) simcot start" >>"$LOG"; return; }
  G compute scp --project=tp-final-nlp --zone=us-central1-a simcot-t4:'~/repo/runs/s_e11_klC/probe_curve.csv' "$DATA/e11_klC_probe.csv" 2>/dev/null
  G compute scp --project=tp-final-nlp --zone=us-central1-a simcot-t4:'~/repo/runs/s_e11_klC/learning_curve.csv' "$DATA/e11_klC_learning.csv" 2>/dev/null
  local NP=$(G compute ssh simcot-t4 --project=tp-final-nlp --zone=us-central1-a --command="pgrep -fc 'e11_klC'" 2>/dev/null | tr -dc 0-9)
  if [ "${NP:-0}" = "0" ]; then
    local c=${RUNCMD//RS/klC}; c=${c//CFG/e11_klC.yaml}; c=${c//RUN/s_e11_klC}
    G compute ssh simcot-t4 --project=tp-final-nlp --zone=us-central1-a --command="$c" 2>/dev/null
    echo "$(date -u) simcot relaunch klC" >>"$LOG"
  fi
}

# --- config B on kvp-ab (SPOT L4, IAP, revive on preempt) ---
kvpab(){
  local ST=$(G compute instances describe kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --format="value(status)" 2>/dev/null)
  if [ "$ST" != "RUNNING" ]; then
    G compute instances start kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --quiet 2>/dev/null
    echo "$(date -u) kvp-ab revive" >>"$LOG"; return
  fi
  G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap kvp-ab:'~/repo/runs/s_e11_klB/probe_curve.csv' "$DATA/e11_klB_probe.csv" 2>/dev/null
  G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap kvp-ab:'~/repo/runs/s_e11_klB/learning_curve.csv' "$DATA/e11_klB_learning.csv" 2>/dev/null
  local NP=$(G compute ssh kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="pgrep -fc 'e11_klB'" 2>/dev/null | tr -dc 0-9)
  if [ "${NP:-0}" = "0" ]; then
    local c=${RUNCMD//RS/klB}; c=${c//CFG/e11_klB.yaml}; c=${c//RUN/s_e11_klB}
    G compute ssh kvp-ab --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap --ssh-flag="-o ConnectTimeout=25" --command="$c" 2>/dev/null
    echo "$(date -u) kvp-ab relaunch klB" >>"$LOG"
  fi
}

# simcot (klC) DONE at 3M (final +0.10). Removed from keeper so it is not re-run.
# PHASE3 DONE 2026-07-06: all E11 runs complete. Relaunches disabled; phase4 keeper2 takes over.
# kvnone; kvpab
echo "$(date -u) keeper cycle done" >>"$LOG"
