#!/bin/bash
# Idempotent keeper for the two active runs — designed to be driven by LOCAL
# CRON every 10 min (long-lived watcher processes kept getting killed).
# Per run: revive VM after preemption, relaunch tmux if the proc died
# (everything resumes from checkpoints/JSONL), live-download curves, and on
# the DONE marker: final download + STOP the VM + drop a local .finished
# marker so later cycles skip it. Uses flock to avoid overlapping cycles.
OUT=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/ab_results
LOG=$OUT/ensure_runs.log
exec 9>"$OUT/.ensure_runs.lock"
flock -n 9 || exit 0
G(){ gcloud --account=atissera@udesa.edu.ar "$@"; }

ensure(){ # name proj zone donefile tmuxsess launchcmd checkproc localdone
  local VM=$1 PROJ=$2 ZONE=$3 DONEF=$4 SESS=$5 LAUNCH=$6 PROCPAT=$7 LOCALDONE=$8
  [ -f "$OUT/$LOCALDONE" ] && return 0
  local ST
  ST=$(G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" --format="value(status)" 2>/dev/null)
  if [ "$ST" = "TERMINATED" ] || [ "$ST" = "STOPPED" ]; then
    echo "$(date -u) [$VM] $ST -> start+relaunch" >> "$LOG"
    G compute instances start "$VM" --project="$PROJ" --zone="$ZONE" --quiet 2>/dev/null || return 0
    sleep 60
    G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=30" \
      --command="tmux kill-session -t $SESS 2>/dev/null; tmux new-session -d -s $SESS '$LAUNCH'" 2>/dev/null
    return 0
  fi
  [ "$ST" = "RUNNING" ] || { echo "$(date -u) [$VM] status=$ST (transitional)" >> "$LOG"; return 0; }
  local R
  R=$(G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=30" \
      --command="test -f $DONEF && echo DONE || echo RUNNING; ps -eo args | grep -c '$PROCPAT'" 2>/dev/null)
  local STATE NP
  STATE=$(echo "$R" | sed -n 1p); NP=$(echo "$R" | sed -n 2p)
  echo "$(date -u) [$VM] $STATE procs=$NP" >> "$LOG"
  if [ "$STATE" = "DONE" ]; then
    dl_"$SESS"   # run-specific downloads
    for t in 1 2 3; do G compute instances stop "$VM" --project="$PROJ" --zone="$ZONE" --quiet 2>/dev/null && break; sleep 20; done
    touch "$OUT/$LOCALDONE"
    echo "$(date -u) [$VM] FINISHED -> VM stopped, marker $LOCALDONE" >> "$LOG"
    return 0
  fi
  if [ -z "$STATE" ]; then return 0; fi   # ssh flake — try next cycle
  if [ "$NP" = "0" ]; then
    echo "$(date -u) [$VM] proc dead -> relaunch" >> "$LOG"
    G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" --tunnel-through-iap \
      --ssh-flag="-o ConnectTimeout=30" \
      --command="tmux kill-session -t $SESS 2>/dev/null; tmux new-session -d -s $SESS '$LAUNCH'" 2>/dev/null
  fi
  dl_"$SESS"
}

dl_e7(){
  G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap \
    kvp-ab:'~/repo/runs/s_e7/probe_curve.csv' "$OUT/s_e7_probe_curve.csv" 2>/dev/null
  G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap \
    kvp-ab:'~/repo/runs/s_e7/learning_curve.csv' "$OUT/s_e7_learning_curve.csv" 2>/dev/null
}
dl_orclg(){
  G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
    kv-chat-v1:'~/repo/runs/oracle_longgen/oracle_results.jsonl' "$OUT/oracle_longgen_results.jsonl" 2>/dev/null
}
dl_poolscreen(){
  G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
    kv-chat-v1:'~/repo/runs/pool_screen/pool_screen.jsonl' "$OUT/pool_screen.jsonl" 2>/dev/null
}
dl_e8(){
  for f in probe_curve learning_curve; do
    G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap \
      kvp-ab:"~/repo/runs/s_e8/$f.csv" "$OUT/s_e8_$f.csv" 2>/dev/null
  done
}
dl_e8attn(){
  for f in probe_curve learning_curve; do
    G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
      kv-chat-v1:"~/repo/runs/s_e8attn/$f.csv" "$OUT/s_e8attn_$f.csv" 2>/dev/null
  done
}
dl_e9(){
  for f in probe_curve learning_curve; do
    G compute scp --project=proyecto-final-425415 --zone=us-west1-a --tunnel-through-iap \
      kv-chat-v1:"~/repo/runs/s_e9/$f.csv" "$OUT/s_e9_$f.csv" 2>/dev/null
  done
}
dl_e9attn(){
  for f in probe_curve learning_curve; do
    G compute scp --project=tp-final-rl-kv-eviction --zone=asia-southeast1-a --tunnel-through-iap \
      kvp-ab:"~/repo/runs/s_e9attn/$f.csv" "$OUT/s_e9attn_$f.csv" 2>/dev/null
  done
}

ensure kv-chat-v1 proyecto-final-425415 us-west1-a '~/scaled/E9_DONE' e9 \
  "bash ~/run_e9.sh" 'configs/e9_perlayer.yaml' e9.finished
ensure kvp-ab tp-final-rl-kv-eviction asia-southeast1-a '~/scaled/E9ATTN_DONE' e9attn \
  "bash ~/run_e9attn.sh" 'configs/e9_perlayer_attn' e9attn.finished
