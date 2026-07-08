#!/bin/bash
# Phase-4 keeper (E12 stability sweep): cron every 10 min.
#
# Improvements over the phase-3 keeper (which restarted finished/preempted runs
# from scratch, twice wasting a full 3M run):
#   1. COMPLETION detection: remote runs/RUN/final_model.zip -> mark done locally
#      (data/RUN.done), download final CSVs, and immediately launch the NEXT run
#      in that lane's queue.
#   2. RESUME on relaunch: if the process died (preemption/crash) the relaunch
#      passes --resume-from <latest checkpoint> (train.py supports it; ckpts are
#      written every ~50k steps by CheckpointCallback).
#   3. Per-lane QUEUES so VMs are never idle during the 48 h window.
#
# Lanes (queue order = priority):
#   kv-none-v2 (L4 24GB, on-demand, us-west4):   e12_lrdecay_s0 -> e12_lrdecay_s1 -> e12_epochs4
#   kvp-ab     (L4 24GB, SPOT, asia-southeast1): e12_seed2 -> e12_seed3
#   simcot-t4  (T4 16GB, on-demand, us-central1): e12_cont_klC -> e12_klw15
#     (cont runs on the SAME T4 that trained s_e11_klC: same stack, most
#      comparable; its first launch resumes from runs/s_e11_klC/final_model.zip)
#
# NEVER stops a VM. flock guards overlap.
PH=/home/anatissera/Documents/UDESA/4th-year/1er-Semestre/RL/tp-final-rl-kv-eviction/kv-eviction-gym/experiments/phase4-stability
DATA=$PH/data
LOG=$PH/keeper2.log
SCRIPTS="${PH%/experiments/*}/scripts"
mkdir -p "$DATA"
exec 9>"$PH/.keeper2.lock"; flock -n 9 || exit 0
G(){ timeout 120 gcloud --account=atissera@udesa.edu.ar "$@"; }

# lane <vm> <project> <zone> <iap:0|1> <queue: cfg1:run1,cfg2:run2,...>
lane(){
  local VM=$1 PROJ=$2 ZONE=$3 IAP=$4 QUEUE=$5
  local IAPFLAG=""
  [ "$IAP" = "1" ] && IAPFLAG="--tunnel-through-iap"

  local ST=$(G compute instances describe "$VM" --project="$PROJ" --zone="$ZONE" --format="value(status)" 2>/dev/null)
  if [ "$ST" != "RUNNING" ]; then
    G compute instances start "$VM" --project="$PROJ" --zone="$ZONE" --quiet 2>/dev/null
    echo "$(date -u) $VM start (was $ST)" >>"$LOG"; return
  fi

  # first queue item without a local done-marker = current run
  local CFG="" RUN=""
  IFS=',' read -ra ITEMS <<< "$QUEUE"
  for item in "${ITEMS[@]}"; do
    local c="${item%%:*}" r="${item##*:}"
    if [ ! -f "$DATA/$r.done" ]; then CFG=$c; RUN=$r; break; fi
  done
  [ -z "$RUN" ] && { echo "$(date -u) $VM queue complete" >>"$LOG"; return; }

  # live-download CSVs for the current run
  G compute scp --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    "$VM:~/repo/runs/$RUN/probe_curve.csv" "$DATA/${RUN}_probe.csv" 2>/dev/null
  G compute scp --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    "$VM:~/repo/runs/$RUN/learning_curve.csv" "$DATA/${RUN}_learning.csv" 2>/dev/null

  # completion check (train.py writes final_model.zip when total_timesteps reached)
  local DONE=$(G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    --ssh-flag="-o ConnectTimeout=25" \
    --command="test -f ~/repo/runs/$RUN/final_model.zip && echo YES" 2>/dev/null | tr -dc 'A-Z')
  if [ "$DONE" = "YES" ]; then
    # extend-if-promising: a run that is beating kv_norm gets +3M steps (capped
    # at 10M) instead of being handed off to the next item in the queue.
    local CURTOT=$(grep -oP '^total_timesteps:\s*\K\d+' "$SCRIPTS/../configs/$CFG")
    local DECISION="STOP"
    if [ -n "$CURTOT" ] && [ -f "$DATA/${RUN}_probe.csv" ]; then
      DECISION=$(python3 "$SCRIPTS/should_extend.py" "$DATA/${RUN}_probe.csv" "$CURTOT" 2>/dev/null)
      [ -z "$DECISION" ] && DECISION="STOP"
    fi
    if [[ "$DECISION" == EXTEND:* ]]; then
      local NEWTOT="${DECISION#EXTEND:}"
      sed -i "s/^total_timesteps:.*/total_timesteps: $NEWTOT/" "$SCRIPTS/../configs/$CFG"
      G compute scp --project="$PROJ" --zone="$ZONE" $IAPFLAG \
        "$SCRIPTS/../configs/$CFG" "$VM:~/repo/configs/$CFG" 2>/dev/null
      G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" $IAPFLAG \
        --ssh-flag="-o ConnectTimeout=25" \
        --command="rm -f ~/repo/runs/$RUN/final_model.zip" 2>/dev/null
      echo "$(date -u) $VM $RUN EXTENDED $CURTOT -> $NEWTOT (beating kv_norm)" >>"$LOG"
      # fall through to the relaunch block below (same cycle, resumes from ckpt)
    else
      touch "$DATA/$RUN.done"
      echo "$(date -u) $VM $RUN COMPLETE ($DECISION) -> next in queue on next cycle" >>"$LOG"
      return   # next cycle launches the next queue item
    fi
  fi

  # prune old checkpoints (keep newest 2; each is ~53 MB, a full run writes ~60)
  G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    --ssh-flag="-o ConnectTimeout=25" \
    --command="ls -t ~/repo/runs/$RUN/checkpoints/*.zip 2>/dev/null | tail -n +3 | xargs -r rm -f" 2>/dev/null

  # process alive? (bracket trick so pgrep does not match this ssh command itself)
  local PAT="[s]${RUN#s}"
  local NP=$(G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    --ssh-flag="-o ConnectTimeout=25" \
    --command="pgrep -fc '$PAT'" 2>/dev/null | tr -dc '0-9')
  if [ "${NP:-0}" != "0" ]; then return; fi

  # (re)launch: own latest ckpt > INIT_FROM (continuation) > fresh
  local INIT=""
  [ "$RUN" = "s_e12_cont_klC" ] && INIT="runs/s_e11_klC/final_model.zip"
  [ "$RUN" = "s_e12_cont_klC_seed1" ] && INIT="runs/s_e11_klC_seed1_final_model.zip"
  local LAUNCH="cd ~/repo && CKPT=\$(ls -t runs/$RUN/checkpoints/*.zip 2>/dev/null | head -1); RES=''; if [ -n \"\$CKPT\" ]; then RES=\"--resume-from \$CKPT\"; elif [ -n '$INIT' ] && [ -f '$INIT' ]; then RES='--resume-from $INIT'; fi; tmux kill-session -t e12 2>/dev/null; tmux new-session -d -s e12 \"source .venv/bin/activate; export PYTHONPATH=~/repo/src; export MALLOC_MMAP_THRESHOLD_=1048576; python scripts/train.py --config configs/$CFG --run-name $RUN \$RES >> ~/$RUN.log 2>&1\"; echo LAUNCHED \$RES"
  local OUT=$(G compute ssh "$VM" --project="$PROJ" --zone="$ZONE" $IAPFLAG \
    --ssh-flag="-o ConnectTimeout=25" --command="$LAUNCH" 2>/dev/null | tail -1)
  echo "$(date -u) $VM launch $RUN: $OUT" >>"$LOG"
}

lane kv-none-v2 proyecto-final-425415 us-west4-a 1 \
  "e12_lrdecay_s0.yaml:s_e12_lrdecay_s0,e12_lrdecay_s1.yaml:s_e12_lrdecay_s1,e12_epochs4.yaml:s_e12_epochs4,e12_seed5.yaml:s_e12_seed5,e12_entcoef.yaml:s_e12_entcoef,e12_entcoef6.yaml:s_e12_entcoef6"
lane kvp-ab tp-final-rl-kv-eviction asia-southeast1-a 1 \
  "e12_seed2.yaml:s_e12_seed2,e12_seed3.yaml:s_e12_seed3,e12_cont_klC_seed1.yaml:s_e12_cont_klC_seed1,e12_seed4.yaml:s_e12_seed4,e12_targetkl.yaml:s_e12_targetkl"
lane simcot-t4 tp-final-nlp us-central1-a 0 \
  "e12_cont_klC.yaml:s_e12_cont_klC,e12_klw15.yaml:s_e12_klw15,e12_epochs15.yaml:s_e12_epochs15"

# mirror all CSV curves into TensorBoard event files (tb_all/) after downloads,
# so a locally-running `tensorboard --logdir tb_all` shows E12 growing live.
python3 "$SCRIPTS/csv_to_tb.py" >>"$LOG" 2>&1

# regenerate progress plots for active phase4 runs
python3 "$SCRIPTS/plot_progress.py" >>"$LOG" 2>&1

# regenerate docs/imgs/ figures that depend on active runs (E12)
python3 "$SCRIPTS/figures.py" >>"$LOG" 2>&1

echo "$(date -u) keeper2 cycle done" >>"$LOG"
