#!/bin/bash
# Canonical way to serve the TensorBoard mirror. ALWAYS use this instead of a
# bare `tensorboard --logdir tb_all`, because csv_to_tb.py writes one event
# file per cron cycle (append-only, to avoid TB's purge_orphaned_data resetting
# the curve on every rewrite). Reading all of those files REQUIRES
# --reload_multifile true; without it TB shows only one slice and the x-axis
# looks frozen even as the run advances.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 -m tensorboard.main \
  --logdir "$REPO/tb_all" \
  --port "${1:-6006}" \
  --reload_interval 30 \
  --reload_multifile true
