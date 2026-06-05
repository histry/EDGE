#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
source /home/disk/lsm/conda_envs/edge/bin/activate 2>/dev/null || true
export PYTHONPATH=$PWD:${PYTHONPATH:-}
MASTER=${MASTER:-output/v20_full_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$MASTER"
RUN_ROOT="$MASTER/db" bash scripts/run_v20_build_dynamic_event_db.sh
RUN_ROOT="$MASTER/scheduler" bash scripts/run_v20_rule_scheduler.sh
RUN_ROOT="$MASTER/transition" bash scripts/run_v20_train_transition_models.sh
echo "DONE: $MASTER"
