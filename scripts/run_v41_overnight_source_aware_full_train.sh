#!/usr/bin/env bash
set -euo pipefail
cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
source scripts/v41_beat_support_env.sh
export V34_MOTION_QUALITY_POSTPROCESS=1
export V34_TRAIN=${V34_TRAIN:-1}
export RUN_ID="v41_beat_support_full_train_$(date +%Y%m%d_%H%M%S)"
export RUN_ROOT="output/$RUN_ID"
echo "$RUN_ROOT" > output/LATEST_V41_BEAT_SUPPORT_FULL_TRAIN.txt
bash scripts/launch_v40_source_aware_rag.sh
