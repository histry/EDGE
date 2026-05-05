#!/usr/bin/env bash
# V7 root-lower fixed training sweep.
#
# Usage:
#   cd /home/disk/lsm/storage/EDGE
#   bash scripts/run_v7_lower_fixed_sweep.sh
#
# Assumptions:
#   1) trajectory_native_control.py and edge_safety_patch.py have been replaced.
#   2) sitecustomize.py imports both patches, or train.py/import path triggers them.
#   3) Your checkpoint path below exists. Edit CHECKPOINT if needed.

set -euo pipefail

CHECKPOINT="${CHECKPOINT:-runs/train_stage45/stage45_riskfix_v1_e504/weights/train-10.pt}"
PROJECT="${PROJECT:-runs/train_stage45}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-5}"

export EDGE_EXPLICIT_LOWER_RELVEL_RATIO="${EDGE_EXPLICIT_LOWER_RELVEL_RATIO:-0.30}"
export EDGE_EXPLICIT_LOWER_MIN_MOTION="${EDGE_EXPLICIT_LOWER_MIN_MOTION:-0.01}"
export EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD="${EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD:-0.006}"
export EDGE_EXPLICIT_LOWER_STRICT="${EDGE_EXPLICIT_LOWER_STRICT:-1}"
export EDGE_EXPLICIT_LOWER_DEBUG="${EDGE_EXPLICIT_LOWER_DEBUG:-1}"

# Keep reporting honest: Dunhuang is unpaired unless you have verified paired data.
export EDGE_ALLOW_FROZEN_RANDOM_AUDIO="${EDGE_ALLOW_FROZEN_RANDOM_AUDIO:-0}"

python test_lower_loss.py

for W in 0.05 0.10 0.20; do
  export EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT="${W}"
  TAG="$(python - <<'PY'
import os
w = float(os.environ["EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT"])
print(f"w{int(round(w*1000)):03d}")
PY
)"

  EXP_NAME="v7_lower_fixed_${TAG}_r03"

  echo "============================================================"
  echo "Running ${EXP_NAME}"
  echo "  CHECKPOINT=${CHECKPOINT}"
  echo "  WEIGHT=${EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT}"
  echo "  RATIO=${EDGE_EXPLICIT_LOWER_RELVEL_RATIO}"
  echo "============================================================"

  python train.py \
    --checkpoint "${CHECKPOINT}" \
    --project "${PROJECT}" \
    --exp_name "${EXP_NAME}" \
    --train_stage adapter \
    --adapter_train_decoder \
    --epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --audio_pairing_mode none \
    --mmr_loss_weight 0.0 \
    --save_interval 2 \
    --val_batches 10
done
