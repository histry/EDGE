#!/usr/bin/env bash
# Single recommended V7 training run.

set -euo pipefail

export EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT="${EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT:-0.10}"
export EDGE_EXPLICIT_LOWER_RELVEL_RATIO="${EDGE_EXPLICIT_LOWER_RELVEL_RATIO:-0.30}"
export EDGE_EXPLICIT_LOWER_MIN_MOTION="${EDGE_EXPLICIT_LOWER_MIN_MOTION:-0.01}"
export EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD="${EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD:-0.006}"
export EDGE_EXPLICIT_LOWER_STRICT="${EDGE_EXPLICIT_LOWER_STRICT:-1}"
export EDGE_EXPLICIT_LOWER_DEBUG="${EDGE_EXPLICIT_LOWER_DEBUG:-1}"

CHECKPOINT="${CHECKPOINT:-runs/train_stage45/stage45_riskfix_v1_e504/weights/train-10.pt}"

python test_lower_loss.py

python train.py \
  --checkpoint "${CHECKPOINT}" \
  --project runs/train_stage45 \
  --exp_name v7_lower_fixed_w010_r03 \
  --train_stage adapter \
  --adapter_train_decoder \
  --epochs 10 \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --save_interval 2 \
  --val_batches 10
