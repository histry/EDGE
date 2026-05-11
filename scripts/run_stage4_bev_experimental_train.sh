#!/usr/bin/env bash
set -Eeuo pipefail

# Stage 4 experimental branch: BEV/stage-map trajectory residual + advanced adapter.
# This is intended after Stage 2 validates gait/trajectory features.

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/adapter-or-v12.pt}"
PROJECT=${PROJECT:-runs/train_bev_stage_map}
EXP_NAME=${EXP_NAME:-v12_bev_stage_map_experimental_v1}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-5}
DATA_PATH=${DATA_PATH:-data/dunhuang_bvh/processed}

export EDGE_GAIT_PHASE_COND=${EDGE_GAIT_PHASE_COND:-1}
export EDGE_TRAJ_PHYSICS_FEATURES=${EDGE_TRAJ_PHYSICS_FEATURES:-1}
export EDGE_TRAJ_FOURIER_FEATURES=${EDGE_TRAJ_FOURIER_FEATURES:-1}
export EDGE_TRAJ_BEV_COND=${EDGE_TRAJ_BEV_COND:-1}
export EDGE_TRAJ_BEV_SIZE=${EDGE_TRAJ_BEV_SIZE:-32}
export EDGE_TRAJ_BEV_SIGMA=${EDGE_TRAJ_BEV_SIGMA:-1.5}
export EDGE_TRAJ_SPARSE_WAYPOINT=${EDGE_TRAJ_SPARSE_WAYPOINT:-1}

python -u train.py \
  --data_path "$DATA_PATH" \
  --project "$PROJECT" \
  --exp_name "$EXP_NAME" \
  --checkpoint "$CHECKPOINT" \
  --train_stage adapter \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --learning_rate "$LR" \
  --trajectory_loss_weight ${TRAJECTORY_LOSS_WEIGHT:-1.0} \
  --trajectory_velocity_loss_weight ${TRAJECTORY_VELOCITY_LOSS_WEIGHT:-0.35} \
  --foot_loss_weight ${FOOT_LOSS_WEIGHT:-0.4} \
  --contact_loss_weight ${CONTACT_LOSS_WEIGHT:-0.4} \
  "$@"
