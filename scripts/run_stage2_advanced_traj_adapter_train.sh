#!/usr/bin/env bash
set -Eeuo pipefail

# Stage 2 advanced adapter training:
# gait phase + Fourier trajectory + physics features + sparse waypoint mask + dynamic CFG.

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/v12/train-50.pt}"
PROJECT=${PROJECT:-runs/train_advanced_traj_phase}
EXP_NAME=${EXP_NAME:-v12_gait_fourier_sparse_adapter_v1}
BATCH_SIZE=${BATCH_SIZE:-4}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
DATA_PATH=${DATA_PATH:-data/dunhuang_bvh/processed}

export EDGE_DUNHUANG_SPLIT_MODE=${EDGE_DUNHUANG_SPLIT_MODE:-source_file}
export EDGE_DUNHUANG_STRICT_SPLIT=${EDGE_DUNHUANG_STRICT_SPLIT:-1}
export EDGE_TRAJECTORY_PLANE=${EDGE_TRAJECTORY_PLANE:-xz}

export EDGE_GAIT_PHASE_COND=${EDGE_GAIT_PHASE_COND:-1}
export EDGE_GAIT_PHASE_DIM=${EDGE_GAIT_PHASE_DIM:-6}
export EDGE_GAIT_CONTACT_LOSS=${EDGE_GAIT_CONTACT_LOSS:-1}
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=${EDGE_GAIT_CONTACT_LOSS_WEIGHT:-0.60}

export EDGE_TRAJ_PHYSICS_FEATURES=${EDGE_TRAJ_PHYSICS_FEATURES:-1}
export EDGE_TRAJ_FOURIER_FEATURES=${EDGE_TRAJ_FOURIER_FEATURES:-1}
export EDGE_TRAJ_FOURIER_BANDS=${EDGE_TRAJ_FOURIER_BANDS:-6}
export EDGE_TRAJ_SPARSE_WAYPOINT=${EDGE_TRAJ_SPARSE_WAYPOINT:-1}
export EDGE_TRAJ_WAYPOINT_FRAMES=${EDGE_TRAJ_WAYPOINT_FRAMES:-0,50,100,149}
export EDGE_DYNAMIC_TRAJ_CFG=${EDGE_DYNAMIC_TRAJ_CFG:-1}
export EDGE_TRAJ_CFG_BASE=${EDGE_TRAJ_CFG_BASE:-2.0}
export EDGE_TRAJ_CFG_SPEED_W=${EDGE_TRAJ_CFG_SPEED_W:-2.0}
export EDGE_TRAJ_CFG_CURVATURE_W=${EDGE_TRAJ_CFG_CURVATURE_W:-1.0}

export EDGE_DIFF_CONTACT_LOSS=${EDGE_DIFF_CONTACT_LOSS:-1}
export EDGE_DCL_CONTACT_SOURCE=${EDGE_DCL_CONTACT_SOURCE:-auto}
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=${EDGE_DCL_MAX_TARGET_CONTACT_RATIO:-0.85}
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=${EDGE_DCL_FALLBACK_CONTACT_SOURCE:-pred_fk_height}

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
  --sync_loss_weight ${SYNC_LOSS_WEIGHT:-1.0} \
  "$@"
