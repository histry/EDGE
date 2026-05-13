#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

# ===== turn-event native adapter =====
export EDGE_TURN_EVENT_MODEL_ADAPTER=1
export EDGE_TURN_EVENT_TRAJ_TOKEN=1
export EDGE_TURN_EVENT_OUTPUT_ADAPTER=0
unset EDGE_TURN_EVENT_ADAPTER_CKPT || true

# adapter-only / freeze backbone
export EDGE_TURN_EVENT_FREEZE_BACKBONE=1

# turn event settings
export EDGE_TURN_SUPPORT_LAG=${EDGE_TURN_SUPPORT_LAG:-8}
export EDGE_TURN_EXPR_LAG=${EDGE_TURN_EXPR_LAG:-4}
export EDGE_TURN_MIN_GAP=${EDGE_TURN_MIN_GAP:-18}
export EDGE_TURN_GATE_SIGMA=${EDGE_TURN_GATE_SIGMA:-5.0}

# safe trajectory behavior
export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_TURN_EVENT_PRESERVE_ROOT_XZ=1
export EDGE_TURN_EVENT_GATE_ROOT_XZ=0.0
export EDGE_TURN_EVENT_GATE_ROOT_Y=0.0
export EDGE_TURN_EVENT_GATE_LOWER=${EDGE_TURN_EVENT_GATE_LOWER:-0.45}
export EDGE_TURN_EVENT_GATE_TORSO=${EDGE_TURN_EVENT_GATE_TORSO:-0.75}
export EDGE_TURN_EVENT_GATE_UPPER=${EDGE_TURN_EVENT_GATE_UPPER:-0.75}

# keep existing trajectory/gait enhancements if supported
export EDGE_GAIT_PHASE_COND=${EDGE_GAIT_PHASE_COND:-1}
export EDGE_GAIT_PHASE_DIM=${EDGE_GAIT_PHASE_DIM:-6}
export EDGE_TRAJ_PHYSICS_FEATURES=${EDGE_TRAJ_PHYSICS_FEATURES:-1}
export EDGE_TRAJ_FOURIER_FEATURES=${EDGE_TRAJ_FOURIER_FEATURES:-1}
export EDGE_TRAJ_FOURIER_BANDS=${EDGE_TRAJ_FOURIER_BANDS:-6}
export EDGE_TRAJ_SPARSE_WAYPOINT=${EDGE_TRAJ_SPARSE_WAYPOINT:-1}

RUN_TAG=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/stage2b_turn_event_native_${RUN_TAG}"
mkdir -p "$LOG_DIR"

echo "========== Stage2B Turn-aware Native Adapter =========="
env | grep -E "EDGE_TURN|EDGE_DYNAMIC_TRAJ_CFG|EDGE_GAIT|EDGE_TRAJ" | sort | tee "$LOG_DIR/env.log"

# Choose existing training script.
if [ -f scripts/run_stage2_advanced_traj_adapter_train.sh ]; then
  bash scripts/run_stage2_advanced_traj_adapter_train.sh 2>&1 | tee "$LOG_DIR/train.log"
elif [ -f scripts/run_train_advanced_traj_phase.sh ]; then
  bash scripts/run_train_advanced_traj_phase.sh 2>&1 | tee "$LOG_DIR/train.log"
elif [ -f scripts/run_train_support_textpose_rag.sh ]; then
  bash scripts/run_train_support_textpose_rag.sh 2>&1 | tee "$LOG_DIR/train.log"
else
  echo "[ERROR] No known trajectory/support adapter training script found." | tee "$LOG_DIR/error.log"
  echo "Available training-like scripts:" | tee -a "$LOG_DIR/error.log"
  ls scripts | grep -E "stage2|traj|gait|support|adapter|train" | tee -a "$LOG_DIR/error.log"
  exit 1
fi

echo "========== Latest checkpoints =========="
find runs -type f -name "train-*.pt" -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort | tail -20 | tee "$LOG_DIR/latest_checkpoints.log"
