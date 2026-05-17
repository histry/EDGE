#!/usr/bin/env bash
set -euo pipefail

# Freeze-aware v2d stationary endpoint-continuous training template.
# Edit paths below if your experiment names differ.

cd /home/disk/lsm/storage/EDGE
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate edge

DATA_PATH=${DATA_PATH:-data/dunhuang_bvh/stationary_whitelist_v2d_nostatic}
CKPT=${CKPT:-runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-200.pt}
EXP_NAME=${EXP_NAME:-stationary_v2d_nostatic_endpoint_freezeaware_x0w05}
LOG=${LOG:-logs/train_${EXP_NAME}.log}

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJECTORY_PLANE=xz

# Main freeze-aware patch switch.
export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_FREEZE_AWARE_AUTO_INSTALL=1
export EDGE_FREEZE_AWARE_FEATURE_MODE=${EDGE_FREEZE_AWARE_FEATURE_MODE:-upper_torso}

# Keep x0 reconstruction but do not let it dominate into endpoint collapse.
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=${EDGE_X0_RECON_LOSS_WEIGHT:-0.5}

# Strengthen coverage internally; current diffusion.py later multiplies this by physical_w * 0.05.
export EDGE_MOTION_ENERGY_LOSS_SCALE=${EDGE_MOTION_ENERGY_LOSS_SCALE:-4.0}
export EDGE_MOTION_CURVE_WEIGHT=${EDGE_MOTION_CURVE_WEIGHT:-1.0}
export EDGE_MOTION_ACTIVE_WEIGHT=${EDGE_MOTION_ACTIVE_WEIGHT:-3.0}
export EDGE_MOTION_COVERAGE_WEIGHT=${EDGE_MOTION_COVERAGE_WEIGHT:-6.0}
export EDGE_MOTION_TAIL_WEIGHT=${EDGE_MOTION_TAIL_WEIGHT:-6.0}
export EDGE_MOTION_PEAK_WEIGHT=${EDGE_MOTION_PEAK_WEIGHT:-0.5}
export EDGE_MOTION_MIN_TARGET_ENERGY_RATIO=${EDGE_MOTION_MIN_TARGET_ENERGY_RATIO:-0.45}
export EDGE_MOTION_TAIL_MIN_RATIO=${EDGE_MOTION_TAIL_MIN_RATIO:-0.45}
export EDGE_MOTION_COVERAGE_TARGET_SCALE=${EDGE_MOTION_COVERAGE_TARGET_SCALE:-0.75}

export EDGE_ANTI_FREEZE_LOSS_SCALE=${EDGE_ANTI_FREEZE_LOSS_SCALE:-4.0}
export EDGE_ANTI_FREEZE_COVERAGE_WEIGHT=${EDGE_ANTI_FREEZE_COVERAGE_WEIGHT:-2.0}
export EDGE_ANTI_FREEZE_TAIL_WEIGHT=${EDGE_ANTI_FREEZE_TAIL_WEIGHT:-2.0}
export EDGE_ANTI_FREEZE_MIN_ACTIVE_RATIO=${EDGE_ANTI_FREEZE_MIN_ACTIVE_RATIO:-0.30}

# Do not mix trajectory/RAG/beat while debugging stationary endpoint collapse.
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

mkdir -p logs

accelerate launch train.py \
  --data_path "$DATA_PATH" \
  --exp_name "$EXP_NAME" \
  --project runs/train_nextgen \
  --checkpoint "$CKPT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size 2 \
  --epochs 800 \
  --save_interval 100 \
  --train_stage full \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --keyframe_condition_prob 0.5 \
  --keyframe_loss_weight 0.1 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --contact_loss_weight 0.1 \
  --foot_loss_weight 0.1 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --mixed_precision bf16 \
  --train_num_workers 2 \
  --val_num_workers 1 \
  2>&1 | tee "$LOG"
