#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0

export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_TEMPORAL_PROGRESS_SUPERVISION=1
export EDGE_BURST_SAFE_PROGRESS=1
export EDGE_KINEMATIC_SMOOTHNESS=1
export EDGE_DIRECTION_CONSISTENCY=1
export EDGE_LOWPASS_TEMPORAL_PRIOR=1
export EDGE_FREEZE_AWARE_DEBUG=${EDGE_FREEZE_AWARE_DEBUG:-0}
export EDGE_FREEZE_AWARE_DEBUG_STEPS=${EDGE_FREEZE_AWARE_DEBUG_STEPS:-10}
export EDGE_FREEZE_AWARE_FEATURE_MODE=${EDGE_FREEZE_AWARE_FEATURE_MODE:-upper_torso}

export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=${EDGE_X0_RECON_LOSS_WEIGHT:-0.30}
export EDGE_X0_RECON_LOSS_DEBUG=0

export EDGE_MOTION_ENERGY_LOSS_SCALE=${EDGE_MOTION_ENERGY_LOSS_SCALE:-1.0}
export EDGE_PROGRESS_SATURATION_MODE=${EDGE_PROGRESS_SATURATION_MODE:-tanh}
export EDGE_PROGRESS_CAP_TARGET_MEAN_MULT=${EDGE_PROGRESS_CAP_TARGET_MEAN_MULT:-0.75}
export EDGE_PROGRESS_ENERGY_CAP_FLOOR=${EDGE_PROGRESS_ENERGY_CAP_FLOOR:-0.05}

export EDGE_MOTION_MIN_TARGET_ENERGY_RATIO=${EDGE_MOTION_MIN_TARGET_ENERGY_RATIO:-0.45}
export EDGE_MOTION_MAX_TARGET_ENERGY_RATIO=${EDGE_MOTION_MAX_TARGET_ENERGY_RATIO:-1.35}
export EDGE_MOTION_TAIL_MIN_RATIO=${EDGE_MOTION_TAIL_MIN_RATIO:-0.45}
export EDGE_MOTION_TAIL_MAX_RATIO=${EDGE_MOTION_TAIL_MAX_RATIO:-1.35}

export EDGE_PROGRESS_CURVE=${EDGE_PROGRESS_CURVE:-gt}
export EDGE_PROGRESS_LOSS_WEIGHT=${EDGE_PROGRESS_LOSS_WEIGHT:-3.0}
export EDGE_PROGRESS_DISTANCE_WEIGHT=${EDGE_PROGRESS_DISTANCE_WEIGHT:-1.0}
export EDGE_PROGRESS_CUM_WEIGHT=${EDGE_PROGRESS_CUM_WEIGHT:-1.0}
export EDGE_PROGRESS_FRONTLOAD_WEIGHT=${EDGE_PROGRESS_FRONTLOAD_WEIGHT:-1.2}
export EDGE_PROGRESS_TOPK_WEIGHT=${EDGE_PROGRESS_TOPK_WEIGHT:-1.2}
export EDGE_PROGRESS_EARLY_WEIGHT=${EDGE_PROGRESS_EARLY_WEIGHT:-0.8}
export EDGE_PROGRESS_FRONTLOAD_MARGIN=${EDGE_PROGRESS_FRONTLOAD_MARGIN:-0.08}

export EDGE_KINEMATIC_SMOOTHNESS_WEIGHT=${EDGE_KINEMATIC_SMOOTHNESS_WEIGHT:-8.0}
export EDGE_ACCEL_MATCH_WEIGHT=${EDGE_ACCEL_MATCH_WEIGHT:-1.0}
export EDGE_JERK_MATCH_WEIGHT=${EDGE_JERK_MATCH_WEIGHT:-1.5}
export EDGE_ACCEL_SPIKE_WEIGHT=${EDGE_ACCEL_SPIKE_WEIGHT:-1.0}
export EDGE_JERK_SPIKE_WEIGHT=${EDGE_JERK_SPIKE_WEIGHT:-2.0}
export EDGE_ACCEL_SPIKE_MAX_RATIO=${EDGE_ACCEL_SPIKE_MAX_RATIO:-1.7}
export EDGE_JERK_SPIKE_MAX_RATIO=${EDGE_JERK_SPIKE_MAX_RATIO:-1.5}

export EDGE_DIRECTION_WEIGHT=${EDGE_DIRECTION_WEIGHT:-2.0}
export EDGE_DIRECTION_TARGET_MATCH_WEIGHT=${EDGE_DIRECTION_TARGET_MATCH_WEIGHT:-0.5}

export EDGE_LOWPASS_WEIGHT=${EDGE_LOWPASS_WEIGHT:-4.0}
export EDGE_LOWPASS_KERNEL_SIZE=${EDGE_LOWPASS_KERNEL_SIZE:-5}
export EDGE_LOWPASS_HIGH_MAX_RATIO=${EDGE_LOWPASS_HIGH_MAX_RATIO:-1.3}

export EDGE_ANTI_FREEZE_LOSS_SCALE=${EDGE_ANTI_FREEZE_LOSS_SCALE:-0.5}
export EDGE_ANTI_FREEZE_COVERAGE_WEIGHT=${EDGE_ANTI_FREEZE_COVERAGE_WEIGHT:-0.5}
export EDGE_ANTI_FREEZE_TAIL_WEIGHT=${EDGE_ANTI_FREEZE_TAIL_WEIGHT:-0.5}

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

DATA=${DATA:-data/dunhuang_bvh/stationary_whitelist_v2d_nostatic_best5}
DEFAULT_V2E="runs/train_nextgen/stationary_v2e_best5_temporal_progress_x0w035/weights/train-100.pt"
DEFAULT_V2D="runs/train_nextgen/stationary_v2d_nostatic_best5_endpoint_soft_x0w025_freezeaware/weights/train-100.pt"
DEFAULT_V2B="runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-200.pt"
if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "$DEFAULT_V2E" ]]; then CKPT="$DEFAULT_V2E";
  elif [[ -f "$DEFAULT_V2D" ]]; then CKPT="$DEFAULT_V2D";
  else CKPT="$DEFAULT_V2B"; fi
fi
EXP_NAME=${EXP_NAME:-stationary_v2f_best5_burstsafe_x0w030}
echo "DATA=$DATA"
echo "CKPT=$CKPT"
echo "EXP_NAME=$EXP_NAME"

accelerate launch train.py \
  --data_path "$DATA" \
  --exp_name "$EXP_NAME" \
  --project runs/train_nextgen \
  --checkpoint "$CKPT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size 1 \
  --epochs 300 \
  --save_interval 50 \
  --train_stage full \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --keyframe_condition_prob 0.60 \
  --keyframe_condition_width 2 \
  --keyframe_loss_weight 0.06 \
  --mid_keyframe_condition_prob 0.25 \
  --mid_keyframe_count 3 \
  --mid_keyframe_condition_width 1 \
  --mid_keyframe_selection random \
  --contact_loss_weight 0.1 \
  --foot_loss_weight 0.1 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --mixed_precision bf16 \
  --train_num_workers 2 \
  --val_num_workers 1 \
  2>&1 | tee "logs/train_${EXP_NAME}.log"
