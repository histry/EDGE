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
export EDGE_FREEZE_AWARE_DEBUG=${EDGE_FREEZE_AWARE_DEBUG:-0}
export EDGE_FREEZE_AWARE_FEATURE_MODE=${EDGE_FREEZE_AWARE_FEATURE_MODE:-upper_torso}

export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=${EDGE_X0_RECON_LOSS_WEIGHT:-0.35}
export EDGE_X0_RECON_LOSS_DEBUG=0

export EDGE_MOTION_ENERGY_LOSS_SCALE=${EDGE_MOTION_ENERGY_LOSS_SCALE:-2.0}
export EDGE_MOTION_MIN_TARGET_ENERGY_RATIO=${EDGE_MOTION_MIN_TARGET_ENERGY_RATIO:-0.45}
export EDGE_MOTION_MAX_TARGET_ENERGY_RATIO=${EDGE_MOTION_MAX_TARGET_ENERGY_RATIO:-2.0}
export EDGE_MOTION_ACTIVE_WEIGHT=${EDGE_MOTION_ACTIVE_WEIGHT:-1.5}
export EDGE_MOTION_OVERACTIVE_WEIGHT=${EDGE_MOTION_OVERACTIVE_WEIGHT:-1.5}
export EDGE_MOTION_COVERAGE_WEIGHT=${EDGE_MOTION_COVERAGE_WEIGHT:-2.0}
export EDGE_MOTION_TAIL_WEIGHT=${EDGE_MOTION_TAIL_WEIGHT:-2.0}
export EDGE_MOTION_TAIL_OVERACTIVE_WEIGHT=${EDGE_MOTION_TAIL_OVERACTIVE_WEIGHT:-1.0}
export EDGE_MOTION_TAIL_MIN_RATIO=${EDGE_MOTION_TAIL_MIN_RATIO:-0.45}
export EDGE_MOTION_TAIL_MAX_RATIO=${EDGE_MOTION_TAIL_MAX_RATIO:-2.0}

export EDGE_PROGRESS_CURVE=${EDGE_PROGRESS_CURVE:-gt}
export EDGE_PROGRESS_LOSS_WEIGHT=${EDGE_PROGRESS_LOSS_WEIGHT:-6.0}
export EDGE_PROGRESS_DISTANCE_WEIGHT=${EDGE_PROGRESS_DISTANCE_WEIGHT:-1.0}
export EDGE_PROGRESS_CUM_WEIGHT=${EDGE_PROGRESS_CUM_WEIGHT:-1.0}
export EDGE_PROGRESS_FRONTLOAD_WEIGHT=${EDGE_PROGRESS_FRONTLOAD_WEIGHT:-1.0}
export EDGE_PROGRESS_TOPK_WEIGHT=${EDGE_PROGRESS_TOPK_WEIGHT:-0.7}
export EDGE_PROGRESS_EARLY_WEIGHT=${EDGE_PROGRESS_EARLY_WEIGHT:-1.0}
export EDGE_PROGRESS_FRONTLOAD_MARGIN=${EDGE_PROGRESS_FRONTLOAD_MARGIN:-0.08}

export EDGE_ANTI_FREEZE_LOSS_SCALE=${EDGE_ANTI_FREEZE_LOSS_SCALE:-1.0}
export EDGE_ANTI_FREEZE_COVERAGE_WEIGHT=${EDGE_ANTI_FREEZE_COVERAGE_WEIGHT:-1.0}
export EDGE_ANTI_FREEZE_TAIL_WEIGHT=${EDGE_ANTI_FREEZE_TAIL_WEIGHT:-1.0}

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

DATA=${DATA:-data/dunhuang_bvh/stationary_whitelist_v2d_nostatic_best5}
DEFAULT_V2D="runs/train_nextgen/stationary_v2d_nostatic_best5_endpoint_soft_x0w025_freezeaware/weights/train-100.pt"
DEFAULT_V2B="runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-200.pt"
if [[ -z "${CKPT:-}" ]]; then
  if [[ -f "$DEFAULT_V2D" ]]; then CKPT="$DEFAULT_V2D"; else CKPT="$DEFAULT_V2B"; fi
fi
EXP_NAME=${EXP_NAME:-stationary_v2e_best5_temporal_progress_x0w035}

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
  --epochs 400 \
  --save_interval 100 \
  --train_stage full \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --keyframe_condition_prob 0.65 \
  --keyframe_condition_width 2 \
  --keyframe_loss_weight 0.08 \
  --mid_keyframe_condition_prob 0.35 \
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
