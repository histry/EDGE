#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

mkdir -p logs runs/train_nextgen

# Use a clean / natural base checkpoint. Avoid V2E/V2F reward-hacking checkpoints.
BASE_CKPT="${BASE_CKPT:-runs/train_nextgen/strict_single_unit45_recon_v16_smooth_dense8_from_v15/weights/train-5000.pt}"
DATA_PATH="${DATA_PATH:-data/dunhuang_bvh/stationary_whitelist_v2d_nostatic_best5}"
EXP_NAME="${EXP_NAME:-stationary_v3_best5_full_unit_recon_x0w08_dct}"

if [[ ! -f "$BASE_CKPT" ]]; then
  echo "❌ BASE_CKPT not found: $BASE_CKPT"
  echo "Set BASE_CKPT to the cleanest pre-V2E/V2F checkpoint, preferably v16 smooth dense8 or a clean EDGE/Dunhuang base."
  exit 1
fi
if [[ ! -d "$DATA_PATH" ]]; then
  echo "❌ DATA_PATH not found: $DATA_PATH"
  echo "Set DATA_PATH to your exported best5 pkl directory."
  exit 1
fi

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1

export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_DUNHUANG_SPLIT_MODE=all
export EDGE_DUNHUANG_STRICT_SPLIT=0

# Explicitly disable old V2/control routes.
export EDGE_FREEZE_AWARE_MOTION=0
export EDGE_TEMPORAL_PROGRESS_SUPERVISION=0
export EDGE_BURST_SAFE_PROGRESS=0
export EDGE_KINEMATIC_SMOOTHNESS=0
export EDGE_DIRECTION_CONSISTENCY=0
export EDGE_LOWPASS_TEMPORAL_PRIOR=0

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJ_EVENT_COND=0
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

# Main V3 losses.
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT="${EDGE_X0_RECON_LOSS_WEIGHT:-0.8}"
export EDGE_X0_RECON_LOSS_DEBUG="${EDGE_X0_RECON_LOSS_DEBUG:-0}"

# Whole-unit low-frequency temporal structure.
export EDGE_V3_TEMPORAL_WEIGHT="${EDGE_V3_TEMPORAL_WEIGHT:-0.20}"
export EDGE_V3_DCT_KEEP="${EDGE_V3_DCT_KEEP:-6}"
export EDGE_V3_TEMPORAL_FEATURES="${EDGE_V3_TEMPORAL_FEATURES:-no_contact}"
export EDGE_V3_VELOCITY_WEIGHT="${EDGE_V3_VELOCITY_WEIGHT:-0.05}"
export EDGE_V3_ACCEL_WEIGHT="${EDGE_V3_ACCEL_WEIGHT:-0.01}"

accelerate launch train.py \
  --data_path "$DATA_PATH" \
  --exp_name "$EXP_NAME" \
  --project runs/train_nextgen \
  --checkpoint "$BASE_CKPT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size "${BATCH_SIZE:-1}" \
  --epochs "${EPOCHS:-300}" \
  --save_interval "${SAVE_INTERVAL:-50}" \
  --train_stage full \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --keyframe_condition_prob 0.0 \
  --keyframe_loss_weight 0.0 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --contact_loss_weight 0.05 \
  --foot_loss_weight 0.05 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --cond_drop_prob 0.0 \
  --mixed_precision bf16 \
  --train_num_workers "${TRAIN_NUM_WORKERS:-2}" \
  --val_num_workers "${VAL_NUM_WORKERS:-1}" \
  2>&1 | tee "logs/train_${EXP_NAME}.log"
