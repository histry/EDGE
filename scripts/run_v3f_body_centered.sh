#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

DATA_PATH="${DATA_PATH:-data/dunhuang_bvh/stationary_whitelist_v3_27units}"
PROJECT="${PROJECT:-runs/train_nextgen}"
EXP_NAME="${EXP_NAME:-stationary_v3f_bodycentered_x0w010_fk8_resp_e100}"

# Prefer V3B multi-unit checkpoint as the base.
CHECKPOINT="${CHECKPOINT:-runs/train_nextgen/stationary_v3b_whitelist24_from_v16_x0w04_noDCT_energy/weights/train-300.pt}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
VAL_BATCHES="${VAL_BATCHES:-4}"

if [ ! -d "$DATA_PATH" ]; then
  echo "❌ DATA_PATH not found: $DATA_PATH"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "❌ CHECKPOINT not found: $CHECKPOINT"
  echo "Try:"
  echo "  find runs/train_nextgen -path '*v3b*' -name '*.pt' | sort"
  exit 1
fi

# Clean V3 profile.
export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_V3_UNIT_RECON_DEBUG="${EDGE_V3_UNIT_RECON_DEBUG:-1}"

# Keep V3C visible-FK, but reduce it. V3F becomes the main new signal.
export EDGE_V3C_VISIBLE_FK="${EDGE_V3C_VISIBLE_FK:-1}"
export EDGE_V3C_VISIBLE_FK_WEIGHT="${EDGE_V3C_VISIBLE_FK_WEIGHT:-8.0}"
export EDGE_V3C_ACTIVITY_FLOOR="${EDGE_V3C_ACTIVITY_FLOOR:-0.75}"
export EDGE_V3C_RANGE_FLOOR="${EDGE_V3C_RANGE_FLOOR:-0.75}"
export EDGE_V3C_FK_SPEED_WEIGHT="${EDGE_V3C_FK_SPEED_WEIGHT:-1.0}"
export EDGE_V3C_FK_ENVELOPE_WEIGHT="${EDGE_V3C_FK_ENVELOPE_WEIGHT:-0.5}"
export EDGE_V3C_FK_ACTIVITY_WEIGHT="${EDGE_V3C_FK_ACTIVITY_WEIGHT:-2.0}"
export EDGE_V3C_FK_RANGE_WEIGHT="${EDGE_V3C_FK_RANGE_WEIGHT:-3.0}"
export EDGE_V3C_FK_HAND_WEIGHT="${EDGE_V3C_FK_HAND_WEIGHT:-0.5}"
export EDGE_V3C_FK_HAND_RANGE_WEIGHT="${EDGE_V3C_FK_HAND_RANGE_WEIGHT:-2.0}"
export EDGE_V3C_VISIBLE_FK_DEBUG="${EDGE_V3C_VISIBLE_FK_DEBUG:-0}"

# V3F body-centered torso-response loss.
export EDGE_V3F_BODY_CENTERED="${EDGE_V3F_BODY_CENTERED:-1}"
export EDGE_V3F_WEIGHT="${EDGE_V3F_WEIGHT:-1.0}"
export EDGE_V3F_TORSO_RANGE_FLOOR="${EDGE_V3F_TORSO_RANGE_FLOOR:-0.55}"
export EDGE_V3F_TORSO_RANGE_WEIGHT="${EDGE_V3F_TORSO_RANGE_WEIGHT:-12.0}"
export EDGE_V3F_TORSO_ARM_RATIO_WEIGHT="${EDGE_V3F_TORSO_ARM_RATIO_WEIGHT:-3.0}"
export EDGE_V3F_TORSO_ENV_WEIGHT="${EDGE_V3F_TORSO_ENV_WEIGHT:-2.0}"
export EDGE_V3F_RESPONSE_WEIGHT="${EDGE_V3F_RESPONSE_WEIGHT:-1.0}"
export EDGE_V3F_ARM_ENV_WEIGHT="${EDGE_V3F_ARM_ENV_WEIGHT:-0.5}"
export EDGE_V3F_HAND_ENV_WEIGHT="${EDGE_V3F_HAND_ENV_WEIGHT:-0.25}"
export EDGE_V3F_SPINE_DIR_WEIGHT="${EDGE_V3F_SPINE_DIR_WEIGHT:-1.0}"
export EDGE_V3F_JERK_WEIGHT="${EDGE_V3F_JERK_WEIGHT:-0.5}"
export EDGE_V3F_ROOT_STABLE_WEIGHT="${EDGE_V3F_ROOT_STABLE_WEIGHT:-0.5}"
export EDGE_V3F_ROOT_RATIO_WEIGHT="${EDGE_V3F_ROOT_RATIO_WEIGHT:-1.0}"
export EDGE_V3F_MAX_ROOT_ENERGY_RATIO="${EDGE_V3F_MAX_ROOT_ENERGY_RATIO:-0.35}"
export EDGE_V3F_DEBUG="${EDGE_V3F_DEBUG:-1}"

# Keep x0 reconstruction weak to avoid mean-motion collapse.
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT="${EDGE_X0_RECON_LOSS_WEIGHT:-0.10}"
export EDGE_X0_RECON_LOSS_DEBUG="${EDGE_X0_RECON_LOSS_DEBUG:-0}"

# Disable old controls.
export EDGE_V3_DCT_UNIT_LOSS=0
export EDGE_V3_DCT_UNIT_LOSS_WEIGHT=0.0
export EDGE_V3_TEMPORAL_WEIGHT=0.0
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_DUNHUANG_STRICT_SPLIT=0
export EDGE_DUNHUANG_SPLIT_MODE=all
export EDGE_FREEZE_AWARE_MOTION=0
export EDGE_TEMPORAL_PROGRESS_SUPERVISION=0
export EDGE_BURST_SAFE_PROGRESS=0
export EDGE_KINEMATIC_SMOOTHNESS=0
export EDGE_DIRECTION_CONSISTENCY=0
export EDGE_LOWPASS_TEMPORAL_PRIOR=0
export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

echo "==== V3F body-centered torso-response training ===="
echo "DATA_PATH=$DATA_PATH"
echo "EXP_NAME=$EXP_NAME"
echo "CHECKPOINT=$CHECKPOINT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "SAVE_INTERVAL=$SAVE_INTERVAL"
echo "MAX_TRAIN_BATCHES=$MAX_TRAIN_BATCHES"
echo "EDGE_X0_RECON_LOSS_WEIGHT=$EDGE_X0_RECON_LOSS_WEIGHT"
echo "EDGE_V3C_VISIBLE_FK_WEIGHT=$EDGE_V3C_VISIBLE_FK_WEIGHT"
echo "EDGE_V3F_WEIGHT=$EDGE_V3F_WEIGHT"
echo "=================================================="

mkdir -p logs

accelerate launch train.py \
  --data_path "$DATA_PATH" \
  --exp_name "$EXP_NAME" \
  --project "$PROJECT" \
  --checkpoint "$CHECKPOINT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --save_interval "$SAVE_INTERVAL" \
  --val_batches "$VAL_BATCHES" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
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
  --contact_loss_weight 0.02 \
  --foot_loss_weight 0.02 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --mixed_precision bf16 \
  --train_num_workers 2 \
  --val_num_workers 1 \
  2>&1 | tee "logs/train_${EXP_NAME}.log"
