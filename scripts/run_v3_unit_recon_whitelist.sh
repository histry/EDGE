#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

DATA_PATH="${DATA_PATH:-data/dunhuang_bvh/stationary_whitelist_v3_27units}"
EXP_NAME="${EXP_NAME:-stationary_v3_whitelist24_full_unit_recon_x0w08_dct003}"
PROJECT="${PROJECT:-runs/train_nextgen}"

CHECKPOINT="${CHECKPOINT-}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-300}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
VAL_BATCHES="${VAL_BATCHES:-4}"

if [ ! -d "$DATA_PATH" ]; then
  echo "❌ DATA_PATH not found: $DATA_PATH"
  exit 1
fi

if ! ls "$DATA_PATH"/*.pkl >/dev/null 2>&1; then
  echo "❌ No .pkl files under DATA_PATH: $DATA_PATH"
  exit 1
fi

if [ -n "$CHECKPOINT" ] && [ ! -f "$CHECKPOINT" ]; then
  echo "❌ CHECKPOINT not found: $CHECKPOINT"
  echo "Set CHECKPOINT=/path/to/clean_base_or_v16_checkpoint.pt"
  exit 1
fi

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_V3_UNIT_RECON_DEBUG=1

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

export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT="${EDGE_X0_RECON_LOSS_WEIGHT:-0.8}"
export EDGE_X0_RECON_LOSS_DEBUG="${EDGE_X0_RECON_LOSS_DEBUG:-0}"

export EDGE_V3_DCT_UNIT_LOSS="${EDGE_V3_DCT_UNIT_LOSS:-1}"
export EDGE_V3_DCT_UNIT_LOSS_WEIGHT="${EDGE_V3_DCT_UNIT_LOSS_WEIGHT:-0.03}"
export EDGE_V3_DCT_KEEP="${EDGE_V3_DCT_KEEP:-8}"

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

echo "==== V3 Temporal Unit Reconstruction ===="
echo "DATA_PATH=$DATA_PATH"
echo "EXP_NAME=$EXP_NAME"
echo "CHECKPOINT=$CHECKPOINT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "EPOCHS=$EPOCHS"
echo "SAVE_INTERVAL=$SAVE_INTERVAL"
echo "MAX_TRAIN_BATCHES=$MAX_TRAIN_BATCHES"
echo "VAL_BATCHES=$VAL_BATCHES"
echo "EDGE_X0_RECON_LOSS_WEIGHT=$EDGE_X0_RECON_LOSS_WEIGHT"
echo "EDGE_V3_DCT_UNIT_LOSS_WEIGHT=$EDGE_V3_DCT_UNIT_LOSS_WEIGHT"
echo "========================================="

mkdir -p logs

CMD=(accelerate launch train.py
  --data_path "$DATA_PATH"
  --exp_name "$EXP_NAME"
  --project "$PROJECT"
)

if [ -n "$CHECKPOINT" ]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

CMD+=(
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
  --train_num_workers 4 \
  --val_num_workers 1
)

"${CMD[@]}" 2>&1 | tee "logs/train_${EXP_NAME}.log"
