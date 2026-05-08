#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
: "${EXP_NAME:=v10_text_context_rag_adapter_e8_fix1}"
: "${DATA_PATH:=data/dunhuang_bvh/processed}"
: "${BATCH_SIZE:=16}"
: "${EPOCHS:=8}"
: "${SAVE_INTERVAL:=1}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 2
fi

# Text/Pose Context RAG
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_TRAIN_SELF=1
export EDGE_TEXT_CONTEXT_TRAIN_COUNT="${EDGE_TEXT_CONTEXT_TRAIN_COUNT:-3}"
export EDGE_TEXT_CONTEXT_DIM="${EDGE_TEXT_CONTEXT_DIM:-512}"
export EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS="${EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS:-64}"
export EDGE_RAG_CONTEXT_MAX_LEN="${EDGE_RAG_CONTEXT_MAX_LEN:-45}"
export EDGE_TEXT_CONTEXT_DROP_PROB="${EDGE_TEXT_CONTEXT_DROP_PROB:-0.10}"

# V9 RAG Summary Token
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1

# Full-landing trajectory representation
export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"

# Optional debug. Set to 1 for a smoke run if you want to print context captions.
export EDGE_TEXT_CONTEXT_DEBUG="${EDGE_TEXT_CONTEXT_DEBUG:-0}"

python train.py \
  --project runs/train_stage45 \
  --exp_name "$EXP_NAME" \
  --data_path "$DATA_PATH" \
  --checkpoint "$CHECKPOINT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --save_interval "$SAVE_INTERVAL" \
  --train_stage adapter \
  --gradient_checkpointing \
  --enable_rag_summary_token \
  --rag_summary_dim 7 \
  --rag_summary_drop_prob 0.15 \
  --energy_condition_prob 0.85 \
  --energy_condition_drop_prob 0.20 \
  --energy_loss_weight 0.35 \
  --trajectory_loss_weight 1.0 \
  --trajectory_velocity_loss_weight 0.35 \
  --root_lower_coupling_loss_weight 0.5 \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0
