#!/usr/bin/env bash
# Full-training launcher for V10 Text/Pose Context RAG adapter.
#
# Drop-in replacement:
#   /home/disk/lsm/storage/EDGE/scripts/run_train_text_context_rag_adapter_fix2.sh
#
# Important:
# - MAX_TRAIN_BATCHES defaults to 0, meaning full epoch training.
# - For a smoke test, explicitly pass MAX_TRAIN_BATCHES=20 in the command line.
# - This avoids accidentally running an "8 epoch smoke" instead of a real e8 run.

set -euo pipefail
cd /home/disk/lsm/storage/EDGE

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
: "${EXP_NAME:=v10_text_context_rag_adapter_e8_fix2_full}"
: "${DATA_PATH:=data/dunhuang_bvh/processed}"
: "${BATCH_SIZE:=16}"
: "${EPOCHS:=8}"
: "${SAVE_INTERVAL:=1}"
: "${MAX_TRAIN_BATCHES:=0}"
: "${TRAIN_NUM_WORKERS:=-1}"
: "${VAL_NUM_WORKERS:=-1}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 2
fi

mkdir -p logs runs/train_stage45

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

# Optional debug
export EDGE_TEXT_CONTEXT_DEBUG="${EDGE_TEXT_CONTEXT_DEBUG:-0}"

echo "============================================================"
echo "EDGE V10 Text/Pose Context RAG Adapter Training"
echo "  CHECKPOINT=$CHECKPOINT"
echo "  EXP_NAME=$EXP_NAME"
echo "  DATA_PATH=$DATA_PATH"
echo "  BATCH_SIZE=$BATCH_SIZE"
echo "  EPOCHS=$EPOCHS"
echo "  SAVE_INTERVAL=$SAVE_INTERVAL"
echo "  MAX_TRAIN_BATCHES=$MAX_TRAIN_BATCHES  (0 = full epoch)"
echo "  EDGE_TRAJECTORY_REP=$EDGE_TRAJECTORY_REP"
echo "============================================================"

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
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --train_num_workers "$TRAIN_NUM_WORKERS" \
  --val_num_workers "$VAL_NUM_WORKERS" \
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
