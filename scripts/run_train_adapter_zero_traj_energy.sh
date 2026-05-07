#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

# Adapter-stage training: freeze the pretrained motion prior and train
# trajectory adapters + energy branch + V9 RAG summary branch.
#
# Usage:
#   bash scripts/run_train_adapter_zero_traj_energy.sh
#
# Override these when needed:
#   CHECKPOINT=...
#   EXP_NAME=...
#   DATA_PATH=...
#   BATCH_SIZE=...

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
: "${EXP_NAME:=v10_adapter_zero_traj_energy}"
: "${DATA_PATH:=data/dunhuang_bvh/processed}"
: "${BATCH_SIZE:=24}"
: "${EPOCHS:=20}"
: "${SAVE_INTERVAL:=1}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 2
fi

export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"

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
