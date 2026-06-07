#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="${V26_PLANNER_DATASET:-data/v26_whole_song_planner_dataset.npz}"
RUN="${V26_PLANNER_RUN:-output/v26_whole_song_planner_$(date +%Y%m%d_%H%M%S)}"

python train_v26_whole_song_planner.py \
  --data "$DATA" \
  --out_dir "$RUN" \
  --epochs "${V26_PLANNER_EPOCHS:-240}" \
  --batch_size "${V26_PLANNER_BATCH_SIZE:-16}" \
  --lr "${V26_PLANNER_LR:-2e-4}" \
  --weight_decay "${V26_PLANNER_WEIGHT_DECAY:-3e-4}" \
  --hidden_dim "${V26_PLANNER_HIDDEN_DIM:-128}" \
  --num_layers "${V26_PLANNER_LAYERS:-4}" \
  --num_heads "${V26_PLANNER_HEADS:-4}" \
  --dropout "${V26_PLANNER_DROPOUT:-0.15}" \
  --val_ratio "${V26_PLANNER_VAL_RATIO:-0.15}" \
  --patience "${V26_PLANNER_PATIENCE:-45}" \
  --seed "${V26_PLANNER_SEED:-20260620}" \
  --num_workers "${V26_NUM_WORKERS:-2}"

echo "[PASS] V26 planner run: $RUN"
