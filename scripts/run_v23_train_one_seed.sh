#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${V23_SEED:?V23_SEED required}"
OUT="${V23_OUT_DIR:?V23_OUT_DIR required}"

python train_v23_monotonic_duration.py \
  --data "${V23_DATASET:-data/v23_monotonic_duration_dataset.npz}" \
  --out_dir "$OUT" \
  --epochs "${V23_EPOCHS:-700}" \
  --batch_size "${V23_BATCH_SIZE:-64}" \
  --lr "${V23_LR:-2e-4}" \
  --min_lr "${V23_MIN_LR:-1e-6}" \
  --weight_decay "${V23_WEIGHT_DECAY:-1e-4}" \
  --hidden_dim "${V23_HIDDEN_DIM:-256}" \
  --dropout "${V23_DROPOUT:-0.10}" \
  --num_workers "${V23_WORKERS:-4}" \
  --amp "${V23_AMP:-1}" \
  --patience "${V23_PATIENCE:-120}" \
  --seed "$SEED"
