#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${V23_SEED:?V23_SEED required}"
OUT="${V23_OUT_DIR:?V23_OUT_DIR required}"

python train_v23_monotonic_duration.py \
  --data "${V23_DATASET:-data/v23_v2_natural_duration_dataset.npz}" \
  --out_dir "$OUT" \
  --epochs "${V23_EPOCHS:-600}" \
  --batch_size "${V23_BATCH_SIZE:-64}" \
  --lr "${V23_LR:-2e-4}" \
  --min_lr "${V23_MIN_LR:-1e-6}" \
  --weight_decay "${V23_WEIGHT_DECAY:-1e-4}" \
  --hidden_dim "${V23_HIDDEN_DIM:-256}" \
  --dropout "${V23_DROPOUT:-0.10}" \
  --duration_min_frames "${V23_MIN_TARGET_DURATION:-10}" \
  --duration_max_frames "${V23_MAX_TARGET_DURATION:-56}" \
  --num_workers "${V23_WORKERS:-4}" \
  --amp "${V23_AMP:-1}" \
  --patience "${V23_PATIENCE:-100}" \
  --balanced_sampler "${V23_BALANCED_SAMPLER:-1}" \
  --lambda_tau "${V23_LAMBDA_TAU:-2.0}" \
  --lambda_duration "${V23_LAMBDA_DURATION:-1.2}" \
  --lambda_duration_linear "${V23_LAMBDA_DURATION_LINEAR:-0.35}" \
  --lambda_duration_rank "${V23_LAMBDA_DURATION_RANK:-0.15}" \
  --lambda_duration_consistency "${V23_LAMBDA_DURATION_CONSISTENCY:-0.8}" \
  --lambda_motion "${V23_LAMBDA_MOTION:-0.8}" \
  --lambda_context "${V23_LAMBDA_CONTEXT:-0.35}" \
  --lambda_velocity "${V23_LAMBDA_VELOCITY:-0.30}" \
  --lambda_activity "${V23_LAMBDA_ACTIVITY:-0.20}" \
  --lambda_yaw "${V23_LAMBDA_YAW:-0.35}" \
  --lambda_peak_yaw "${V23_LAMBDA_PEAK_YAW:-0.12}" \
  --lambda_smooth "${V23_LAMBDA_SMOOTH:-0.06}" \
  --lambda_identity_tau "${V23_LAMBDA_IDENTITY_TAU:-0.45}" \
  --lambda_identity_motion "${V23_LAMBDA_IDENTITY_MOTION:-0.35}" \
  --lambda_edit "${V23_LAMBDA_EDIT:-0.15}" \
  --lambda_endpoint "${V23_LAMBDA_ENDPOINT:-0.20}" \
  --seed "$SEED"
