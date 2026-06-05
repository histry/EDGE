#!/usr/bin/env bash
set -Eeuo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA="${V22_TURN_DATASET:-data/v22_turn_pace_dataset.npz}"
OUT="${V22_TURN_TRAIN_OUT:-output/v22_turn_pace_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

python train_v22_turn_pace.py \
  --data "$DATA" \
  --out_dir "$OUT" \
  --epochs "${V22_TURN_EPOCHS:-320}" \
  --batch_size "${V22_TURN_BATCH_SIZE:-96}" \
  --lr "${V22_TURN_LR:-2e-4}" \
  --min_lr "${V22_TURN_MIN_LR:-2e-6}" \
  --weight_decay "${V22_TURN_WEIGHT_DECAY:-1e-4}" \
  --hidden_dim "${V22_TURN_HIDDEN:-256}" \
  --residual_scale "${V22_TURN_RESIDUAL_SCALE:-0.22}" \
  --dropout "${V22_TURN_DROPOUT:-0.10}" \
  --val_ratio "${V22_TURN_VAL_RATIO:-0.15}" \
  --num_workers "${V22_TURN_WORKERS:-4}" \
  --amp "${V22_TURN_AMP:-1}" \
  --patience "${V22_TURN_PATIENCE:-70}" \
  --lambda_recon "${V22_LAMBDA_RECON:-1.0}" \
  --lambda_velocity "${V22_LAMBDA_VELOCITY:-0.80}" \
  --lambda_acceleration "${V22_LAMBDA_ACCELERATION:-0.30}" \
  --lambda_yaw "${V22_LAMBDA_YAW:-0.65}" \
  --lambda_peak "${V22_LAMBDA_PEAK:-0.30}" \
  --lambda_unmasked "${V22_LAMBDA_UNMASKED:-0.50}" \
  --seed "${V22_SEED:-20260605}" \
  2>&1 | tee "$OUT/train.log"

printf '%s\n' "$OUT/checkpoints/best.pt" > "$OUT/BEST_TURN_PACE_CKPT.txt"
printf '\nDONE: %s\n' "$OUT"
