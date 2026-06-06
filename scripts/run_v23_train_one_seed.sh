#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${V23_SEED:?V23_SEED required}"
OUT="${V23_OUT_DIR:?V23_OUT_DIR required}"
DATA="${V23_DATASET:-data/v23_v2_3_slowaware_w120_d88.npz}"
mkdir -p "$OUT"

STAGE1="$OUT/stage1_duration"
STAGE2="$OUT/stage2_timewarp"

python train_v23_monotonic_duration.py \
  --data "$DATA" \
  --out_dir "$STAGE1" \
  --stage duration \
  --epochs "${V23_STAGE1_EPOCHS:-160}" \
  --batch_size "${V23_BATCH_SIZE:-48}" \
  --lr "${V23_STAGE1_LR:-8e-5}" \
  --min_lr "${V23_MIN_LR:-5e-7}" \
  --weight_decay "${V23_WEIGHT_DECAY:-5e-4}" \
  --hidden_dim "${V23_HIDDEN_DIM:-128}" \
  --dropout "${V23_DROPOUT:-0.18}" \
  --val_ratio "${V23_VAL_RATIO:-0.15}" \
  --split_seed "${V23_SPLIT_SEED:-20260620}" \
  --split_trials "${V23_SPLIT_TRIALS:-4096}" \
  --num_workers "${V23_WORKERS:-4}" \
  --amp "${V23_AMP:-1}" \
  --patience "${V23_STAGE1_PATIENCE:-35}" \
  --lr_patience "${V23_STAGE1_LR_PATIENCE:-8}" \
  --balanced_sampler "${V23_BALANCED_SAMPLER:-1}" \
  --lambda_bin "${V23_LAMBDA_BIN:-1.0}" \
  --lambda_residual "${V23_LAMBDA_RESIDUAL:-1.0}" \
  --lambda_relative "${V23_LAMBDA_RELATIVE:-1.2}" \
  --lambda_log_duration "${V23_LAMBDA_LOG_DURATION:-0.8}" \
  --lambda_linear_duration "${V23_LAMBDA_LINEAR_DURATION:-0.5}" \
  --lambda_duration_rank "${V23_LAMBDA_DURATION_RANK:-0.15}" \
  --lambda_edit "${V23_LAMBDA_EDIT:-0.30}" \
  --long_duration_weight "${V23_LONG_DURATION_WEIGHT:-1.25}" \
  --seed "$SEED"

STAGE1_BEST=$(cat "$STAGE1/BEST_V23_CKPT.txt")
python train_v23_monotonic_duration.py \
  --data "$DATA" \
  --out_dir "$STAGE2" \
  --stage timewarp \
  --init_checkpoint "$STAGE1_BEST" \
  --epochs "${V23_STAGE2_EPOCHS:-220}" \
  --batch_size "${V23_BATCH_SIZE:-48}" \
  --lr "${V23_STAGE2_LR:-1e-4}" \
  --min_lr "${V23_MIN_LR:-5e-7}" \
  --weight_decay "${V23_WEIGHT_DECAY:-5e-4}" \
  --hidden_dim "${V23_HIDDEN_DIM:-128}" \
  --dropout "${V23_DROPOUT:-0.18}" \
  --val_ratio "${V23_VAL_RATIO:-0.15}" \
  --split_seed "${V23_SPLIT_SEED:-20260620}" \
  --split_trials "${V23_SPLIT_TRIALS:-4096}" \
  --num_workers "${V23_WORKERS:-4}" \
  --amp "${V23_AMP:-1}" \
  --patience "${V23_STAGE2_PATIENCE:-50}" \
  --lr_patience "${V23_STAGE2_LR_PATIENCE:-10}" \
  --balanced_sampler "${V23_BALANCED_SAMPLER:-1}" \
  --teacher_forcing_start "${V23_TF_START:-1.0}" \
  --teacher_forcing_end "${V23_TF_END:-0.0}" \
  --teacher_forcing_decay_epochs "${V23_TF_DECAY_EPOCHS:-70}" \
  --lambda_tau "${V23_LAMBDA_TAU:-2.0}" \
  --lambda_duration_consistency "${V23_LAMBDA_DURATION_CONSISTENCY:-0.8}" \
  --lambda_motion "${V23_LAMBDA_MOTION:-0.9}" \
  --lambda_context "${V23_LAMBDA_CONTEXT:-0.30}" \
  --lambda_velocity "${V23_LAMBDA_VELOCITY:-0.35}" \
  --lambda_activity "${V23_LAMBDA_ACTIVITY:-0.25}" \
  --lambda_yaw "${V23_LAMBDA_YAW:-0.45}" \
  --lambda_peak_yaw "${V23_LAMBDA_PEAK_YAW:-0.18}" \
  --lambda_smooth "${V23_LAMBDA_SMOOTH:-0.05}" \
  --lambda_identity_tau "${V23_LAMBDA_IDENTITY_TAU:-0.55}" \
  --lambda_identity_motion "${V23_LAMBDA_IDENTITY_MOTION:-0.40}" \
  --seed "$SEED"

STAGE2_BEST=$(cat "$STAGE2/BEST_V23_CKPT.txt")
printf '%s\n' "$STAGE1_BEST" > "$OUT/BEST_DURATION_CKPT.txt"
printf '%s\n' "$STAGE2_BEST" > "$OUT/BEST_V23_CKPT.txt"
echo "SEED=$SEED"
echo "DURATION=$STAGE1_BEST"
echo "TIMEWARP=$STAGE2_BEST"
