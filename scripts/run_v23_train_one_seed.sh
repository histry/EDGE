#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

SEED="${V23_SEED:?V23_SEED required}"
OUT="${V23_OUT_DIR:?V23_OUT_DIR required}"
DATA="${V23_DATASET:-data/v23_v2_4_slowaware_w120_d88_9k.npz}"
mkdir -p "$OUT"

STAGE1="$OUT/stage1_duration"
STAGE2="$OUT/stage2_timewarp"
STAGE1_OVERRIDE="${V23_STAGE1_CHECKPOINT_OVERRIDE:-}"

if [[ -n "$STAGE1_OVERRIDE" ]]; then
  if [[ ! -s "$STAGE1_OVERRIDE" ]]; then
    echo "[ERROR] V23_STAGE1_CHECKPOINT_OVERRIDE not found: $STAGE1_OVERRIDE" >&2
    exit 2
  fi
  STAGE1_BEST=$(readlink -f "$STAGE1_OVERRIDE")
  mkdir -p "$STAGE1"
  printf '%s\n' "$STAGE1_BEST" > "$STAGE1/BEST_V23_CKPT.txt"
  echo "[REUSE] Stage-1 checkpoint: $STAGE1_BEST"
else
  python train_v23_monotonic_duration.py \
    --data "$DATA" \
    --out_dir "$STAGE1" \
    --stage duration \
    --epochs "${V23_STAGE1_EPOCHS:-240}" \
    --batch_size "${V23_STAGE1_EVENT_BATCH:-40}" \
    --lr "${V23_STAGE1_LR:-3e-5}" \
    --min_lr "${V23_MIN_LR:-3e-7}" \
    --warmup_epochs "${V23_STAGE1_WARMUP:-10}" \
    --weight_decay "${V23_WEIGHT_DECAY:-1e-3}" \
    --hidden_dim "${V23_HIDDEN_DIM:-96}" \
    --dropout "${V23_DROPOUT:-0.24}" \
    --slow_feature_span "${V23_SLOW_FEATURE_SPAN:-10}" \
    --ordinal_blend "${V23_ORDINAL_BLEND:-0.82}" \
    --val_ratio "${V23_VAL_RATIO:-0.15}" \
    --split_seed "${V23_SPLIT_SEED:-20260620}" \
    --split_trials "${V23_SPLIT_TRIALS:-4096}" \
    --num_workers "${V23_WORKERS:-4}" \
    --event_num_workers "${V23_EVENT_WORKERS:-0}" \
    --amp "${V23_AMP:-1}" \
    --patience "${V23_STAGE1_PATIENCE:-60}" \
    --balanced_sampler "${V23_BALANCED_SAMPLER:-1}" \
    --ema_decay "${V23_STAGE1_EMA_DECAY:-0.995}" \
    --condition_noise_std "${V23_CONDITION_NOISE_STD:-0.01}" \
    --lambda_ordinal "${V23_LAMBDA_ORDINAL:-1.0}" \
    --lambda_residual "${V23_LAMBDA_RESIDUAL:-0.8}" \
    --lambda_relative "${V23_LAMBDA_RELATIVE:-1.0}" \
    --lambda_log_duration "${V23_LAMBDA_LOG_DURATION:-0.6}" \
    --lambda_direct "${V23_LAMBDA_DIRECT:-0.35}" \
    --lambda_underestimate "${V23_LAMBDA_UNDERESTIMATE:-0.8}" \
    --lambda_duration_rank "${V23_LAMBDA_DURATION_RANK:-0.18}" \
    --lambda_pair_duration "${V23_LAMBDA_PAIR_DURATION:-0.9}" \
    --lambda_pair_distribution "${V23_LAMBDA_PAIR_DISTRIBUTION:-0.35}" \
    --lambda_moment "${V23_LAMBDA_MOMENT:-0.30}" \
    --lambda_edit "${V23_LAMBDA_EDIT:-0.25}" \
    --long_duration_weight "${V23_LONG_DURATION_WEIGHT:-1.5}" \
    --seed "$SEED"
  STAGE1_BEST=$(cat "$STAGE1/BEST_V23_CKPT.txt")
fi

# Re-evaluate with the final continuous-duration prediction.  This makes old
# v2.4 checkpoints compatible with the corrected v2.5 scientific gate.
STAGE1_EVAL="$OUT/STAGE1_CONTINUOUS_EVALUATION.json"
python tools/evaluate_v23_stage1_gate.py \
  --data "$DATA" \
  --checkpoint "$STAGE1_BEST" \
  --out "$STAGE1_EVAL" \
  --batch_size "${V23_EVAL_BATCH_SIZE:-64}" \
  --val_ratio "${V23_VAL_RATIO:-0.15}" \
  --split_seed "${V23_SPLIT_SEED:-20260620}" \
  --split_trials "${V23_SPLIT_TRIALS:-4096}"

set +e
python - "$STAGE1_EVAL" "$OUT" <<'PY'
import json, os, sys
path, out = sys.argv[1], sys.argv[2]
report = json.load(open(path, 'r', encoding='utf-8'))
m = report['metrics']
core = {
    'event_mae': float(m.get('event_duration_mae_frames', 1e9)) <= float(os.getenv('V23_GATE_EVENT_MAE', '7.0')),
    'event_long_mae': float(m.get('event_duration_long_mae', 1e9)) <= float(os.getenv('V23_GATE_LONG_MAE', '8.0')),
    'event_corr': float(m.get('event_duration_correlation', -1.0)) >= float(os.getenv('V23_GATE_EVENT_CORR', '0.90')),
    'continuous_within_one_bin': float(m.get('event_duration_continuous_within_one_bin_accuracy', 0.0)) >= float(os.getenv('V23_GATE_CONTINUOUS_WITHIN1', '0.95')),
    'p90_calibration': float(m.get('event_duration_p90_error', 1e9)) <= float(os.getenv('V23_GATE_P90_ERROR', '8.0')),
    'quantile_calibration': float(m.get('event_duration_quantile_calibration_mae', 1e9)) <= float(os.getenv('V23_GATE_QUANTILE_MAE', '7.0')),
}
auxiliary = {
    'ordinal_within_one_bin': float(m.get('event_duration_ordinal_within_one_bin_accuracy', 0.0)) >= float(os.getenv('V23_WARN_ORDINAL_WITHIN1', '0.60')),
    'edit_balanced_accuracy': float(m.get('edit_optimal_balanced_accuracy', 0.0)) >= float(os.getenv('V23_WARN_EDIT_BALANCED', '0.72')),
    'edit_auroc': float(m.get('edit_auroc', 0.0)) >= float(os.getenv('V23_WARN_EDIT_AUROC', '0.72')),
}
mode = os.getenv('V23_GATE_MODE', 'duration_core').strip().lower()
if mode not in {'duration_core', 'strict'}:
    raise RuntimeError(f'Unsupported V23_GATE_MODE={mode}')
passed = all(core.values()) and (all(auxiliary.values()) if mode == 'strict' else True)
gate = {
    'version': 'V23-v2.5-continuous-gate',
    'evaluation': path,
    'mode': mode,
    'metrics': m,
    'core_checks': core,
    'auxiliary_checks': auxiliary,
    'passed': passed,
}
open(os.path.join(out, 'STAGE1_GATE.json'), 'w', encoding='utf-8').write(json.dumps(gate, indent=2))
print('STAGE1_CORE_GATE', json.dumps(core), 'passed=', all(core.values()))
print('STAGE1_AUXILIARY', json.dumps(auxiliary))
print('STAGE1_GATE_MODE', mode, 'passed=', passed)
if os.getenv('V23_STAGE1_REQUIRE', '1') == '1' and not passed:
    open(os.path.join(out, 'STAGE1_REJECTED.txt'), 'w').write(report['checkpoint'] + '\n')
    sys.exit(42)
PY
GATE_CODE=$?
set -e
if [[ "$GATE_CODE" == "42" ]]; then
  echo "[SKIP] Stage 1 failed the continuous-duration core gate; Stage 2 was not started."
  exit 0
elif [[ "$GATE_CODE" != "0" ]]; then
  echo "[ERROR] Stage-1 gate evaluation failed with code $GATE_CODE"
  exit "$GATE_CODE"
fi

python train_v23_monotonic_duration.py \
  --data "$DATA" \
  --out_dir "$STAGE2" \
  --stage timewarp \
  --init_checkpoint "$STAGE1_BEST" \
  --epochs "${V23_STAGE2_EPOCHS:-240}" \
  --batch_size "${V23_STAGE2_BATCH_SIZE:-40}" \
  --lr "${V23_STAGE2_LR:-8e-5}" \
  --min_lr "${V23_MIN_LR:-3e-7}" \
  --warmup_epochs "${V23_STAGE2_WARMUP:-8}" \
  --weight_decay "${V23_WEIGHT_DECAY:-1e-3}" \
  --hidden_dim "${V23_HIDDEN_DIM:-96}" \
  --dropout "${V23_DROPOUT:-0.24}" \
  --slow_feature_span "${V23_SLOW_FEATURE_SPAN:-10}" \
  --ordinal_blend "${V23_ORDINAL_BLEND:-0.82}" \
  --val_ratio "${V23_VAL_RATIO:-0.15}" \
  --split_seed "${V23_SPLIT_SEED:-20260620}" \
  --split_trials "${V23_SPLIT_TRIALS:-4096}" \
  --num_workers "${V23_WORKERS:-4}" \
  --event_num_workers 0 \
  --amp "${V23_AMP:-1}" \
  --patience "${V23_STAGE2_PATIENCE:-60}" \
  --balanced_sampler "${V23_BALANCED_SAMPLER:-1}" \
  --ema_decay "${V23_STAGE2_EMA_DECAY:-0.997}" \
  --teacher_forcing_start "${V23_TF_START:-1.0}" \
  --teacher_forcing_end "${V23_TF_END:-0.0}" \
  --teacher_forcing_decay_epochs "${V23_TF_DECAY_EPOCHS:-90}" \
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
