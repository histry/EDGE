#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON to the original V21 JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"
: "${V33_EVENT_CONTACT_CACHE:?Set V33_EVENT_CONTACT_CACHE}"

RUN_ID="${RUN_ID:-v34_c3_contact_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1
finish() {
  code=$?
  date > "$RUN_ROOT/finished_at.txt"
  if [[ $code -eq 0 ]]; then
    echo SUCCESS > "$RUN_ROOT/STATUS"
  else
    echo "FAILED exit_code=$code" > "$RUN_ROOT/STATUS"
  fi
}
trap finish EXIT

echo "[START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[COMMIT] $(git rev-parse HEAD 2>/dev/null || echo unknown)"

run_stage() {
  local name="$1"
  shift
  echo "[STAGE START] $name $(date)"
  "$@"
  echo "[STAGE DONE] $name $(date)"
}

check_motion_outputs() {
  local directory="$1"
  local label="$2"
  IFS=';' read -ra CHECK_KEYS <<< "$V32_KEYS"
  for key in "${CHECK_KEYS[@]}"; do
    local motion="$directory/${key}_v26.npy"
    local report="$directory/${key}_v26.schedule_report.json"
    [[ -f "$motion" ]] || {
      echo "[STAGE ERROR] $label missing motion: $motion" >&2
      return 3
    }
    [[ -f "$report" ]] || {
      echo "[STAGE ERROR] $label missing schedule report: $report" >&2
      return 3
    }
  done
}

python -m py_compile \
  tools/v34_boundary_dynamics.py \
  tools/v34_boundary_inpainting.py \
  tools/v34_warp_aware_retrieval.py \
  tools/build_v34_contact_event_library.py \
  tools/calibrate_v34_boundary_thresholds.py \
  tools/v32_contact_inr.py \
  tools/v32_transition_quality.py \
  tools/v27_transition_diffusion.py \
  tools/v34_motion_quality_postprocess.py \
  tools/schedule_v34_whole_song.py \
  tools/evaluate_v34_boundary_dynamics.py \
  train_v27_transition_diffusion.py

# ------------------------------------------------------------------
# 1. Back-inject event-level contacts into a mirrored V34 event library.
# ------------------------------------------------------------------
V34_LIBRARY_DIR="${V34_LIBRARY_DIR:-$RUN_ROOT/v34_event_library}"
V34_INDEX_JSON="${V34_INDEX_JSON:-$RUN_ROOT/v34_shared_event_index.json}"
if [[ "${V34_BUILD_EVENT_LIBRARY:-1}" == "1" || ! -f "$V34_INDEX_JSON" ]]; then
  python tools/build_v34_contact_event_library.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --event_contact_cache "$V33_EVENT_CONTACT_CACHE" \
    --out_dir "$V34_LIBRARY_DIR" \
    --out_json "$V34_INDEX_JSON" \
    --overwrite "${V34_OVERWRITE_EVENT_LIBRARY:-0}"
fi
export V34_INDEX_JSON

# ------------------------------------------------------------------
# 2. Calibrate absolute gates from natural intra-event boundaries.
# User-supplied V34_MAX_* variables always take precedence.
# ------------------------------------------------------------------
CALIBRATION_JSON="$RUN_ROOT/v34_boundary_thresholds.json"
CALIBRATION_ENV="$RUN_ROOT/v34_boundary_thresholds.env"
if [[ "${V34_CALIBRATE_THRESHOLDS:-1}" == "1" ]]; then
  python tools/calibrate_v34_boundary_thresholds.py \
    --index_json "$V34_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_json "$CALIBRATION_JSON" \
    --out_env "$CALIBRATION_ENV" \
    --fps "${V26_FPS:-30}" \
    --samples_per_event "${V34_CALIBRATION_SAMPLES_PER_EVENT:-4}" \
    --quantile "${V34_CALIBRATION_QUANTILE:-0.995}" \
    --multiplier "${V34_CALIBRATION_MULTIPLIER:-2.0}"
  # shellcheck disable=SC1090
  source "$CALIBRATION_ENV"
  export V34_MAX_BOUNDARY_JERK="${V34_MAX_BOUNDARY_JERK:-$V34_CALIBRATED_MAX_BOUNDARY_JERK}"
  export V34_MAX_BOUNDARY_ANGULAR_JERK="${V34_MAX_BOUNDARY_ANGULAR_JERK:-$V34_CALIBRATED_MAX_BOUNDARY_ANGULAR_JERK}"
  export V34_MAX_ENTRY_ROTATION_STEP_RAD="${V34_MAX_ENTRY_ROTATION_STEP_RAD:-$V34_CALIBRATED_MAX_ENTRY_ROTATION_STEP_RAD}"
  export V34_MAX_EXIT_ROTATION_STEP_RAD="${V34_MAX_EXIT_ROTATION_STEP_RAD:-$V34_CALIBRATED_MAX_EXIT_ROTATION_STEP_RAD}"
  export V34_MAX_ENTRY_FK_JUMP="${V34_MAX_ENTRY_FK_JUMP:-$V34_CALIBRATED_MAX_ENTRY_FK_JUMP}"
  export V34_MAX_EXIT_FK_JUMP="${V34_MAX_EXIT_FK_JUMP:-$V34_CALIBRATED_MAX_EXIT_FK_JUMP}"
  export V34_MAX_EXIT_ACCELERATION="${V34_MAX_EXIT_ACCELERATION:-$V34_CALIBRATED_MAX_EXIT_ACCELERATION}"
fi

# ------------------------------------------------------------------
# 3. Reuse the synchronised V33 dataset or rebuild it externally.
# V34 boundary derivatives are derived on the fly from start/target/end.
# ------------------------------------------------------------------
V34_DATASET="${V34_DATASET:-data/v33_transition_dataset.npz}"
[[ -f "$V34_DATASET" ]] || {
  echo "[ERROR] Missing V34_DATASET=$V34_DATASET" >&2
  exit 2
}

# ------------------------------------------------------------------
# 4. Optional full retraining around the septic C3 base.
# The parameter shapes remain checkpoint-compatible, but a paper run should
# retrain because the residual target changed from quintic/C2 to septic/C3.
# ------------------------------------------------------------------
TRAIN_DIR="$RUN_ROOT/v34_contact_inr_training"
if [[ "${V34_TRAIN:-1}" == "1" ]]; then
  python train_v27_transition_diffusion.py \
    --data "$V34_DATASET" \
    --out_dir "$TRAIN_DIR" \
    --stage all \
    --include_synthetic "${V34_INCLUDE_SYNTHETIC:-1}" \
    --ae_epochs "${V34_AE_EPOCHS:-180}" \
    --contact_epochs "${V34_CONTACT_EPOCHS:-70}" \
    --diffusion_epochs "${V34_DIFFUSION_EPOCHS:-280}" \
    --batch_size "${V34_AE_BATCH_SIZE:-36}" \
    --contact_batch_size "${V34_CONTACT_BATCH_SIZE:-28}" \
    --latent_batch_size "${V34_LATENT_BATCH_SIZE:-160}" \
    --lr "${V34_AE_LR:-1.2e-4}" \
    --contact_lr "${V34_CONTACT_LR:-3e-5}" \
    --diffusion_lr "${V34_DIFFUSION_LR:-1.8e-4}" \
    --latent_dim "${V32_LATENT_DIM:-128}" \
    --condition_dim "${V32_CONDITION_DIM:-256}" \
    --encoder_hidden "${V32_ENCODER_HIDDEN:-320}" \
    --inr_hidden "${V32_INR_HIDDEN:-320}" \
    --inr_layers "${V32_INR_LAYERS:-6}" \
    --fourier_bands "${V32_FOURIER_BANDS:-5}" \
    --diffusion_hidden "${V32_DIFFUSION_HIDDEN:-512}" \
    --diffusion_blocks "${V32_DIFFUSION_BLOCKS:-6}" \
    --diffusion_steps "${V32_DIFFUSION_STEPS:-100}" \
    --rotation_residual_scale "${V32_ROTATION_RESIDUAL_SCALE:-0.14}" \
    --root_y_residual_scale "${V32_ROOT_Y_RESIDUAL_SCALE:-0.040}" \
    --contact_logit_scale "${V32_CONTACT_LOGIT_SCALE:-3.0}" \
    --w_endpoint_acceleration "${V34_W_ENDPOINT_ACCELERATION:-0.45}" \
    --w_endpoint_jerk "${V34_W_ENDPOINT_JERK:-0.20}" \
    --decoded_weight "${V34_DECODED_WEIGHT:-0.10}" \
    --decoded_batch_limit "${V34_DECODED_BATCH_LIMIT:-8}" \
    --val_ratio "${V32_VAL_RATIO:-0.12}" \
    --ae_patience "${V34_AE_PATIENCE:-45}" \
    --contact_patience "${V34_CONTACT_PATIENCE:-25}" \
    --diffusion_patience "${V34_DIFFUSION_PATIENCE:-60}" \
    --num_workers "${V32_NUM_WORKERS:-2}" \
    --amp "${V32_AMP:-1}" \
    --fps "${V26_FPS:-30}" \
    --w_contact_bce "${V32_W_CONTACT_BCE:-0.20}" \
    --w_contact_skate "${V32_W_CONTACT_SKATE:-0.60}" \
    --w_contact_height "${V32_W_CONTACT_HEIGHT:-0.25}" \
    --w_foot_penetration "${V32_W_FOOT_PENETRATION:-0.50}" \
    --w_swing_clearance "${V32_W_SWING_CLEARANCE:-0.06}" \
    --w_contact_temporal "${V32_W_CONTACT_TEMPORAL:-0.04}" \
    --w_contact_binary "${V32_W_CONTACT_BINARY:-0.01}" \
    --seed "${V32_SEED:-20260610}"
  V27_TRANSITION_DIFFUSION_CKPT="$(cat "$TRAIN_DIR/BEST_V32_CONTACT_INR_CKPT.txt")"
else
  : "${V27_TRANSITION_DIFFUSION_CKPT:?Set checkpoint when V34_TRAIN=0}"
fi
export V27_TRANSITION_DIFFUSION_CKPT

# ------------------------------------------------------------------
# 5. Strict whole-song generation.
# ------------------------------------------------------------------
export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
unset V26_MUSIC_GLOB || true
export V32_KEYS="${V32_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
export V26_MIN_TIME_WARP="${V26_MIN_TIME_WARP:-0.82}"
export V26_MAX_TIME_WARP="${V26_MAX_TIME_WARP:-1.30}"
export V32_STRICT_LOCKED_WARP=1
export V32_MAX_WARP_VIOLATIONS=0
export V34_WARP_HARD_PRUNE=1
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export V34_DEFER_UNSAFE_BOUNDARY="${V34_DEFER_UNSAFE_BOUNDARY:-1}"
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE="${V34_HANDSHAKE_FALLBACK_ON_UNSAFE:-1}"
export V34_POST_HANDSHAKE_REPAIR="${V34_POST_HANDSHAKE_REPAIR:-0}"
export V34_EXIT_HANDSHAKE="${V34_EXIT_HANDSHAKE:-1}"
export V34_EXIT_HANDSHAKE_FRAMES="${V34_EXIT_HANDSHAKE_FRAMES:-10}"
export V32_CANDIDATES="${V32_CANDIDATES:-8}"
export V32_INR_TRUST="${V32_INR_TRUST:-0.25}"
export V32_INFERENCE_STEPS="${V32_INFERENCE_STEPS:-40}"

if [[ "${V34_RUN_SEPTIC_BASELINE:-1}" == "1" ]]; then
  export V26_OUT_DIR="$RUN_ROOT/septic_handshake_baseline"
  export V27_TRANSITION_DIFFUSION=0
  run_stage septic_handshake_baseline bash scripts/run_v34_whole_song.sh
  check_motion_outputs "$V26_OUT_DIR" septic_handshake_baseline
fi

export V26_OUT_DIR="$RUN_ROOT/v34_contact_inr"
export V27_TRANSITION_DIFFUSION=1
run_stage v34_contact_inr bash scripts/run_v34_whole_song.sh
check_motion_outputs "$V26_OUT_DIR" v34_contact_inr

# ------------------------------------------------------------------
# 6. Evaluation and scientific render.
# ------------------------------------------------------------------
evaluate_directory() {
  local directory="$1"
  local label="$2"
  IFS=';' read -ra KEYS <<< "$V32_KEYS"
  for key in "${KEYS[@]}"; do
    local raw_motion="$directory/${key}_v26.npy"
    local motion="$raw_motion"
    local report="$directory/${key}_v26.schedule_report.json"
    local audio="test_music_bank/${key}.wav"
    [[ -f "$raw_motion" ]] || { echo "[ERROR] Missing $raw_motion" >&2; return 3; }
    [[ -f "$report" ]] || { echo "[ERROR] Missing $report" >&2; return 3; }

    if [[ "${V34_MOTION_QUALITY_POSTPROCESS:-1}" == "1" ]]; then
      motion="$directory/${key}_v26_motion_quality.npy"
      python tools/v34_motion_quality_postprocess.py \
        --motion "$raw_motion" \
        --out "$motion" \
        --summary_json "$directory/${key}_${label}.motion_quality_postprocess.json" \
        --contact_lock "${V34_CONTACT_LOCK_POSTPROCESS:-1}" \
        --contact_threshold "${V34_CONTACT_LOCK_THRESHOLD:-0.65}" \
        --min_contact_frames "${V34_CONTACT_LOCK_MIN_FRAMES:-8}" \
        --contact_lock_strength "${V34_CONTACT_LOCK_STRENGTH:-0.85}" \
        --contact_smooth_window "${V34_CONTACT_LOCK_SMOOTH_WINDOW:-11}" \
        --max_root_correction "${V34_CONTACT_LOCK_MAX_ROOT_CORRECTION:-0.18}" \
        --smooth "${V34_OUTPUT_SMOOTH:-1}" \
        --rotation_smooth_window "${V34_OUTPUT_ROT_SMOOTH_WINDOW:-3}" \
        --root_y_smooth_window "${V34_OUTPUT_ROOT_Y_SMOOTH_WINDOW:-5}" \
        --smooth_strength "${V34_OUTPUT_SMOOTH_STRENGTH:-0.35}"
    fi

    python tools/evaluate_v26_long_dance.py \
      --motion "$motion" --schedule_report "$report" \
      --out_json "$directory/${key}_${label}.long_dance.json"
    python tools/evaluate_v27_public_metrics.py \
      --motion "$motion" --audio "$audio" \
      --index_json "$V34_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$directory/${key}_${label}.public_metrics.json"
    python tools/evaluate_v30_frequency_metrics.py \
      --motion "$motion" --schedule_report "$report" \
      --index_json "$V34_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$directory/${key}_${label}.frequency_foot.json" \
      --out_png "$directory/${key}_${label}.dct_spectrum.png"
    python tools/evaluate_v32_contact_metrics.py \
      --motion "$motion" --schedule_report "$report" \
      --out_json "$directory/${key}_${label}.contact_metrics.json"
    python tools/diagnose_v29_jitter.py \
      --motion "$motion" \
      --out_json "$directory/${key}_${label}.jitter.json"
    python tools/evaluate_v34_boundary_dynamics.py \
      --motion "$motion" --schedule_report "$report" \
      --out_json "$directory/${key}_${label}.boundary_v34.json" \
      --max_boundary_jerk "${V34_MAX_BOUNDARY_JERK:-5000}" \
      --max_exit_rotation_step_rad "${V34_MAX_EXIT_ROTATION_STEP_RAD:-0.12}" \
      --max_exit_fk_jump "${V34_MAX_EXIT_FK_JUMP:-0.040}"
    python render_from_npy.py \
      --motion "$motion" --audio "$audio" \
      --output "$directory/${key}_${label}_scientific_fixed.mp4" \
      --camera_mode fixed --render_smooth_window 1
  done
}

run_stage evaluate_v34_contact_inr evaluate_directory "$RUN_ROOT/v34_contact_inr" v34
if [[ -d "$RUN_ROOT/septic_handshake_baseline" ]]; then
  run_stage evaluate_septic_baseline evaluate_directory "$RUN_ROOT/septic_handshake_baseline" septic
fi

run_stage summarize_v34_transition_gate \
  python tools/summarize_v32_transition_gate.py \
    --report_glob "$RUN_ROOT/v34_contact_inr/*_v26.schedule_report.json" \
    --out_json "$RUN_ROOT/v34_transition_gate_summary.json"

echo "$RUN_ROOT" > output/LATEST_V34_RUN.txt
echo "[DONE] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
