#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

RUN_ID="${RUN_ID:-v32_contact_inr_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

finish() {
  code=$?
  date > "$RUN_ROOT/finished_at.txt"
  if [[ $code -eq 0 ]]; then
    echo "SUCCESS" > "$RUN_ROOT/STATUS"
  else
    echo "FAILED exit_code=$code" > "$RUN_ROOT/STATUS"
  fi
}
trap finish EXIT

echo "[START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[COMMIT] $(git rev-parse HEAD 2>/dev/null || echo unknown)"

EVENT_CONTACT_CACHE="$RUN_ROOT/v33_event_contact_cache.npz"
EVENT_CONTACT_REPORT="$RUN_ROOT/v33_event_contact_cache.json"
DATA="$RUN_ROOT/v33_transition_dataset.npz"
AUDIT="$RUN_ROOT/v33_contact_dataset_audit.json"
TRAIN_DIR="$RUN_ROOT/v32_contact_inr_training"
MODE="${V32_SUPERVISION_MODE:-weak}"

# ------------------------------------------------------------------
# Supervision policy
#
# weak:
#   usable with the current 4225 cropped events. Real intra-event masks are the
#   main target; low-weight synthetic adjacent bridges regularise cross-event
#   conditioning. This is an engineering model, not proof of real transitions.
#
# strict:
#   requires a full-sequence source manifest and real cross-boundary masks.
# ------------------------------------------------------------------
BUILD_ARGS=()
INCLUDE_SYNTHETIC=1
if [[ "$MODE" == "strict" ]]; then
  : "${V32_SOURCE_MANIFEST:?Strict mode requires V32_SOURCE_MANIFEST}"
  [[ -f "$V32_SOURCE_MANIFEST" ]] || {
    echo "[ERROR] Missing V32_SOURCE_MANIFEST=$V32_SOURCE_MANIFEST" >&2
    exit 2
  }
  BUILD_ARGS+=(
    --source_manifest "$V32_SOURCE_MANIFEST"
    --allow_synthetic_adjacent 0
    --pseudo_pairs_per_event 0.0
    --require_real_boundary_count "${V32_REQUIRE_REAL_BOUNDARY_SAMPLES:-1000}"
    --require_real_boundary_ratio "${V32_REQUIRE_REAL_BOUNDARY_RATIO:-0.10}"
    --require_unique_real_boundary_count "${V32_REQUIRE_UNIQUE_BOUNDARIES:-250}"
  )
  [[ -n "${V32_FULL_MOTION_ROOT:-}" ]] &&
    BUILD_ARGS+=(--full_motion_root "$V32_FULL_MOTION_ROOT")
  INCLUDE_SYNTHETIC=0
else
  BUILD_ARGS+=(
    --allow_synthetic_adjacent "${V32_ALLOW_SYNTHETIC_ADJACENT:-1}"
    --pseudo_pairs_per_event "${V32_PSEUDO_PAIRS_PER_EVENT:-0.05}"
    --require_real_boundary_count 0
    --require_real_boundary_ratio 0.0
    --require_unique_real_boundary_count 0
  )
  if [[ -n "${V32_SOURCE_MANIFEST:-}" && -f "${V32_SOURCE_MANIFEST}" ]]; then
    BUILD_ARGS+=(--source_manifest "$V32_SOURCE_MANIFEST")
  fi
  [[ -n "${V32_FULL_MOTION_ROOT:-}" ]] &&
    BUILD_ARGS+=(--full_motion_root "$V32_FULL_MOTION_ROOT")
fi
[[ -n "${V32_EXTERNAL_PRIOR_NPZ:-}" ]] &&
  BUILD_ARGS+=(--external_prior_npz "$V32_EXTERNAL_PRIOR_NPZ")

python -m py_compile \
  tools/v29_motion_geometry.py \
  tools/v33_event_contacts.py \
  tools/build_v33_event_contact_cache.py \
  tools/v32_contact_inr.py \
  tools/v32_contact_losses.py \
  tools/v32_transition_quality.py \
  tools/v27_transition_diffusion.py \
  tools/build_v27_transition_diffusion_dataset.py \
  tools/audit_v32_contact_dataset.py \
  train_v27_transition_diffusion.py \
  tools/schedule_v32_whole_song.py \
  tools/evaluate_v32_contact_metrics.py \
  tools/evaluate_v30_frequency_metrics.py \
  tools/summarize_v32_transition_gate.py \
  render_from_npy.py

# ------------------------------------------------------------------
# 1. Reconstruct contacts once on each complete indexed event.
# Window-level contact relabeling is forbidden.
# ------------------------------------------------------------------
if [[ "${V33_BUILD_EVENT_CONTACT_CACHE:-1}" == "1" ]]; then
  python tools/build_v33_event_contact_cache.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_npz "$EVENT_CONTACT_CACHE" \
    --out_json "$EVENT_CONTACT_REPORT" \
    --device "${V33_CONTACT_DEVICE:-cuda:0}" \
    --batch_size "${V33_CONTACT_BATCH_SIZE:-96}" \
    --fps "${V26_FPS:-30}" \
    --target_rates "${V33_CONTACT_TARGET_RATES:-0.42,0.42,0.38,0.38}" \
    --transition_penalty "${V33_CONTACT_TRANSITION_PENALTY:-1.40}" \
    --min_run "${V33_CONTACT_MIN_RUN:-2}" \
    --max_gap "${V33_CONTACT_MAX_GAP:-2}" \
    --probability_temperature "${V33_CONTACT_TEMPERATURE:-0.20}" \
    --existing_contact_policy "${V33_EXISTING_CONTACT_POLICY:-auto}" \
    --min_overall_rate "${V33_MIN_CONTACT_RATE:-0.05}" \
    --max_overall_rate "${V33_MAX_CONTACT_RATE:-0.75}" \
    --max_all_four_rate "${V33_MAX_ALL_FOUR_RATE:-0.55}"
else
  : "${V33_EVENT_CONTACT_CACHE:?Set V33_EVENT_CONTACT_CACHE}"
  EVENT_CONTACT_CACHE="$V33_EVENT_CONTACT_CACHE"
  [[ -f "$EVENT_CONTACT_CACHE" ]] || {
    echo "[ERROR] Missing event contact cache: $EVENT_CONTACT_CACHE" >&2
    exit 2
  }
fi

# ------------------------------------------------------------------
# 2. Build transition samples by synchronously slicing event contacts.
# ------------------------------------------------------------------
if [[ "${V32_BUILD_DATASET:-1}" == "1" ]]; then
  python tools/build_v27_transition_diffusion_dataset.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_npz "$DATA" \
    --event_contact_cache "$EVENT_CONTACT_CACHE" \
    --require_event_contacts 1 \
    --assert_contact_consistency 1 \
    "${BUILD_ARGS[@]}" \
    --max_len "${V32_MAX_LEN:-120}" \
    --min_len "${V32_MIN_LEN:-8}" \
    --samples_per_event "${V32_SAMPLES_PER_EVENT:-6}" \
    --real_masks_per_boundary "${V32_REAL_MASKS_PER_BOUNDARY:-4}" \
    --source_pairs_per_event "${V32_SOURCE_PAIRS_PER_EVENT:-0.75}" \
    --condition_dropout "${V32_DATA_CONDITION_DROPOUT:-0.08}" \
    --pseudo_max_pose_deg "${V32_PSEUDO_MAX_POSE_DEG:-28}" \
    --pseudo_max_velocity_deg_s "${V32_PSEUDO_MAX_VELOCITY_DEG_S:-160}" \
    --pseudo_max_root_y "${V32_PSEUDO_MAX_ROOT_Y:-0.10}" \
    --pseudo_max_contact_jump "${V32_PSEUDO_MAX_CONTACT_JUMP:-0.25}" \
    --seed "${V32_SEED:-20260610}"
else
  : "${V32_DATASET:?Set V32_DATASET when V32_BUILD_DATASET=0}"
  DATA="$V32_DATASET"
fi

python tools/audit_v32_contact_dataset.py \
  --data "$DATA" \
  --out_json "$AUDIT" \
  --require_real_samples "${V32_REQUIRE_REAL_SAMPLES:-1000}" \
  --min_contact_rate "${V33_MIN_CONTACT_RATE:-0.05}" \
  --max_contact_rate "${V33_MAX_CONTACT_RATE:-0.75}" \
  --min_channel_rate "${V33_MIN_CHANNEL_RATE:-0.02}" \
  --max_channel_rate "${V33_MAX_CHANNEL_RATE:-0.80}" \
  --max_all_four_rate "${V33_MAX_ALL_FOUR_RATE:-0.55}" \
  --min_mean_switch_rate "${V33_MIN_SWITCH_RATE:-0.001}" \
  --require_overlap_comparisons "${V33_REQUIRE_OVERLAP_COMPARISONS:-1000}" \
  --require_event_level_pipeline 1

# ------------------------------------------------------------------
# 3. Train continuous INR, contact fine-tuning, then latent diffusion.
# ------------------------------------------------------------------
if [[ "${V32_TRAIN:-1}" == "1" ]]; then
  python train_v27_transition_diffusion.py \
    --data "$DATA" \
    --out_dir "$TRAIN_DIR" \
    --stage all \
    --include_synthetic "$INCLUDE_SYNTHETIC" \
    --ae_epochs "${V32_AE_EPOCHS:-220}" \
    --contact_epochs "${V32_CONTACT_EPOCHS:-90}" \
    --diffusion_epochs "${V32_DIFFUSION_EPOCHS:-320}" \
    --batch_size "${V32_AE_BATCH_SIZE:-40}" \
    --contact_batch_size "${V32_CONTACT_BATCH_SIZE:-32}" \
    --latent_batch_size "${V32_LATENT_BATCH_SIZE:-192}" \
    --lr "${V32_AE_LR:-1.5e-4}" \
    --contact_lr "${V32_CONTACT_LR:-4e-5}" \
    --diffusion_lr "${V32_DIFFUSION_LR:-2e-4}" \
    --latent_dim "${V32_LATENT_DIM:-128}" \
    --condition_dim "${V32_CONDITION_DIM:-256}" \
    --encoder_hidden "${V32_ENCODER_HIDDEN:-320}" \
    --inr_hidden "${V32_INR_HIDDEN:-320}" \
    --inr_layers "${V32_INR_LAYERS:-6}" \
    --fourier_bands "${V32_FOURIER_BANDS:-5}" \
    --diffusion_hidden "${V32_DIFFUSION_HIDDEN:-512}" \
    --diffusion_blocks "${V32_DIFFUSION_BLOCKS:-6}" \
    --diffusion_steps "${V32_DIFFUSION_STEPS:-100}" \
    --rotation_residual_scale "${V32_ROTATION_RESIDUAL_SCALE:-0.16}" \
    --root_y_residual_scale "${V32_ROOT_Y_RESIDUAL_SCALE:-0.045}" \
    --contact_logit_scale "${V32_CONTACT_LOGIT_SCALE:-3.0}" \
    --condition_dropout "${V32_CONDITION_DROPOUT:-0.10}" \
    --decoded_weight "${V32_DECODED_WEIGHT:-0.08}" \
    --decoded_batch_limit "${V32_DECODED_BATCH_LIMIT:-8}" \
    --val_ratio "${V32_VAL_RATIO:-0.12}" \
    --ae_patience "${V32_AE_PATIENCE:-55}" \
    --contact_patience "${V32_CONTACT_PATIENCE:-30}" \
    --diffusion_patience "${V32_DIFFUSION_PATIENCE:-70}" \
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
  export V27_TRANSITION_DIFFUSION_CKPT
  V27_TRANSITION_DIFFUSION_CKPT="$(
    cat "$TRAIN_DIR/BEST_V32_CONTACT_INR_CKPT.txt"
  )"
else
  : "${V27_TRANSITION_DIFFUSION_CKPT:?Set V27_TRANSITION_DIFFUSION_CKPT}"
fi

# ------------------------------------------------------------------
# 4. Whole-song generation.
# ------------------------------------------------------------------
export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
unset V26_MUSIC_GLOB || true
export V32_KEYS="${V32_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
export V32_ENABLE_EDGE_DAMPING="${V32_ENABLE_EDGE_DAMPING:-0}"
export V32_STRICT_LOCKED_WARP="${V32_STRICT_LOCKED_WARP:-1}"
export V32_CANDIDATES="${V32_CANDIDATES:-8}"
export V32_GUIDANCE="${V32_GUIDANCE:-1.0}"
export V32_INR_TRUST="${V32_INR_TRUST:-0.35}"
export V32_INFERENCE_STEPS="${V32_INFERENCE_STEPS:-40}"

if [[ "${V32_RUN_C2_BASELINE:-1}" == "1" ]]; then
  export V26_OUT_DIR="$RUN_ROOT/c2_baseline"
  export V27_TRANSITION_DIFFUSION=0
  bash scripts/run_v32_whole_song.sh
fi

export V26_OUT_DIR="$RUN_ROOT/v32_contact_inr"
export V27_TRANSITION_DIFFUSION=1
bash scripts/run_v32_whole_song.sh

# ------------------------------------------------------------------
# 5. Evaluate V32 and the deterministic baseline.
# ------------------------------------------------------------------
evaluate_directory() {
  local directory="$1"
  local label="$2"
  IFS=';' read -ra KEYS <<< "$V32_KEYS"
  for key in "${KEYS[@]}"; do
    local motion="$directory/${key}_v26.npy"
    local report="$directory/${key}_v26.schedule_report.json"
    local audio="test_music_bank/${key}.wav"
    [[ -f "$motion" ]] || {
      echo "[ERROR] Missing motion $motion" >&2
      return 3
    }
    python tools/evaluate_v26_long_dance.py \
      --motion "$motion" \
      --schedule_report "$report" \
      --out_json "$directory/${key}_${label}.long_dance.json"

    python tools/evaluate_v27_public_metrics.py \
      --motion "$motion" \
      --audio "$audio" \
      --index_json "$V26_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$directory/${key}_${label}.public_metrics.json"

    python tools/evaluate_v30_frequency_metrics.py \
      --motion "$motion" \
      --schedule_report "$report" \
      --index_json "$V26_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$directory/${key}_${label}.frequency_foot.json" \
      --out_png "$directory/${key}_${label}.dct_spectrum.png"

    python tools/evaluate_v32_contact_metrics.py \
      --motion "$motion" \
      --schedule_report "$report" \
      --out_json "$directory/${key}_${label}.contact_metrics.json"

    python tools/diagnose_v29_jitter.py \
      --motion "$motion" \
      --out_json "$directory/${key}_${label}.jitter.json"

    python render_from_npy.py \
      --motion "$motion" \
      --audio "$audio" \
      --output "$directory/${key}_${label}_scientific_fixed.mp4" \
      --camera_mode fixed \
      --render_smooth_window 1
  done
}

evaluate_directory "$RUN_ROOT/v32_contact_inr" "v32"
if [[ -d "$RUN_ROOT/c2_baseline" ]]; then
  evaluate_directory "$RUN_ROOT/c2_baseline" "c2"
fi

python tools/summarize_v32_transition_gate.py \
  --report_glob "$RUN_ROOT/v32_contact_inr/*_v26.schedule_report.json" \
  --out_json "$RUN_ROOT/v32_transition_gate_summary.json"

python - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "version": "v32_contact_inr_experiment_summary",
    "run_root": str(root),
    "supervision_mode": __import__("os").environ.get(
        "V32_SUPERVISION_MODE", "weak"
    ),
    "methods": {},
}
for method, directory, suffix in (
    ("v32_contact_inr", root / "v32_contact_inr", "v32"),
    ("c2_baseline", root / "c2_baseline", "c2"),
):
    if not directory.is_dir():
        continue
    rows = {}
    for key in ("dunhuangwu2", "dunhuangwu3", "dunhuangwu4"):
        row = {}
        files = {
            "public": directory / f"{key}_{suffix}.public_metrics.json",
            "frequency": directory / f"{key}_{suffix}.frequency_foot.json",
            "contact": directory / f"{key}_{suffix}.contact_metrics.json",
            "jitter": directory / f"{key}_{suffix}.jitter.json",
            "long": directory / f"{key}_{suffix}.long_dance.json",
        }
        for name, path in files.items():
            if path.is_file():
                row[name] = json.loads(path.read_text(encoding="utf-8"))
        rows[key] = row
    summary["methods"][method] = rows

gate = root / "v32_transition_gate_summary.json"
if gate.is_file():
    summary["gate"] = json.loads(gate.read_text(encoding="utf-8"))
audit = root / "v32_contact_dataset_audit.json"
if audit.is_file():
    summary["dataset_audit"] = json.loads(audit.read_text(encoding="utf-8"))

path = root / "v32_experiment_summary.json"
path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"[SAVED] {path}")
PY

echo "$RUN_ROOT" > output/LATEST_V32_CONTACT_INR_RUN.txt
echo "[DONE] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
