#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${EDGE_ROOT:-$(pwd)}"
cd "$EDGE_ROOT"

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/train-4.pt}"
: "${EDGE_V10_RAG_DB:?Set EDGE_V10_RAG_DB=/path/to/rag_db.npz}"
: "${EDGE_RAG_STATS_CACHE:?Set EDGE_RAG_STATS_CACHE=/path/to/stats_cache.npz}"

MUSIC="${MUSIC:-test_music_bank/dunhuangwu2.wav}"
START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_end.npy}"
TRAJECTORY="${TRAJECTORY:-0,0;0.5,0.7;-0.3,1.2;0,1.6}"
FEATURE_TYPE="${FEATURE_TYPE:-hybrid}"
AUDIO_DIM="${AUDIO_DIM:-803}"
OUTPUT_DIR="${OUTPUT_DIR:-output/v10_energy_temporal_ablation}"
LOG_DIR="${LOG_DIR:-logs/v10_energy_temporal_ablation}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

test -f "$CHECKPOINT" || { echo "Missing CHECKPOINT: $CHECKPOINT"; exit 2; }
test -f "$EDGE_V10_RAG_DB" || { echo "Missing EDGE_V10_RAG_DB: $EDGE_V10_RAG_DB"; exit 2; }
test -f "$EDGE_RAG_STATS_CACHE" || { echo "Missing EDGE_RAG_STATS_CACHE: $EDGE_RAG_STATS_CACHE"; exit 2; }
test -f "$MUSIC" || { echo "Missing MUSIC: $MUSIC"; exit 2; }
test -f "$START_POSE" || { echo "Missing START_POSE: $START_POSE"; exit 2; }
test -f "$END_POSE" || { echo "Missing END_POSE: $END_POSE"; exit 2; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export EDGE_EXPERIMENT_PROFILE="${EDGE_EXPERIMENT_PROFILE:-v10_text_context}"
export EDGE_STRICT_EXPERIMENT_GUARD="${EDGE_STRICT_EXPERIMENT_GUARD:-1}"
export EDGE_STRICT_RUNTIME_PATCHES="${EDGE_STRICT_RUNTIME_PATCHES:-1}"
export EDGE_ENABLE_RAG_SUMMARY_TOKEN="${EDGE_ENABLE_RAG_SUMMARY_TOKEN:-1}"
export EDGE_ENABLE_TEXT_CONTEXT_RAG="${EDGE_ENABLE_TEXT_CONTEXT_RAG:-1}"
export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"
export EDGE_RAG_SUMMARY_MODE="${EDGE_RAG_SUMMARY_MODE:-mean}"

export EDGE_V10_MODE="${EDGE_V10_MODE:-auto_multiunit}"
export EDGE_V10_SEARCH_METHOD="${EDGE_V10_SEARCH_METHOD:-beam}"
export EDGE_V10_TOP_K="${EDGE_V10_TOP_K:-64}"
export EDGE_V10_BEAM_WIDTH="${EDGE_V10_BEAM_WIDTH:-8}"
export EDGE_V10_MAX_RAG_UNITS="${EDGE_V10_MAX_RAG_UNITS:-1000000}"

unset EDGE_V10_MANUAL_UNITS || true
unset EDGE_V10_MANUAL_MID_POSES || true
unset EDGE_V10_MANUAL_MID_FRAMES || true

export EDGE_TEXT_BRIDGE_WEIGHT="${EDGE_TEXT_BRIDGE_WEIGHT:-0.0}"

ABL_MIN_ENERGY="${ABL_MIN_ENERGY:-0.35}"
ABL_MIN_EXPRESSIVENESS="${ABL_MIN_EXPRESSIVENESS:-0.40}"
ABL_ENERGY_BONUS="${ABL_ENERGY_BONUS:-0.15}"
ABL_EXPR_BONUS="${ABL_EXPR_BONUS:-0.25}"
ABL_TEXT_W="${ABL_TEXT_W:-0.30}"
ABL_PRIOR_STRENGTH="${ABL_PRIOR_STRENGTH:-0.012}"
ABL_PRIOR_FEATURES="${ABL_PRIOR_FEATURES:-upper+torso}"
ABL_LOW_FREQ_K="${ABL_LOW_FREQ_K:-4}"
ABL_PRIOR_WINDOW="${ABL_PRIOR_WINDOW:-41}"

run_case() {
  local name="$1"
  local min_energy="$2"
  local min_expr="$3"
  local ban_low="$4"
  local energy_bonus="$5"
  local expr_bonus="$6"
  local text_w="$7"
  local prior_on="$8"

  echo "===== CASE: ${name} ====="
  export EDGE_V10_OUT_PREFIX="${OUTPUT_DIR}/${name}"
  export OUT_PATH="${OUTPUT_DIR}/${name}.npy"

  export EDGE_UNIT_MIN_ENERGY="$min_energy"
  export EDGE_UNIT_MIN_EXPRESSIVENESS="$min_expr"
  export EDGE_UNIT_BAN_LOW_ENERGY="$ban_low"
  export EDGE_UNIT_ENERGY_BONUS="$energy_bonus"
  export EDGE_UNIT_EXPRESSIVENESS_BONUS="$expr_bonus"
  export EDGE_V10_TEXT_SCORE_W="$text_w"

  if [[ "$prior_on" == "1" ]]; then
    export EDGE_UNIT_SOFT_PRIOR=1
    export EDGE_UNIT_PRIOR_TEMPORAL=1
    export EDGE_UNIT_PRIOR_DCT=1
    export EDGE_UNIT_PRIOR_LOW_FREQ_K="$ABL_LOW_FREQ_K"
    export EDGE_UNIT_PRIOR_STRENGTH="$ABL_PRIOR_STRENGTH"
    export EDGE_UNIT_PRIOR_FEATURES="$ABL_PRIOR_FEATURES"
    export EDGE_UNIT_PRIOR_WINDOW="$ABL_PRIOR_WINDOW"
    export EDGE_UNIT_PRIOR_MAX_LEN="$ABL_PRIOR_WINDOW"
  else
    export EDGE_UNIT_SOFT_PRIOR=0
    export EDGE_UNIT_PRIOR_TEMPORAL=0
  fi

  python generate_v10_choreo.py     --checkpoint "$CHECKPOINT"     --music "$MUSIC"     --start_pose "$START_POSE"     --end_pose "$END_POSE"     --trajectory "$TRAJECTORY"     --feature_type "$FEATURE_TYPE"     --audio_dim "$AUDIO_DIM"     --out "$OUT_PATH"     2>&1 | tee "${LOG_DIR}/${name}.log"
}

run_case "energy_aware_off"                "0.0"             "0.0"                    "0" "0.0"              "0.0"            "0.0"        "0"
run_case "energy_filter_only"              "$ABL_MIN_ENERGY" "0.0"                    "1" "0.0"              "0.0"            "0.0"        "0"
run_case "energy_expr_filter"              "$ABL_MIN_ENERGY" "$ABL_MIN_EXPRESSIVENESS" "1" "0.0"              "0.0"            "0.0"        "0"
run_case "energy_expr_text"                "$ABL_MIN_ENERGY" "$ABL_MIN_EXPRESSIVENESS" "1" "$ABL_ENERGY_BONUS" "$ABL_EXPR_BONUS" "$ABL_TEXT_W" "0"
run_case "energy_expr_text_temporal_prior" "$ABL_MIN_ENERGY" "$ABL_MIN_EXPRESSIVENESS" "1" "$ABL_ENERGY_BONUS" "$ABL_EXPR_BONUS" "$ABL_TEXT_W" "1"

echo "===== Summary grep ====="
grep -E "CASE|V10 Energy/Expressiveness|selected units|unit soft prior|Text/Pose Context|V9 RAG summary|Traceback|ERROR|OutOfMemory" -n "$LOG_DIR"/*.log || true
