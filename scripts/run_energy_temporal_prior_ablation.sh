#!/usr/bin/env bash
# Standard V10 Energy/Expressiveness-aware ChoreoRAG + temporal prior ablation.
#
# Required env:
#   CHECKPOINT=/path/to/checkpoint.pt
#   EDGE_V10_RAG_DB=/path/to/index_u45_s15_e10_expr_loco.npz
#   COMMON_ARGS="--music ... --start_pose ... --end_pose ... --trajectory '0,0;0.5,0.7;-0.3,1.2;0,1.6' --out output/tmp.npy ..."
#
# Optional env:
#   EDGE_ROOT=/path/to/EDGE
#   OUT_DIR=output/v10_energy_temporal_ablation
#   EDGE_RAG_STATS_CACHE=/path/to/index_stats.npz
#   RUN_RENDER=0/1
#   RENDER_CMD="python render_choreorag_results.py --input"
#
set -euo pipefail
EDGE_ROOT="${EDGE_ROOT:-$(pwd)}"
cd "$EDGE_ROOT"

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/checkpoint.pt}"
: "${EDGE_V10_RAG_DB:?Set EDGE_V10_RAG_DB=/path/to/rag_db.npz}"
: "${COMMON_ARGS:?Set COMMON_ARGS='arguments passed to generate_v10_choreo.py, except --checkpoint/--out can be overridden'}"

OUT_DIR="${OUT_DIR:-output/v10_energy_temporal_ablation}"
mkdir -p "$OUT_DIR" logs

# Strict contracts keep baseline/V10 comparisons auditable if you use edge_experiment_guard.py.
export EDGE_EXPERIMENT_PROFILE="${EDGE_EXPERIMENT_PROFILE:-v10_text_context}"
export EDGE_STRICT_EXPERIMENT_GUARD="${EDGE_STRICT_EXPERIMENT_GUARD:-1}"
export EDGE_STRICT_RUNTIME_PATCHES="${EDGE_STRICT_RUNTIME_PATCHES:-1}"

# Use full DB unless the caller explicitly overrides this for smoke tests.
export EDGE_V10_MAX_RAG_UNITS="${EDGE_V10_MAX_RAG_UNITS:-1000000}"
export EDGE_V10_MODE="${EDGE_V10_MODE:-auto_multiunit}"
export EDGE_V10_SEARCH_METHOD="${EDGE_V10_SEARCH_METHOD:-beam}"
export EDGE_V10_TOP_K="${EDGE_V10_TOP_K:-64}"
export EDGE_V10_BEAM_WIDTH="${EDGE_V10_BEAM_WIDTH:-8}"

# V9/V10 conditions.
export EDGE_ENABLE_RAG_SUMMARY_TOKEN="${EDGE_ENABLE_RAG_SUMMARY_TOKEN:-1}"
export EDGE_RAG_SUMMARY_MODE="${EDGE_RAG_SUMMARY_MODE:-mean}"
export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"

run_case() {
  local tag="$1"
  shift
  echo "===== ${tag} ====="
  local out_prefix="${OUT_DIR}/${tag}"
  local out_path="${out_prefix}.npy"

  (
    export EDGE_V10_OUT_PREFIX="$out_prefix"
    export OUT_PATH="$out_path"
    "$@"
    # shellcheck disable=SC2086
    python generate_v10_choreo.py \
      --checkpoint "$CHECKPOINT" \
      --out "$out_path" \
      $COMMON_ARGS
  ) 2>&1 | tee "logs/${tag}.log"

  # Collect standard artifacts if names follow generate_controlled.py conventions.
  mkdir -p "${OUT_DIR}/${tag}_assets"
  cp -f "${out_prefix}"*"v10_plan.json" "${OUT_DIR}/${tag}_assets/" 2>/dev/null || true
  cp -f "${out_prefix}"*"score_parts.json" "${OUT_DIR}/${tag}_assets/" 2>/dev/null || true
  cp -f "${out_prefix}"*.npy "${OUT_DIR}/${tag}_assets/" 2>/dev/null || true
  cp -f "${OUT_DIR}/${tag}"*.json "${OUT_DIR}/${tag}_assets/" 2>/dev/null || true
  cp -f "${OUT_DIR}/${tag}"*.mp4 "${OUT_DIR}/${tag}_assets/" 2>/dev/null || true

  if [[ "${RUN_RENDER:-0}" == "1" && -n "${RENDER_CMD:-}" ]]; then
    # Example: RENDER_CMD="python render_choreorag_results.py --input"
    ${RENDER_CMD} "$out_path" 2>&1 | tee -a "logs/${tag}.log" || true
  fi
}

# 1) Energy-aware off: current planner-like behavior, no temporal unit prior.
run_case "energy_aware_off" \
  env \
    EDGE_UNIT_MIN_ENERGY=0.0 \
    EDGE_UNIT_MIN_EXPRESSIVENESS=0.0 \
    EDGE_UNIT_BAN_LOW_ENERGY=0 \
    EDGE_UNIT_ENERGY_BONUS=0.0 \
    EDGE_UNIT_EXPRESSIVENESS_BONUS=0.0 \
    EDGE_V10_TEXT_SCORE_W="${EDGE_V10_TEXT_SCORE_W_OFF:-0.15}" \
    EDGE_UNIT_SOFT_PRIOR=0 \
    EDGE_UNIT_PRIOR_TEMPORAL=0

# 2) Energy filter only.
run_case "energy_filter_only" \
  env \
    EDGE_UNIT_MIN_ENERGY="${ABL_MIN_ENERGY:-0.35}" \
    EDGE_UNIT_MIN_EXPRESSIVENESS=0.0 \
    EDGE_UNIT_BAN_LOW_ENERGY=1 \
    EDGE_UNIT_LOW_ENERGY_THRESHOLD="${ABL_LOW_ENERGY_THRESHOLD:-0.25}" \
    EDGE_UNIT_ENERGY_BONUS="${ABL_ENERGY_BONUS:-0.15}" \
    EDGE_UNIT_EXPRESSIVENESS_BONUS=0.0 \
    EDGE_V10_TEXT_SCORE_W="${EDGE_V10_TEXT_SCORE_W:-0.15}" \
    EDGE_UNIT_SOFT_PRIOR=0 \
    EDGE_UNIT_PRIOR_TEMPORAL=0

# 3) Energy + expressiveness filter.
run_case "energy_expr_filter" \
  env \
    EDGE_UNIT_MIN_ENERGY="${ABL_MIN_ENERGY:-0.35}" \
    EDGE_UNIT_MIN_EXPRESSIVENESS="${ABL_MIN_EXPRESSIVENESS:-0.40}" \
    EDGE_UNIT_BAN_LOW_ENERGY=1 \
    EDGE_UNIT_LOW_ENERGY_THRESHOLD="${ABL_LOW_ENERGY_THRESHOLD:-0.25}" \
    EDGE_UNIT_ENERGY_BONUS="${ABL_ENERGY_BONUS:-0.15}" \
    EDGE_UNIT_EXPRESSIVENESS_BONUS="${ABL_EXPR_BONUS:-0.25}" \
    EDGE_V10_TEXT_SCORE_W="${EDGE_V10_TEXT_SCORE_W:-0.15}" \
    EDGE_UNIT_SOFT_PRIOR=0 \
    EDGE_UNIT_PRIOR_TEMPORAL=0

# 4) Energy + expressiveness + stronger text scoring.
run_case "energy_expr_text" \
  env \
    EDGE_UNIT_MIN_ENERGY="${ABL_MIN_ENERGY:-0.35}" \
    EDGE_UNIT_MIN_EXPRESSIVENESS="${ABL_MIN_EXPRESSIVENESS:-0.40}" \
    EDGE_UNIT_BAN_LOW_ENERGY=1 \
    EDGE_UNIT_LOW_ENERGY_THRESHOLD="${ABL_LOW_ENERGY_THRESHOLD:-0.25}" \
    EDGE_UNIT_ENERGY_BONUS="${ABL_ENERGY_BONUS:-0.15}" \
    EDGE_UNIT_EXPRESSIVENESS_BONUS="${ABL_EXPR_BONUS:-0.25}" \
    EDGE_V10_TEXT_SCORE_W="${ABL_TEXT_W:-0.30}" \
    EDGE_UNIT_SOFT_PRIOR=0 \
    EDGE_UNIT_PRIOR_TEMPORAL=0

# 5) Energy + expressiveness + text + temporal DCT unit prior.
run_case "energy_expr_text_temporal_prior" \
  env \
    EDGE_UNIT_MIN_ENERGY="${ABL_MIN_ENERGY:-0.35}" \
    EDGE_UNIT_MIN_EXPRESSIVENESS="${ABL_MIN_EXPRESSIVENESS:-0.40}" \
    EDGE_UNIT_BAN_LOW_ENERGY=1 \
    EDGE_UNIT_LOW_ENERGY_THRESHOLD="${ABL_LOW_ENERGY_THRESHOLD:-0.25}" \
    EDGE_UNIT_ENERGY_BONUS="${ABL_ENERGY_BONUS:-0.15}" \
    EDGE_UNIT_EXPRESSIVENESS_BONUS="${ABL_EXPR_BONUS:-0.25}" \
    EDGE_V10_TEXT_SCORE_W="${ABL_TEXT_W:-0.30}" \
    EDGE_UNIT_SOFT_PRIOR=1 \
    EDGE_UNIT_PRIOR_TEMPORAL=1 \
    EDGE_UNIT_PRIOR_WINDOW="${ABL_PRIOR_WINDOW:-41}" \
    EDGE_UNIT_PRIOR_FEATURES="${ABL_PRIOR_FEATURES:-upper+torso}" \
    EDGE_UNIT_PRIOR_DCT=1 \
    EDGE_UNIT_PRIOR_LOW_FREQ_K="${ABL_LOW_FREQ_K:-4}" \
    EDGE_UNIT_PRIOR_STRENGTH="${ABL_PRIOR_STRENGTH:-0.012}"

echo "✅ Ablation complete. Outputs: ${OUT_DIR}; logs: logs/energy_*.log"
