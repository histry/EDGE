#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

mkdir -p logs output/v10_eval

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 2
fi

# 1) ZeroInitTrajectoryAdapter practical setting:
#    remove global X/Z translation offset while preserving path shape/velocity.
export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"

# 2) Energy-conditioned CFG:
#    use audio-derived music energy/tension if generate_controlled.py does not
#    already provide cond["energy"].
export EDGE_AUDIO_ENERGY_AS_COND="${EDGE_AUDIO_ENERGY_AS_COND:-1}"
export EDGE_MUSIC_TENSION_AS_ENERGY="${EDGE_MUSIC_TENSION_AS_ENERGY:-1}"
export EDGE_AUDIO_ENERGY_SMOOTH="${EDGE_AUDIO_ENERGY_SMOOTH:-5}"
export EDGE_ENERGY_CFG_SCALE="${EDGE_ENERGY_CFG_SCALE:-0.65}"

# 3) Context RAG summary enhancement:
#    same 7D V9 summary, no checkpoint shape change.
export EDGE_CONTEXT_RAG_ENHANCE="${EDGE_CONTEXT_RAG_ENHANCE:-1}"
export EDGE_CONTEXT_RAG_SMOOTH="${EDGE_CONTEXT_RAG_SMOOTH:-5}"
export EDGE_CONTEXT_RAG_SCALE="${EDGE_CONTEXT_RAG_SCALE:-0.50}"

# V9 RAG summary token.
export EDGE_ENABLE_RAG_SUMMARY_TOKEN="${EDGE_ENABLE_RAG_SUMMARY_TOKEN:-1}"
export EDGE_RAG_SUMMARY_MODE="${EDGE_RAG_SUMMARY_MODE:-mean}"
export EDGE_RAG_SUMMARY_DIM="${EDGE_RAG_SUMMARY_DIM:-7}"

# Tension / expressiveness-aware ChoreoRAG.
export EDGE_TENSION_AWARE_PLANNER="${EDGE_TENSION_AWARE_PLANNER:-1}"
export EDGE_UNIT_MIN_ENERGY="${EDGE_UNIT_MIN_ENERGY:-0.45}"
export EDGE_UNIT_ENERGY_BONUS="${EDGE_UNIT_ENERGY_BONUS:-0.20}"
export EDGE_UNIT_MIN_EXPRESSIVENESS="${EDGE_UNIT_MIN_EXPRESSIVENESS:-0.40}"
export EDGE_UNIT_EXPRESSIVENESS_BONUS="${EDGE_UNIT_EXPRESSIVENESS_BONUS:-0.35}"
export EDGE_UNIT_HOMOGENEITY_WEIGHT="${EDGE_UNIT_HOMOGENEITY_WEIGHT:-0.10}"

# Soft unit prior: keep it weak and upper-body-safe.
export EDGE_UNIT_SOFT_PRIOR="${EDGE_UNIT_SOFT_PRIOR:-1}"
export EDGE_UNIT_PRIOR_DCT="${EDGE_UNIT_PRIOR_DCT:-1}"
export EDGE_UNIT_PRIOR_LOW_FREQ_K="${EDGE_UNIT_PRIOR_LOW_FREQ_K:-4}"
export EDGE_UNIT_PRIOR_FEATURES="${EDGE_UNIT_PRIOR_FEATURES:-upper}"
export EDGE_UNIT_PRIOR_STRENGTH="${EDGE_UNIT_PRIOR_STRENGTH:-0.012}"
export EDGE_UNIT_PRIOR_MAX_LEN="${EDGE_UNIT_PRIOR_MAX_LEN:-45}"

export EDGE_V10_SEARCH_METHOD="${EDGE_V10_SEARCH_METHOD:-beam}"
export EDGE_V10_TOP_K="${EDGE_V10_TOP_K:-64}"
export EDGE_V10_BEAM_WIDTH="${EDGE_V10_BEAM_WIDTH:-8}"
export EDGE_V10_OUT_PREFIX="${EDGE_V10_OUT_PREFIX:-output/v10_eval/v10_full_landing_step4}"
export OUT_PATH="${OUT_PATH:-${EDGE_V10_OUT_PREFIX}.npy}"

CHECKPOINT="$CHECKPOINT" \
bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee logs/v10_full_landing_step4.log
