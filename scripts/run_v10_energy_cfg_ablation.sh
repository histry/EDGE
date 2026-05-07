#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

mkdir -p logs output/v10_eval

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT"
  exit 2
fi

export EDGE_TRAJECTORY_REP="${EDGE_TRAJECTORY_REP:-relative_abs_vel}"
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
export EDGE_RAG_SUMMARY_MODE="${EDGE_RAG_SUMMARY_MODE:-mean}"
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_FEATURES="${EDGE_UNIT_PRIOR_FEATURES:-upper}"
export EDGE_UNIT_PRIOR_STRENGTH="${EDGE_UNIT_PRIOR_STRENGTH:-0.012}"

for ENERGY in 0.35 0.55 0.75; do
  for SCALE in 0.00 0.50 1.00; do
    TAG="energycfg_e${ENERGY}_s${SCALE}"
    echo "===== ${TAG} ====="
    EDGE_DEFAULT_ENERGY="$ENERGY" \
    EDGE_AUDIO_ENERGY_AS_COND=0 \
    EDGE_MUSIC_TENSION_AS_ENERGY=0 \
    EDGE_ENERGY_CFG_SCALE="$SCALE" \
    EDGE_V10_SEARCH_METHOD=beam \
    EDGE_V10_OUT_PREFIX="output/v10_eval/${TAG}" \
    OUT_PATH="output/v10_eval/${TAG}.npy" \
    CHECKPOINT="$CHECKPOINT" \
    bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee "logs/${TAG}.log"
  done
done

grep -E "checkpoint_has_rag|V9 RAG summary attached|mode=|search_method=|manual_units=|Traceback|ERROR" -n logs/energycfg_*.log || true
