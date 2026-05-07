#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

mkdir -p logs output/v10_eval

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
: "${EDGE_TEXT_QUERY:=敦煌飞天，上肢大幅舒展，高能量，流动转身，空间展开，亮相收束，强表现力}"

for MODE in rerank hybrid filter force_topk; do
  for W in 1.0; do
    TAG="textbridge_${MODE}_w${W}"
    echo "===== ${TAG} ====="
    EDGE_TEXT_BRIDGE_MODE="$MODE" \
    EDGE_TEXT_BRIDGE_TOP_K="${EDGE_TEXT_BRIDGE_TOP_K:-256}" \
    EDGE_TEXT_BRIDGE_WEIGHT="$W" \
    EDGE_TEXT_QUERY="$EDGE_TEXT_QUERY" \
    EDGE_V10_OUT_PREFIX="output/v10_eval/${TAG}" \
    OUT_PATH="output/v10_eval/${TAG}.npy" \
    CHECKPOINT="$CHECKPOINT" \
    bash scripts/run_v10_text_context_step4.sh 2>&1 | tee "logs/${TAG}.log"
  done
done

grep -E "Text Bridge candidate filtering|top_semantic|top_final|V10 Unified planner selected units|Text/Pose Context RAG attached|appended to decoder memory|Traceback|ERROR|RuntimeError" -n \
  logs/textbridge_*_w*.log || true
