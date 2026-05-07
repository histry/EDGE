#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
mkdir -p logs output/v10_eval
: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
for W in 0.00 0.15 0.30 0.50; do
  echo "===== Text Bridge weight ${W} ====="
  EDGE_TEXT_BRIDGE_WEIGHT="$W" \
  EDGE_V10_OUT_PREFIX="output/v10_eval/textbridge_w${W}" \
  OUT_PATH="output/v10_eval/textbridge_w${W}.npy" \
  CHECKPOINT="$CHECKPOINT" \
  bash scripts/run_v10_text_context_step4.sh 2>&1 | tee "logs/textbridge_w${W}.log"
done
grep -E "Text Bridge semantic|Text/Pose Context RAG attached|V10 Unified planner selected units|semantic_score|Traceback|ERROR" -n logs/textbridge_w*.log || true
