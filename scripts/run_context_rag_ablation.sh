#!/usr/bin/env bash
set -euo pipefail

EDGE_ROOT="${EDGE_ROOT:-$(pwd)}"
cd "$EDGE_ROOT"
mkdir -p logs output/v10_context_ablation

: "${CHECKPOINT:?Set CHECKPOINT to a V10/Text-Context checkpoint path}"
: "${EDGE_V10_RAG_DB:?Set EDGE_V10_RAG_DB to the motion-unit RAG DB .npz path}"
: "${COMMON_ARGS:?Set COMMON_ARGS to the usual generate_v10_choreo.py args, quoted as one string}"

export EDGE_STRICT_EXPERIMENT_GUARD=1
export EDGE_STRICT_RUNTIME_PATCHES=1
export EDGE_EXPERIMENT_PROFILE=v10
export EDGE_V10_MAX_RAG_UNITS="${EDGE_V10_MAX_RAG_UNITS:-1000000}"

run_one() {
  local tag="$1"
  shift
  echo "===== ${tag} ====="
  env "$@" \
    EDGE_V10_OUT_PREFIX="output/v10_context_ablation/${tag}" \
    OUT_PATH="output/v10_context_ablation/${tag}.npy" \
    CHECKPOINT="$CHECKPOINT" \
    python generate_v10_choreo.py $COMMON_ARGS 2>&1 | tee "logs/${tag}.log"
}

# 1) no_context: clean RAG summary/unit-prior planner but no Text/Pose Context memory.
run_one no_context \
  EDGE_ENABLE_TEXT_CONTEXT_RAG=0

# 2) context: real selected retrieved units + text embeddings as decoder memory.
run_one context \
  EDGE_ENABLE_TEXT_CONTEXT_RAG=1

# 3) shuffled_context: IO patch should consume explicit context paths if supplied.
# Set EDGE_RAG_CONTEXT_UNIT_PATHS manually before this script to use a shuffled list.
if [ -n "${EDGE_SHUFFLED_RAG_CONTEXT_UNIT_PATHS:-}" ]; then
  run_one shuffled_context \
    EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
    EDGE_RAG_CONTEXT_UNIT_PATHS="$EDGE_SHUFFLED_RAG_CONTEXT_UNIT_PATHS"
else
  echo "Skip shuffled_context: set EDGE_SHUFFLED_RAG_CONTEXT_UNIT_PATHS to a comma-separated shuffled unit list."
fi

# 4) wrong_text: keep motion context but alter text query/filter semantics.
run_one wrong_text \
  EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
  EDGE_TEXT_QUERY="现代街舞，强节拍，大幅跳跃，非敦煌风格"

