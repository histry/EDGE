#!/usr/bin/env bash
set -euo pipefail

# Four-way Text/Pose Context RAG ablation.
# Put this under scripts/run_context_rag_ablation.sh and run from EDGE repo root.
#
# Required env vars:
#   CHECKPOINT
#   START_POSE
#   END_POSE
#   AUDIO
#   OUT_DIR
#   EDGE_V10_RAG_DB or RAG_DB
#
# Optional:
#   TRAJECTORY="0,0;0.5,0.7;-0.3,1.2;0,1.6"
#   FEATURE_TYPE=hybrid
#   SAMPLER=ddim
#   EXTRA_ARGS="..."

: "${CHECKPOINT:?Set CHECKPOINT=/path/to/train-4.pt}"
: "${START_POSE:?Set START_POSE=/path/to/start.npy}"
: "${END_POSE:?Set END_POSE=/path/to/end.npy}"
: "${AUDIO:?Set AUDIO=/path/to/audio.wav}"
: "${OUT_DIR:?Set OUT_DIR=output/context_ablation}"

TRAJECTORY=${TRAJECTORY:-"0,0;0.5,0.7;-0.3,1.2;0,1.6"}
FEATURE_TYPE=${FEATURE_TYPE:-hybrid}
SAMPLER=${SAMPLER:-ddim}
EXTRA_ARGS=${EXTRA_ARGS:-}
mkdir -p "$OUT_DIR"

COMMON_ENV=(
  EDGE_RUN_MODE=formal
  EDGE_EXPERIMENT_PROFILE=v10
  EDGE_STRICT_EXPERIMENT_GUARD=1
  EDGE_STRICT_RUNTIME_PATCHES=1
  EDGE_ENABLE_TEXT_CONTEXT_RAG=1
  EDGE_TEXT_CONTEXT_REQUIRED=1
  EDGE_UNIT_SOFT_PRIOR=1
  EDGE_UNIT_PRIOR_REQUIRED=1
  EDGE_UNIT_PRIOR_TEMPORAL=1
  EDGE_UNIT_PRIOR_DCT=1
  EDGE_UNIT_PRIOR_LOW_FREQ_K=4
  EDGE_UNIT_PRIOR_FEATURES=upper+torso
  EDGE_UNIT_PRIOR_STRENGTH=0.006
)

run_case() {
  local name="$1"
  local mode="$2"
  local out_prefix="$OUT_DIR/$name"
  echo "===== Context RAG ablation: $name / mode=$mode ====="
  env "${COMMON_ENV[@]}" \
    EDGE_RAG_CONTEXT_MODE="$mode" \
    EDGE_V10_OUT_PREFIX="$out_prefix" \
    EDGE_RAG_CONTEXT_REPORT_JSON="${out_prefix}_context_report.json" \
    EDGE_UNIT_PRIOR_REPORT_JSON="${out_prefix}_unit_prior_report.json" \
    python generate_v10_choreo.py \
      --checkpoint "$CHECKPOINT" \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --audio "$AUDIO" \
      --feature_type "$FEATURE_TYPE" \
      --trajectory "$TRAJECTORY" \
      --sampler "$SAMPLER" \
      --out "${out_prefix}.npy" \
      $EXTRA_ARGS

  if [[ -f "${out_prefix}_raw.npy" && -f "${out_prefix}.npy" && -f "${out_prefix}_target_traj.npy" ]]; then
    python scripts/eval_generated_motions.py \
      --raw_motion "${out_prefix}_raw.npy" \
      --final_motion "${out_prefix}.npy" \
      --target_traj "${out_prefix}_target_traj.npy" \
      --meta "${out_prefix}_meta.json" \
      --out "${out_prefix}_eval.json" \
      --formal || true
  fi
}

run_case "no_context" "no_context"
run_case "context" "normal"
run_case "shuffled_context" "shuffled"
run_case "wrong_text" "wrong_text"

python scripts/analyze_context_rag_ablation.py --eval_dir "$OUT_DIR" --out "$OUT_DIR/context_ablation_summary.json" || true
