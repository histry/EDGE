#!/usr/bin/env bash
set -euo pipefail

# Stage 1: no retraining. Build footstep-aware RAG DB, select mobile units,
# then blend lower-body units into an existing generated base motion.

RAG_DB=${RAG_DB:-data/dunhuang_choreo_unit_rag/index_footstep_u45_s15.npz}
CKPT=${CKPT:-runs/train_stage45/stage45_riskfix_v1_e504/weights/train-10.pt}
INPUT_DIR=${INPUT_DIR:-data/dunhuang_bvh/processed}
BASE_MOTION=${BASE_MOTION:-output/text_context_rag_ckpt_eval/dhw4_textctx_e4_hd6000.npy}
TARGET_TRAJ=${TARGET_TRAJ:-output/text_context_rag_ckpt_eval/dhw4_textctx_e4_hd6000_target_traj.npy}
OUT_PREFIX=${OUT_PREFIX:-output/footstep_stage1/dhw4_footstep}
MID_FRAMES=${MID_FRAMES:-25,50,75,100,125}

mkdir -p "$(dirname "$OUT_PREFIX")"

python build_choreo_unit_rag_db.py \
  --input_dir "$INPUT_DIR" \
  --out "$RAG_DB" \
  --checkpoint "$CKPT" \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_device ${TEXT_DEVICE:-cpu}

python footstep_aware_unit_selector.py \
  --rag_db "$RAG_DB" \
  --target_traj "$TARGET_TRAJ" \
  --out_prefix "$OUT_PREFIX" \
  --mid_frames "$MID_FRAMES" \
  --mobile_speed_threshold ${EDGE_RAG_MOBILE_SPEED_THRESHOLD:-0.010}

UNIT_FILES=$(python - <<PY
from pathlib import Path
prefix = Path("$OUT_PREFIX")
frames = "$MID_FRAMES".split(',')
print(','.join(str(p) for p in sorted(prefix.parent.glob(prefix.name + '_mid*_unit.npy'))))
PY
)

python segment_lower_body_compositor.py \
  --base "$BASE_MOTION" \
  --unit_files "$UNIT_FILES" \
  --frames "$MID_FRAMES" \
  --out "${OUT_PREFIX}_composited.npy" \
  --window ${EDGE_COMPOSITOR_WINDOW:-45} \
  --lower_strength ${EDGE_COMPOSITOR_LOWER_STRENGTH:-0.85} \
  --torso_strength ${EDGE_COMPOSITOR_TORSO_STRENGTH:-0.25} \
  --upper_strength ${EDGE_COMPOSITOR_UPPER_STRENGTH:-0.00} \
  --contact_strength ${EDGE_COMPOSITOR_CONTACT_STRENGTH:-0.75}

printf '\nStage 1 output: %s\n' "${OUT_PREFIX}_composited.npy"
