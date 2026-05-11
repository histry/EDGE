#!/usr/bin/env bash
set -euo pipefail

# Stage 3 practical decoupled validation:
# lower/root/contact from lower motion + upper style from Text/Pose Context RAG motion.

LOWER_MOTION=${LOWER_MOTION:?set LOWER_MOTION to Stage-1/Stage-2 lower motion .npy}
UPPER_MOTION=${UPPER_MOTION:?set UPPER_MOTION to upper/TextContext motion .npy}
OUT=${OUT:-output/decoupled/dhw4_decoupled_merge.npy}

python decoupled_upper_lower_merge.py \
  --lower_motion "$LOWER_MOTION" \
  --upper_motion "$UPPER_MOTION" \
  --out "$OUT" \
  --torso_from ${TORSO_FROM:-blend} \
  --torso_blend ${TORSO_BLEND:-0.35}

printf '\nStage 3 output: %s\n' "$OUT"
