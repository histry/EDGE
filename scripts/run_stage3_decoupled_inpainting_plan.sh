#!/usr/bin/env bash
set -Eeuo pipefail

# Deterministic decoupled merge validation. Full diffusion inpainting training
# uses EDGE_DECOUPLED_TRAIN=1 in a later architecture branch; this script verifies
# the upper/lower separation target first.

: "${LOWER_MOTION:?Set LOWER_MOTION=lower/root/contact .npy}"
: "${UPPER_MOTION:?Set UPPER_MOTION=upper/style .npy}"
OUT=${OUT:-output/decoupled/dhw4_decoupled_upper_lower.npy}

python -u decoupled_upper_lower_merge.py \
  --lower_motion "$LOWER_MOTION" \
  --upper_motion "$UPPER_MOTION" \
  --out "$OUT" \
  --torso_blend ${EDGE_DECOUPLED_TORSO_BLEND:-0.35} \
  "$@"
