#!/usr/bin/env bash
set -euo pipefail
cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
source scripts/v40_floor_aware_env.sh
MOTION="${1:?Usage: bash scripts/run_v40_reprocess_motion.sh path/to/*_v26.npy}"
OUT="${2:-${MOTION%.npy}_v40_floor_aware.npy}"
SUMMARY="${OUT%.npy}.motion_quality_postprocess.v40.json"
python tools/v34_motion_quality_postprocess.py \
  --motion "$MOTION" \
  --out "$OUT" \
  --summary_json "$SUMMARY" \
  --device "${V40_DEVICE:-cuda}"
python tools/v40_contact_stability_audit.py --summary_json "$SUMMARY" || true
echo "[V40 SAVED] $OUT"
echo "[V40 SUMMARY] $SUMMARY"
