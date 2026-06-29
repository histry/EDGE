#!/usr/bin/env bash
set -euo pipefail
cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
source scripts/v41_beat_support_env.sh
MOTION="${1:?Usage: bash scripts/run_v41_reprocess_motion.sh path/to/*_v26.npy [out.npy]}"
OUT="${2:-${MOTION%.npy}_v41_beat_support_stable.npy}"
SUMMARY="${OUT%.npy}.motion_quality_postprocess.v41.json"
python tools/v34_motion_quality_postprocess.py \
  --motion "$MOTION" \
  --out "$OUT" \
  --summary_json "$SUMMARY" \
  --device "${V41_DEVICE:-cuda}"
python tools/v41_contact_stability_audit.py --summary_json "$SUMMARY" || true
echo "[V41 SAVED] $OUT"
echo "[V41 SUMMARY] $SUMMARY"
