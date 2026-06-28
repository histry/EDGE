#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
source scripts/v39_contact_stability_env.sh

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_v39_reprocess_motion.sh path/to/input.npy [path/to/output.npy]" >&2
  exit 2
fi

IN="$1"
if [[ $# -ge 2 ]]; then
  OUT="$2"
else
  base="${IN%.npy}"
  OUT="${base}_v39_contact_stable.npy"
fi
SUMMARY="${OUT%.npy}.motion_quality_postprocess.v39.json"

python tools/v34_motion_quality_postprocess.py \
  --motion "$IN" \
  --out "$OUT" \
  --summary_json "$SUMMARY" \
  --device "${V39_DEVICE:-cuda}"

python tools/v39_contact_stability_audit.py \
  --summary_json "$SUMMARY" \
  --out_json "${OUT%.npy}.contact_stability_audit.v39.json"

echo "[V39 SAVED] $OUT"
echo "[V39 SUMMARY] $SUMMARY"
