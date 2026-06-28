#!/usr/bin/env bash
set -euo pipefail
cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
source scripts/v40_floor_aware_env.sh
IN="${1:-data/v34_source_aware/v34_shared_event_index_source_aware.json}"
OUT="${2:-data/v40_source_aware/v34_shared_event_index_source_aware_floorclean.json}"
AUDIT="${OUT%.json}.audit.json"
python tools/v40_native_floor_audit.py \
  --index_json "$IN" \
  --out_json "$OUT" \
  --audit_json "$AUDIT" \
  --search_root data \
  --search_root output \
  --filter_mode "${V40_NATIVE_FLOOR_FILTER_MODE:-quality}"
echo "$OUT" > output/LATEST_V40_NATIVE_FLOOR_INDEX.txt
echo "[V40 NATIVE FLOOR INDEX] $OUT"
echo "[V40 NATIVE FLOOR AUDIT] $AUDIT"
