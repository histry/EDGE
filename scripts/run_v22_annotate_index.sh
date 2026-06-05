#!/usr/bin/env bash
set -Eeuo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INPUT_PREFIX="${V22_INPUT_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"
OUTPUT_PREFIX="${V22_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v22_turn_aware_event_index}"

python tools/annotate_v22_turn_index.py \
  --input_prefix "$INPUT_PREFIX" \
  --output_prefix "$OUTPUT_PREFIX" \
  --fps "${V22_FPS:-30}" \
  --min_peak_dps "${V22_INDEX_MIN_TURN_DPS:-35}"

printf '\nDONE: %s.json + %s.npz\n' "$OUTPUT_PREFIX" "$OUTPUT_PREFIX"
