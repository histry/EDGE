#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INPUT_DB="${V21_INPUT_DB:-data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_prototype.json}"
OUTPUT_PREFIX="${V21_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"

python tools/build_v21_shared_event_index.py \
  --input_db "$INPUT_DB" \
  --output_prefix "$OUTPUT_PREFIX" \
  --max_events "${V21_MAX_EVENTS:-7000}" \
  --min_style_percentile "${V21_MIN_STYLE_PERCENTILE:-10}" \
  --min_quality "${V21_MIN_QUALITY:-0.0}" \
  --min_safety "${V21_MIN_SAFETY:-0.0}" \
  --family_span "${V21_FAMILY_SPAN:-600}" \
  --mmr_dim "${V21_MMR_DIM:-64}"

printf '\nDONE: %s.json + %s.npz\n' "$OUTPUT_PREFIX" "$OUTPUT_PREFIX"
