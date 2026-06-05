#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INDEX_PREFIX="${V21_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"
STYLE_CKPT="${V21_STYLE_RANKER_CKPT:?Set V21_STYLE_RANKER_CKPT to a trained checkpoint}"

python tools/rescore_v21_index_with_style_ranker.py \
  --index_json "${INDEX_PREFIX}.json" \
  --index_npz "${INDEX_PREFIX}.npz" \
  --checkpoint "$STYLE_CKPT" \
  --blend "${V21_STYLE_RANKER_BLEND:-0.70}"
