#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON to the V21 shared index JSON}"
: "${V26_INDEX_NPZ:?Set V26_INDEX_NPZ to the V21 shared index NPZ}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT to the V23-v2.5 Stage-2 checkpoint}"

OUT_NPZ="${V26_DURATION_INDEX_NPZ:-data/v26_duration_augmented_event_index.npz}"
OUT_JSON="${V26_DURATION_INDEX_JSON:-data/v26_duration_augmented_event_index.json}"

python tools/build_v26_duration_index.py \
  --index_json "$V26_INDEX_JSON" \
  --index_npz "$V26_INDEX_NPZ" \
  --v23_checkpoint "$V26_V23_CKPT" \
  --out_npz "$OUT_NPZ" \
  --out_json "$OUT_JSON" \
  --min_turn_angle "${V26_MIN_TURN_ANGLE:-10}" \
  --min_peak_dps "${V26_MIN_PEAK_DPS:-14}"

echo "[PASS] V26 duration index: $OUT_NPZ"
