#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${V26_MUSIC_GLOB:?Set V26_MUSIC_GLOB, e.g. /path/to/music/**/*.wav}"
: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"

OUT="${V26_PLANNER_DATASET:-data/v26_whole_song_planner_dataset.npz}"

python tools/build_v26_planner_dataset.py \
  --music_glob "$V26_MUSIC_GLOB" \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --router_ckpt "$V26_ROUTER_CKPT" \
  --out_npz "$OUT" \
  --cache_dir "${V26_FEATURE_CACHE:-data/v26_music_features}" \
  --fps "${V26_FPS:-30}" \
  --min_phrase_seconds "${V26_MIN_PHRASE_SECONDS:-2.5}" \
  --max_phrase_seconds "${V26_MAX_PHRASE_SECONDS:-7.5}" \
  --boundary_quantile "${V26_BOUNDARY_QUANTILE:-0.68}" \
  --beat_snap_seconds "${V26_BEAT_SNAP_SECONDS:-0.35}" \
  --candidate_top_k "${V26_PSEUDO_TOP_K:-1200}" \
  --family_repeat_weight "${V26_FAMILY_REPEAT_WEIGHT:-0.55}" \
  --max_songs "${V26_MAX_SONGS:-0}"

echo "[PASS] V26 planner dataset: $OUT"
