#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

AUDIO_GLOB="${V21_AUDIO_GLOB:-test_music_bank/*.wav}"
OUT_DIR="${V21_MUSIC_FEATURE_DIR:-data/v21_music_features}"
mkdir -p "$OUT_DIR"

shopt -s nullglob
for AUDIO in $AUDIO_GLOB; do
  NAME="$(basename "$AUDIO")"
  STEM="${NAME%.*}"
  python tools/extract_v21_music_features.py \
    --audio "$AUDIO" \
    --out_npy "$OUT_DIR/${STEM}_v21_music.npy" \
    --out_json "$OUT_DIR/${STEM}_v21_music.json" \
    --num_frames "${V21_NUM_FRAMES:-150}"
done

printf '\nDONE: %s\n' "$OUT_DIR"
