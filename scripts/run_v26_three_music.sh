#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"

export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
export V26_OUT_DIR="${V26_OUT_DIR:-output/v26_three_music_$(date +%Y%m%d_%H%M%S)}"

bash scripts/run_v26_whole_song.sh

for key in dunhuangwu2 dunhuangwu3 dunhuangwu4; do
  motion="$V26_OUT_DIR/${key}_v26.npy"
  audio="test_music_bank/${key}.wav"
  report="$V26_OUT_DIR/${key}_v26.schedule_report.json"
  if [[ -f "$motion" ]]; then
    python render_from_npy.py \
      --motion "$motion" \
      --audio "$audio" \
      --output "$V26_OUT_DIR/${key}_v26_fixed.mp4" \
      --camera_mode fixed
    python tools/evaluate_v26_long_dance.py \
      --motion "$motion" \
      --schedule_report "$report" \
      --out_json "$V26_OUT_DIR/${key}_v26.evaluation.json"
  fi
done

echo "[PASS] V26 three-music experiment: $V26_OUT_DIR"
