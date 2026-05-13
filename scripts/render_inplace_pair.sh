#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

AUDIO="test_music_bank/dunhuangwu2.wav"

for CASE in dunhuangwu2_static_beat0 dunhuangwu2_static_beat0.03; do
  for CAM in fixed follow; do
    echo "Rendering ${CASE}_${CAM}.mp4"

    python render_from_npy.py \
      --motion "output/inplace_music_align/${CASE}.npy" \
      --audio "$AUDIO" \
      --output "output/inplace_music_align/${CASE}_${CAM}.mp4" \
      --camera_mode "$CAM"
  done
done

echo "✅ rendered inplace pair"
