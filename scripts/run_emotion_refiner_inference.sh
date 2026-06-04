#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

INPUT=${INPUT:-output/night_v16_visual_first_20260602_204417/scheduled_demos_v16c}
AUDIO=${AUDIO:-test_music_bank/dunhuangwu2.wav}
CHECKPOINT=${CHECKPOINT:-output/v17_emotion_refiner/checkpoints/best.pt}
OUT_DIR=${OUT_DIR:-output/v17_emotion_refined}
STARTS=${STARTS:-"0,32,64,96"}
SEAM_RADIUS=${SEAM_RADIUS:-8}

mkdir -p "$OUT_DIR"

MUSIC_NPY="$OUT_DIR/music_emotion.npy"
python tools/extract_music_emotion_features.py \
  --audio "$AUDIO" \
  --out_npy "$MUSIC_NPY" \
  --out_json "$OUT_DIR/music_emotion.json" \
  --num_frames 150

python infer_emotion_refiner.py \
  --input "$INPUT" \
  --checkpoint "$CHECKPOINT" \
  --out_dir "$OUT_DIR" \
  --music_npy "$MUSIC_NPY" \
  --starts "$STARTS" \
  --seam_radius "$SEAM_RADIUS"

if [ "${RENDER:-1}" = "1" ]; then
  for f in "$OUT_DIR"/*_emotion_refined.npy; do
    [ -f "$f" ] || continue
    stem=$(basename "$f" .npy)
    python render_from_npy.py --motion "$f" --audio "$AUDIO" --output "$OUT_DIR/${stem}_fixed.mp4" --camera_mode fixed || true
    python render_from_npy.py --motion "$f" --audio "$AUDIO" --output "$OUT_DIR/${stem}_follow.mp4" --camera_mode follow || true
  done
fi

echo "DONE: $OUT_DIR"
