#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-runs/train/exp/weights/train-100.pt}"
MUSIC="${2:-test_music_bank/dunhuang_demo.wav}"
OUT_DIR="${3:-output/demo_controlled}"

python infer_controlled.py \
  --checkpoint "$CKPT" \
  --music "$MUSIC" \
  --feature_type hybrid \
  --audio_dim 803 \
  --frames 150 \
  --model_seq_len 150 \
  --start_pose test_keyframes/start.npy \
  --end_pose test_keyframes/end.npy \
  --keyframe_space physical \
  --keyframe_width 3 \
  --hard_keyframe_project \
  --trajectory "0,0;1,2;-1,4;0,5" \
  --beat_guidance_weight 0.3 \
  --postprocess_trajectory \
  --postprocess_strength 1.0 \
  --foot_lock_postprocess \
  --restore_trajectory_after_foot_lock \
  --restore_trajectory_strength 0.25 \
  --out_dir "$OUT_DIR" \
  --out_name demo_dunhuang_controlled