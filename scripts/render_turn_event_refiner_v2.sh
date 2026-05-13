#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

MOTION=${1:-output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2.npy}
TAG=${2:-dhw4_turn_event_refined_v2}
MUSIC=${3:-test_music_bank/dunhuangwu2_20s.wav}

mkdir -p output/rendered_candidates_real

python render_choreorag_results.py \
  --motion "$MOTION" \
  --music "$MUSIC" \
  --out "output/rendered_candidates_real/${TAG}_fixed_rows.mp4" \
  --camera_mode fixed \
  --sixd_layout rows \
  --smooth_window 7

python render_choreorag_results.py \
  --motion "$MOTION" \
  --music "$MUSIC" \
  --out "output/rendered_candidates_real/${TAG}_bodycenter_rows.mp4" \
  --camera_mode fixed \
  --sixd_layout rows \
  --body_centered \
  --smooth_window 7

ls -lh "output/rendered_candidates_real/${TAG}_"*.mp4
