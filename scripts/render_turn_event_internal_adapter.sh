#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
MOTION=${1:-output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy}
TAG=${2:-dhw4_turn_event_internal_adapter}
MUSIC=${3:-test_music_bank/dunhuangwu2_20s.wav}
mkdir -p output/rendered_candidates_real logs/turn_event_internal_adapter
python render_choreorag_results.py \
  --motion "$MOTION" \
  --music "$MUSIC" \
  --out "output/rendered_candidates_real/${TAG}_fixed_rows.mp4" \
  --camera_mode fixed \
  --sixd_layout rows \
  --smooth_window 7 \
  2>&1 | tee "logs/turn_event_internal_adapter/${TAG}_render_fixed.log"
python render_choreorag_results.py \
  --motion "$MOTION" \
  --music "$MUSIC" \
  --out "output/rendered_candidates_real/${TAG}_bodycenter_rows.mp4" \
  --camera_mode fixed \
  --sixd_layout rows \
  --body_centered \
  --smooth_window 7 \
  2>&1 | tee "logs/turn_event_internal_adapter/${TAG}_render_bodycenter.log"
ls -lh "output/rendered_candidates_real/${TAG}_"*.mp4
