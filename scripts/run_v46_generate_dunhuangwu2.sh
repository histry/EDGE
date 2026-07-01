#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-1}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-1}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_ENABLE_ROOT_Y_PHYSICS="${V46_ENABLE_ROOT_Y_PHYSICS:-1}"
export V46_ROOT_Y_MAX_FLIGHT_SECONDS="${V46_ROOT_Y_MAX_FLIGHT_SECONDS:-1.2}"
export V46_ROOT_Y_DAMPING_MAX_SECONDS="${V46_ROOT_Y_DAMPING_MAX_SECONDS:-0.28}"
export V46_IK_CHUNK_OVERLAP="${V46_IK_CHUNK_OVERLAP:-24}"
CFG="configs/v46_motionrag_diff_config.json"
RUN_ROOT="${1:-$(cat output/LATEST_V46_MOTIONRAG_DIFF.txt)}"
DB="$RUN_ROOT/db/events.npz"
AUDIO="test_music_bank/dunhuangwu2.wav"
[[ -f "$AUDIO" ]] || AUDIO="data/music/dunhuangwu2.wav"
[[ -f "$AUDIO" ]] || AUDIO="custom_music/dunhuangwu2.wav"
OUT="$RUN_ROOT/dunhuangwu2_v46_8_MotionRAG_Diff_regen.npy"
JSON="$RUN_ROOT/dunhuangwu2_v46_8_MotionRAG_Diff_regen.report.json"
MP4="$RUN_ROOT/dunhuangwu2_v46_8_MotionRAG_Diff_regen.mp4"
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --db "$DB" \
  --contrastive "$RUN_ROOT/v44_contrastive.pt" \
  --refiner "$RUN_ROOT/v45_refiner.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion.pt" \
  --out "$OUT" \
  --json "$JSON" \
  --render_output "$MP4"
