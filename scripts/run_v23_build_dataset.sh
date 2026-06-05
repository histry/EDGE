#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/build_v23_monotonic_duration_dataset.py \
  --motion_glob "${V23_MOTION_GLOB:-data/dunhuang_151d_physical/**/*}" \
  --event_db "${V23_EVENT_DB:-}" \
  --out "${V23_DATASET:-data/v23_monotonic_duration_dataset.npz}" \
  --window_len "${V23_WINDOW_LEN:-72}" \
  --min_peak_dps "${V23_MIN_PEAK_DPS:-32}" \
  --min_turn_angle_deg "${V23_MIN_TURN_ANGLE:-14}" \
  --augmentations_per_event "${V23_AUGMENTATIONS:-10}" \
  --min_speed_factor "${V23_MIN_SPEED_FACTOR:-1.15}" \
  --max_speed_factor "${V23_MAX_SPEED_FACTOR:-8.0}" \
  --identity_ratio "${V23_IDENTITY_RATIO:-0.10}" \
  --center_jitter "${V23_CENTER_JITTER:-6}" \
  --max_samples "${V23_MAX_SAMPLES:-16000}" \
  --seed "${V23_DATA_SEED:-20260606}"
