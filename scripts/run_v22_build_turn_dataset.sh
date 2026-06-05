#!/usr/bin/env bash
set -Eeuo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

OUT="${V22_TURN_DATASET:-data/v22_turn_pace_dataset.npz}"
ARGS=(
  --out "$OUT"
  --window_len "${V22_TURN_WINDOW:-72}"
  --fps "${V22_FPS:-30}"
  --min_peak_dps "${V22_DATA_MIN_TURN_DPS:-38}"
  --min_turn_angle_deg "${V22_DATA_MIN_TURN_ANGLE:-18}"
  --min_gap "${V22_DATA_MIN_GAP:-24}"
  --max_events_per_source "${V22_DATA_MAX_EVENTS_PER_SOURCE:-160}"
  --augmentations_per_event "${V22_DATA_AUGMENTATIONS:-3}"
  --min_speed_factor "${V22_DATA_MIN_SPEED_FACTOR:-1.30}"
  --max_speed_factor "${V22_DATA_MAX_SPEED_FACTOR:-2.35}"
  --identity_ratio "${V22_DATA_IDENTITY_RATIO:-0.12}"
  --max_samples "${V22_DATA_MAX_SAMPLES:-12000}"
  --seed "${V22_SEED:-20260605}"
)

MOTION_GLOB="${V22_MOTION_GLOB:-data/dunhuang_151d_physical/*.npy}"
ARGS+=(--motion_glob "$MOTION_GLOB")

if [[ -n "${V22_EVENT_DB:-}" ]]; then
  ARGS+=(--event_db "$V22_EVENT_DB")
fi

python tools/build_v22_turn_pace_dataset.py "${ARGS[@]}"
printf '\nDONE: %s\n' "$OUT"
