#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python tools/build_v23_monotonic_duration_dataset.py \
  --motion_glob "${V23_MOTION_GLOB:-data/dunhuang_151d_physical/**/*}" \
  --event_db "${V23_EVENT_DB:-}" \
  --out "${V23_DATASET:-data/v23_v2_4_slowaware_w120_d88_9k.npz}" \
  --window_len "${V23_WINDOW_LEN:-120}" \
  --min_peak_dps "${V23_MIN_PEAK_DPS:-14}" \
  --min_turn_angle_deg "${V23_MIN_TURN_ANGLE:-10}" \
  --min_gap "${V23_MIN_GAP:-16}" \
  --min_target_duration "${V23_MIN_TARGET_DURATION:-12}" \
  --max_target_duration "${V23_MAX_TARGET_DURATION:-88}" \
  --turn_threshold_ratio "${V23_TURN_THRESHOLD_RATIO:-0.12}" \
  --activity_threshold_ratio "${V23_ACTIVITY_THRESHOLD_RATIO:-0.22}" \
  --boundary_yaw_ratio "${V23_BOUNDARY_YAW_RATIO:-0.04}" \
  --quiet_run "${V23_QUIET_RUN:-8}" \
  --opposite_run "${V23_OPPOSITE_RUN:-4}" \
  --phrase_margin "${V23_PHRASE_MARGIN:-3}" \
  --slow_pose_span "${V23_SLOW_POSE_SPAN:-10}" \
  --slow_angle_window "${V23_SLOW_ANGLE_WINDOW:-24}" \
  --search_duration_multiplier "${V23_SEARCH_DURATION_MULTIPLIER:-1.80}" \
  --split_valley_radius "${V23_SPLIT_VALLEY_RADIUS:-3}" \
  --reversal_angle_deg "${V23_REVERSAL_ANGLE_DEG:-7.0}" \
  --secondary_peak_ratio "${V23_SECONDARY_PEAK_RATIO:-0.48}" \
  --split_score_threshold "${V23_SPLIT_SCORE_THRESHOLD:-0.68}" \
  --long_split_score_threshold "${V23_LONG_SPLIT_SCORE_THRESHOLD:-0.42}" \
  --min_direction_consistency "${V23_MIN_DIRECTION_CONSISTENCY:-0.18}" \
  --cumulative_low "${V23_CUMULATIVE_LOW:-0.03}" \
  --cumulative_high "${V23_CUMULATIVE_HIGH:-0.97}" \
  --augmentations_per_event "${V23_AUGMENTATIONS:-16}" \
  --min_speed_factor "${V23_MIN_SPEED_FACTOR:-1.15}" \
  --max_speed_factor "${V23_MAX_SPEED_FACTOR:-3.0}" \
  --identity_fraction "${V23_IDENTITY_FRACTION:-0.25}" \
  --center_jitter "${V23_CENTER_JITTER:-8}" \
  --mask_context "${V23_MASK_CONTEXT:-6}" \
  --min_corrupted_duration "${V23_MIN_CORRUPTED_DURATION:-4}" \
  --duration_bins "${V23_DURATION_BINS:-auto:6}" \
  --balance_power "${V23_BALANCE_POWER:-0.35}" \
  --max_bin_fraction "${V23_MAX_BIN_FRACTION:-0.45}" \
  --max_samples "${V23_MAX_SAMPLES:-9000}" \
  --seed "${V23_DATA_SEED:-20260626}"
