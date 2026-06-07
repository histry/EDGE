#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

OUT="${V26_OUT_DIR:-output/v26_whole_song_music_dominant_$(date +%Y%m%d_%H%M%S)}"
ARGS=()
if [[ -n "${V26_MUSIC_GLOB:-}" ]]; then
  ARGS+=(--music_glob "$V26_MUSIC_GLOB")
fi
if [[ -n "${V26_MUSIC:-}" ]]; then
  IFS=';' read -ra MUSIC_ITEMS <<< "$V26_MUSIC"
  for item in "${MUSIC_ITEMS[@]}"; do
    ARGS+=(--music "$item")
  done
fi
if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "[ERROR] Set V26_MUSIC_GLOB or semicolon-separated V26_MUSIC" >&2
  exit 2
fi

python tools/schedule_v26_whole_song.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  "${ARGS[@]}" \
  --out_dir "$OUT" \
  --router_ckpt "$V26_ROUTER_CKPT" \
  --v23_ckpt "$V26_V23_CKPT" \
  --planner_ckpt "${V26_PLANNER_CKPT:-}" \
  --transition_ckpt "${V26_TRANSITION_CKPT:-}" \
  --feature_dir "${V26_FEATURE_CACHE:-data/v26_music_features}" \
  --start_pose "${V26_START_POSE:-}" \
  --fps "${V26_FPS:-30}" \
  --max_seconds "${V26_MAX_SECONDS:-0}" \
  --min_phrase_seconds "${V26_MIN_PHRASE_SECONDS:-2.5}" \
  --max_phrase_seconds "${V26_MAX_PHRASE_SECONDS:-7.5}" \
  --boundary_quantile "${V26_BOUNDARY_QUANTILE:-0.68}" \
  --beat_snap_seconds "${V26_BEAT_SNAP_SECONDS:-0.35}" \
  --max_phrases "${V26_MAX_PHRASES:-96}" \
  --beam_size "${V26_BEAM_SIZE:-24}" \
  --candidate_top_k "${V26_CANDIDATE_TOP_K:-256}" \
  --music_dominant_timing "${V26_MUSIC_DOMINANT_TIMING:-1}" \
  --allow_music_bound_override "${V26_ALLOW_MUSIC_BOUND_OVERRIDE:-1}" \
  --global_music_weight "${V26_GLOBAL_MUSIC_WEIGHT:-1.60}" \
  --global_natural_weight "${V26_GLOBAL_NATURAL_WEIGHT:-0.85}" \
  --global_planner_weight "${V26_GLOBAL_PLANNER_WEIGHT:-0.75}" \
  --min_content_frames "${V26_MIN_CONTENT_FRAMES:-12}" \
  --min_time_warp "${V26_MIN_TIME_WARP:-0.70}" \
  --max_time_warp "${V26_MAX_TIME_WARP:-1.50}" \
  --transition_min_frames "${V26_TRANSITION_MIN_FRAMES:-12}" \
  --transition_max_frames "${V26_TRANSITION_MAX_FRAMES:-48}" \
  --planner_duration_weight "${V26_PLANNER_DURATION_WEIGHT:-0.15}" \
  --boundary_velocity_penalty_weight "${V26_BOUNDARY_VELOCITY_PENALTY_WEIGHT:-0.35}" \
  --boundary_acceleration_penalty_weight "${V26_BOUNDARY_ACCELERATION_PENALTY_WEIGHT:-0.35}" \
  --boundary_penalty_cap "${V26_BOUNDARY_PENALTY_CAP:-4.0}" \
  --turn_peak_soft_dps "${V26_TURN_PEAK_SOFT_DPS:-360}" \
  --turn_peak_hard_dps "${V26_TURN_PEAK_HARD_DPS:-720}" \
  --turn_angle_soft_deg "${V26_TURN_ANGLE_SOFT_DEG:-220}" \
  --turn_angle_hard_deg "${V26_TURN_ANGLE_HARD_DEG:-420}" \
  --turn_peak_penalty_weight "${V26_TURN_PEAK_PENALTY_WEIGHT:-0.75}" \
  --edge_damping_frames "${V26_EDGE_DAMPING_FRAMES:-10}" \
  --edge_damping_strength "${V26_EDGE_DAMPING_STRENGTH:-0.65}" \
  --pose_jump_reference "${V26_POSE_JUMP_REFERENCE:-0.120}" \
  --velocity_jump_reference "${V26_VELOCITY_JUMP_REFERENCE:-0.010}" \
  --acceleration_jump_reference "${V26_ACCELERATION_JUMP_REFERENCE:-0.018}" \
  --physical_pose_frames "${V26_PHYSICAL_POSE_FRAMES:-8}" \
  --physical_velocity_frames "${V26_PHYSICAL_VELOCITY_FRAMES:-10}" \
  --physical_acceleration_frames "${V26_PHYSICAL_ACCELERATION_FRAMES:-8}" \
  --physical_contact_frames "${V26_PHYSICAL_CONTACT_FRAMES:-8}"

echo "[PASS] V26 music-dominant whole-song output: $OUT"
