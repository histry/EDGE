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

OUT="${V26_OUT_DIR:-output/v30_whole_song_$(date +%Y%m%d_%H%M%S)}"
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

# V30 continuous-INR and geometric retrieval runtime defaults.
export V30_TRANSITION_SEED="${V30_TRANSITION_SEED:-20260610}"
export V30_LATENT_GUIDANCE="${V30_LATENT_GUIDANCE:-1.20}"
export V30_INR_BLEND="${V30_INR_BLEND:-0.85}"
export V30_INR_BLEND_POWER="${V30_INR_BLEND_POWER:-1.0}"
export V30_TRANSITION_FILTER_WINDOW="${V30_TRANSITION_FILTER_WINDOW:-3}"
export V30_TRANSITION_FILTER_STRENGTH="${V30_TRANSITION_FILTER_STRENGTH:-0.10}"
export V30_PRESERVE_ROUGH_CONTACTS="${V30_PRESERVE_ROUGH_CONTACTS:-1}"
export V30_CROSSMODAL_RETRIEVAL_WEIGHT="${V30_CROSSMODAL_RETRIEVAL_WEIGHT:-0.35}"
: "${V30_ALIGNMENT_CKPT:?Set V30_ALIGNMENT_CKPT}"

python tools/schedule_v30_whole_song.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  "${ARGS[@]}" \
  --out_dir "$OUT" \
  --router_ckpt "$V26_ROUTER_CKPT" \
  --v23_ckpt "$V26_V23_CKPT" \
  --planner_ckpt "${V26_PLANNER_CKPT:-}" \
  --transition_ckpt "${V26_TRANSITION_CKPT:-}" \
  --transition_diffusion_ckpt "${V27_TRANSITION_DIFFUSION_CKPT:-}" \
  --hierarchy_index_npz "${V26_HIERARCHY_INDEX_NPZ:-}" \
  --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-}" \
  --feature_dir "${V26_FEATURE_CACHE:-data/v26_music_features}" \
  --deep_music_cache "${V27_DEEP_MUSIC_CACHE:-data/v27_deep_music_features}" \
  --start_pose "${V26_START_POSE:-}" \
  --fps "${V26_FPS:-30}" \
  --max_seconds "${V26_MAX_SECONDS:-0}" \
  --min_phrase_seconds "${V26_MIN_PHRASE_SECONDS:-2.5}" \
  --max_phrase_seconds "${V26_MAX_PHRASE_SECONDS:-7.5}" \
  --boundary_quantile "${V26_BOUNDARY_QUANTILE:-0.68}" \
  --beat_snap_seconds "${V26_BEAT_SNAP_SECONDS:-0.35}" \
  --max_phrases "${V26_MAX_PHRASES:-160}" \
  --multi_event_phrases "${V26_MULTI_EVENT_PHRASES:-1}" \
  --lock_music_boundaries "${V26_LOCK_MUSIC_BOUNDARIES:-1}" \
  --max_single_event_seconds "${V26_MAX_SINGLE_EVENT_SECONDS:-2.80}" \
  --calm_max_single_event_seconds "${V26_CALM_MAX_SINGLE_EVENT_SECONDS:-2.60}" \
  --min_subphrase_seconds "${V26_MIN_SUBPHRASE_SECONDS:-1.45}" \
  --max_events_per_phrase "${V26_MAX_EVENTS_PER_PHRASE:-4}" \
  --slot_beat_snap_seconds "${V26_SLOT_BEAT_SNAP_SECONDS:-0.25}" \
  --beam_size "${V26_BEAM_SIZE:-48}" \
  --candidate_top_k "${V26_CANDIDATE_TOP_K:-768}" \
  --music_dominant_timing "${V26_MUSIC_DOMINANT_TIMING:-1}" \
  --allow_music_bound_override "${V26_ALLOW_MUSIC_BOUND_OVERRIDE:-1}" \
  --global_music_weight "${V26_GLOBAL_MUSIC_WEIGHT:-1.60}" \
  --global_natural_weight "${V26_GLOBAL_NATURAL_WEIGHT:-0.85}" \
  --global_planner_weight "${V26_GLOBAL_PLANNER_WEIGHT:-0.75}" \
  --min_content_frames "${V26_MIN_CONTENT_FRAMES:-18}" \
  --min_time_warp "${V26_MIN_TIME_WARP:-0.88}" \
  --max_time_warp "${V26_MAX_TIME_WARP:-1.35}" \
  --transition_min_frames "${V26_TRANSITION_MIN_FRAMES:-12}" \
  --transition_max_frames "${V26_TRANSITION_MAX_FRAMES:-48}" \
  --transition_diffusion "${V27_TRANSITION_DIFFUSION:-1}" \
  --transition_diffusion_blend "${V27_TRANSITION_DIFFUSION_BLEND:-0.18}" \
  --transition_diffusion_steps "${V27_TRANSITION_DIFFUSION_STEPS:-32}" \
  --transition_yaw_limit_dps "${V26_TRANSITION_YAW_LIMIT_DPS:-150}" \
  --yaw_transition_safety_factor "${V26_YAW_TRANSITION_SAFETY_FACTOR:-2.20}" \
  --planner_duration_weight "${V26_PLANNER_DURATION_WEIGHT:-0.15}" \
  --activity_weight "${V26_ACTIVITY_WEIGHT:-0.25}" \
  --hierarchical_retrieval "${V26_HIERARCHICAL_RETRIEVAL:-1}" \
  --hierarchy_weight "${V26_HIERARCHY_WEIGHT:-0.60}" \
  --deep_music_features "${V27_DEEP_MUSIC_FEATURES:-1}" \
  --deep_music_model "${V27_DEEP_MUSIC_MODEL:-clap}" \
  --deep_music_weight "${V27_DEEP_MUSIC_WEIGHT:-0.30}" \
  --require_deep_music "${V27_REQUIRE_DEEP_MUSIC:-1}" \
  --deep_music_min_success "${V27_DEEP_MUSIC_MIN_SUCCESS:-0.80}" \
  --graph_scheduler "${V26_GRAPH_SCHEDULER:-1}" \
  --graph_node_top_k "${V26_GRAPH_NODE_TOP_K:-128}" \
  --graph_edge_weight "${V26_GRAPH_EDGE_WEIGHT:-0.65}" \
  --graph_hard_prune "${V26_GRAPH_HARD_PRUNE:-0}" \
  --graph_hard_prune_threshold "${V26_GRAPH_HARD_PRUNE_THRESHOLD:-1.25}" \
  --anti_static_weight "${V26_ANTI_STATIC_WEIGHT:-0.50}" \
  --anti_static_activity_threshold "${V26_ANTI_STATIC_ACTIVITY_THRESHOLD:-0.030}" \
  --anti_static_min_content_frames "${V26_ANTI_STATIC_MIN_CONTENT_FRAMES:-54}" \
  --boundary_velocity_penalty_weight "${V26_BOUNDARY_VELOCITY_PENALTY_WEIGHT:-0.85}" \
  --boundary_acceleration_penalty_weight "${V26_BOUNDARY_ACCELERATION_PENALTY_WEIGHT:-0.85}" \
  --boundary_penalty_cap "${V26_BOUNDARY_PENALTY_CAP:-4.0}" \
  --turn_peak_soft_dps "${V26_TURN_PEAK_SOFT_DPS:-240}" \
  --turn_peak_hard_dps "${V26_TURN_PEAK_HARD_DPS:-420}" \
  --turn_angle_soft_deg "${V26_TURN_ANGLE_SOFT_DEG:-160}" \
  --turn_angle_hard_deg "${V26_TURN_ANGLE_HARD_DEG:-300}" \
  --turn_peak_penalty_weight "${V26_TURN_PEAK_PENALTY_WEIGHT:-1.25}" \
  --edge_damping_frames "${V26_EDGE_DAMPING_FRAMES:-4}" \
  --edge_damping_strength "${V26_EDGE_DAMPING_STRENGTH:-0.25}" \
  --pose_jump_reference "${V26_POSE_JUMP_REFERENCE:-0.10}" \
  --velocity_jump_reference "${V26_VELOCITY_JUMP_REFERENCE:-0.010}" \
  --acceleration_jump_reference "${V26_ACCELERATION_JUMP_REFERENCE:-0.018}" \
  --physical_pose_frames "${V26_PHYSICAL_POSE_FRAMES:-8}" \
  --physical_velocity_frames "${V26_PHYSICAL_VELOCITY_FRAMES:-12}" \
  --physical_acceleration_frames "${V26_PHYSICAL_ACCELERATION_FRAMES:-10}" \
  --physical_contact_frames "${V26_PHYSICAL_CONTACT_FRAMES:-8}"

echo "[PASS] V30 whole-song output: $OUT"
