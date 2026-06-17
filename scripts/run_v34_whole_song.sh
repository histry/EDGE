#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

OUT="${V26_OUT_DIR:-output/v34_c3_contact_$(date +%Y%m%d_%H%M%S)}"
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

export V32_TRANSITION_SEED="${V32_TRANSITION_SEED:-20260610}"
export V32_CANDIDATES="${V32_CANDIDATES:-8}"
export V32_GUIDANCE="${V32_GUIDANCE:-1.0}"
export V32_INR_TRUST="${V32_INR_TRUST:-0.35}"
export V32_MAX_TOTAL_RISK_RATIO="${V32_MAX_TOTAL_RISK_RATIO:-1.02}"
export V32_MAX_ENTRY_RATIO="${V32_MAX_ENTRY_RATIO:-1.05}"
export V32_MAX_EXIT_RATIO="${V32_MAX_EXIT_RATIO:-1.03}"
export V32_MAX_JERK_RATIO="${V32_MAX_JERK_RATIO:-1.03}"
export V32_MAX_FOOT_RATIO="${V32_MAX_FOOT_RATIO:-1.02}"
export V32_MAX_PENETRATION_RATIO="${V32_MAX_PENETRATION_RATIO:-1.02}"
export V32_MAX_ROTATION_STEP_RAD="${V32_MAX_ROTATION_STEP_RAD:-0.20}"
export V32_ENABLE_EDGE_DAMPING="${V32_ENABLE_EDGE_DAMPING:-0}"
export V32_STRICT_LOCKED_WARP="${V32_STRICT_LOCKED_WARP:-1}"
export V32_MAX_WARP_VIOLATIONS="${V32_MAX_WARP_VIOLATIONS:-0}"
export V32_WARP_TOLERANCE="${V32_WARP_TOLERANCE:-0.02}"

export V34_WARP_HARD_PRUNE="${V34_WARP_HARD_PRUNE:-1}"
export V34_WARP_MIN="${V34_WARP_MIN:-${V26_MIN_TIME_WARP:-0.82}}"
export V34_WARP_MAX="${V34_WARP_MAX:-${V26_MAX_TIME_WARP:-1.30}}"
export V34_WARP_TOLERANCE="${V34_WARP_TOLERANCE:-0.0}"
export V34_WARP_PENALTY_WEIGHT="${V34_WARP_PENALTY_WEIGHT:-1.25}"
export V34_WARP_PREFILTER_TOP_K="${V34_WARP_PREFILTER_TOP_K:-512}"
export V34_EXIT_HANDSHAKE="${V34_EXIT_HANDSHAKE:-1}"
export V34_EXIT_HANDSHAKE_FRAMES="${V34_EXIT_HANDSHAKE_FRAMES:-10}"
export V34_EXIT_HANDSHAKE_CANDIDATES="${V34_EXIT_HANDSHAKE_CANDIDATES:-8,10,12,16,20}"
export V34_EXIT_HANDSHAKE_STRENGTH="${V34_EXIT_HANDSHAKE_STRENGTH:-1.0}"
export V34_HANDSHAKE_MODE="${V34_HANDSHAKE_MODE:-replace}"
export V34_HANDSHAKE_MAX_ROTATION_DEG="${V34_HANDSHAKE_MAX_ROTATION_DEG:-18.0}"
export V34_HANDSHAKE_MAX_ROOT="${V34_HANDSHAKE_MAX_ROOT:-0.08}"
export V34_MAX_BOUNDARY_JERK="${V34_MAX_BOUNDARY_JERK:-5000}"
export V34_MAX_BOUNDARY_ANGULAR_JERK="${V34_MAX_BOUNDARY_ANGULAR_JERK:-5000}"
export V34_MAX_ENTRY_ROTATION_STEP_RAD="${V34_MAX_ENTRY_ROTATION_STEP_RAD:-0.16}"
export V34_MAX_EXIT_ROTATION_STEP_RAD="${V34_MAX_EXIT_ROTATION_STEP_RAD:-0.12}"
export V34_MAX_ENTRY_FK_JUMP="${V34_MAX_ENTRY_FK_JUMP:-0.060}"
export V34_MAX_EXIT_FK_JUMP="${V34_MAX_EXIT_FK_JUMP:-0.040}"
export V34_MAX_EXIT_ACCELERATION="${V34_MAX_EXIT_ACCELERATION:-12.0}"
export V34_POST_HANDSHAKE_ABSOLUTE_VETO="${V34_POST_HANDSHAKE_ABSOLUTE_VETO:-1}"
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export V34_JERK_MATCH_SHRINK="${V34_JERK_MATCH_SHRINK:-0.35}"
export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_INPAINT_REQUIRE_DIFFUSION="${V34_INPAINT_REQUIRE_DIFFUSION:-1}"
export V34_INPAINT_TRIGGER_RATIO="${V34_INPAINT_TRIGGER_RATIO:-0.72}"
export V34_INPAINT_COMPAT_SCORE_TRIGGER="${V34_INPAINT_COMPAT_SCORE_TRIGGER:-0.10}"
export V34_INPAINT_TAIL_FRAMES="${V34_INPAINT_TAIL_FRAMES:-6}"
export V34_INPAINT_HEAD_FRAMES="${V34_INPAINT_HEAD_FRAMES:-6}"
export V34_INPAINT_CONTEXT_FRAMES="${V34_INPAINT_CONTEXT_FRAMES:-4}"
export V34_INPAINT_BLEND="${V34_INPAINT_BLEND:-${V32_INR_TRUST:-0.35}}"
export V34_INPAINT_STEPS="${V34_INPAINT_STEPS:-${V32_INFERENCE_STEPS:-40}}"
export V34_INPAINT_MAX_RISK_RATIO="${V34_INPAINT_MAX_RISK_RATIO:-1.03}"
export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"
export V34_LATENT_BLEND_TOP_K="${V34_LATENT_BLEND_TOP_K:-3}"
export V34_LATENT_BLEND_TEMPERATURE="${V34_LATENT_BLEND_TEMPERATURE:-0.08}"
export V34_LATENT_BLEND_KEEP_RATIO="${V34_LATENT_BLEND_KEEP_RATIO:-1.01}"

python tools/schedule_v34_whole_song.py \
  --index_json "${V34_INDEX_JSON:-$V26_INDEX_JSON}" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  "${ARGS[@]}" \
  --out_dir "$OUT" \
  --router_ckpt "$V26_ROUTER_CKPT" \
  --v23_ckpt "$V26_V23_CKPT" \
  --planner_ckpt "${V26_PLANNER_CKPT:-}" \
  --transition_ckpt "" \
  --transition_diffusion_ckpt "${V27_TRANSITION_DIFFUSION_CKPT:-}" \
  --hierarchy_index_npz "${V26_HIERARCHY_INDEX_NPZ:-}" \
  --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-}" \
  --feature_dir "${V26_FEATURE_CACHE:-data/v26_music_features}" \
  --deep_music_cache "${V27_DEEP_MUSIC_CACHE:-data/v32_deep_music_features}" \
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
  --min_time_warp "${V26_MIN_TIME_WARP:-0.82}" \
  --max_time_warp "${V26_MAX_TIME_WARP:-1.30}" \
  --transition_min_frames "${V26_TRANSITION_MIN_FRAMES:-16}" \
  --transition_max_frames "${V26_TRANSITION_MAX_FRAMES:-54}" \
  --transition_diffusion "${V27_TRANSITION_DIFFUSION:-1}" \
  --transition_diffusion_blend "${V32_INR_TRUST:-0.35}" \
  --transition_diffusion_steps "${V32_INFERENCE_STEPS:-40}" \
  --transition_yaw_limit_dps "${V26_TRANSITION_YAW_LIMIT_DPS:-140}" \
  --yaw_transition_safety_factor "${V26_YAW_TRANSITION_SAFETY_FACTOR:-2.30}" \
  --planner_duration_weight "${V26_PLANNER_DURATION_WEIGHT:-0.15}" \
  --activity_weight "${V26_ACTIVITY_WEIGHT:-0.30}" \
  --hierarchical_retrieval "${V26_HIERARCHICAL_RETRIEVAL:-1}" \
  --hierarchy_weight "${V26_HIERARCHY_WEIGHT:-0.60}" \
  --deep_music_features "${V27_DEEP_MUSIC_FEATURES:-0}" \
  --deep_music_model "${V27_DEEP_MUSIC_MODEL:-clap}" \
  --deep_music_weight "${V27_DEEP_MUSIC_WEIGHT:-0.0}" \
  --require_deep_music "${V27_REQUIRE_DEEP_MUSIC:-0}" \
  --deep_music_min_success "${V27_DEEP_MUSIC_MIN_SUCCESS:-0.90}" \
  --graph_scheduler "${V26_GRAPH_SCHEDULER:-1}" \
  --graph_node_top_k "${V26_GRAPH_NODE_TOP_K:-512}" \
  --graph_edge_weight "${V26_GRAPH_EDGE_WEIGHT:-0.70}" \
  --graph_hard_prune "${V26_GRAPH_HARD_PRUNE:-0}" \
  --graph_hard_prune_threshold "${V26_GRAPH_HARD_PRUNE_THRESHOLD:-1.25}" \
  --anti_static_weight "${V26_ANTI_STATIC_WEIGHT:-0.50}" \
  --anti_static_activity_threshold "${V26_ANTI_STATIC_ACTIVITY_THRESHOLD:-0.030}" \
  --anti_static_min_content_frames "${V26_ANTI_STATIC_MIN_CONTENT_FRAMES:-54}" \
  --boundary_velocity_penalty_weight "${V26_BOUNDARY_VELOCITY_PENALTY_WEIGHT:-1.20}" \
  --boundary_acceleration_penalty_weight "${V26_BOUNDARY_ACCELERATION_PENALTY_WEIGHT:-1.20}" \
  --boundary_penalty_cap "${V26_BOUNDARY_PENALTY_CAP:-4.0}" \
  --turn_peak_soft_dps "${V26_TURN_PEAK_SOFT_DPS:-220}" \
  --turn_peak_hard_dps "${V26_TURN_PEAK_HARD_DPS:-360}" \
  --turn_angle_soft_deg "${V26_TURN_ANGLE_SOFT_DEG:-150}" \
  --turn_angle_hard_deg "${V26_TURN_ANGLE_HARD_DEG:-270}" \
  --turn_peak_penalty_weight "${V26_TURN_PEAK_PENALTY_WEIGHT:-1.35}" \
  --edge_damping_frames "${V26_EDGE_DAMPING_FRAMES:-0}" \
  --edge_damping_strength "${V26_EDGE_DAMPING_STRENGTH:-0.0}" \
  --pose_jump_reference "${V26_POSE_JUMP_REFERENCE:-0.10}" \
  --velocity_jump_reference "${V26_VELOCITY_JUMP_REFERENCE:-0.010}" \
  --acceleration_jump_reference "${V26_ACCELERATION_JUMP_REFERENCE:-0.018}" \
  --physical_pose_frames "${V26_PHYSICAL_POSE_FRAMES:-10}" \
  --physical_velocity_frames "${V26_PHYSICAL_VELOCITY_FRAMES:-14}" \
  --physical_acceleration_frames "${V26_PHYSICAL_ACCELERATION_FRAMES:-12}" \
  --physical_contact_frames "${V26_PHYSICAL_CONTACT_FRAMES:-10}"

echo "[PASS] V34 whole-song output: $OUT"
