#!/usr/bin/env bash
set -Eeuo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INDEX_PREFIX="${V22_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v22_turn_aware_event_index}"
AUDIO_GLOB="${V22_AUDIO_GLOB:-test_music_bank/dunhuangwu[234].wav}"
RUN_ROOT="${V22_RUN_ROOT:-output/v22_turn_aware_$(date +%Y%m%d_%H%M%S)}"
FEATURE_DIR="${V22_MUSIC_FEATURE_DIR:-$RUN_ROOT/music_features}"
START_POSE="${V22_START_POSE:-data/dunhuang_dynamic_event_rag_physical/v20_common_start_pose.npy}"
ROUTER_CKPT="${V22_ROUTER_CKPT:-${V21_ROUTER_CKPT:-}}"
TRANSITION_CKPT="${V22_TRANSITION_CKPT:-${V21_TRANSITION_CKPT:-}}"
PACE_CKPT="${V22_PACE_REFINER_CKPT:-}"
mkdir -p "$RUN_ROOT"

ARGS=(
  --index_json "${INDEX_PREFIX}.json"
  --index_npz "${INDEX_PREFIX}.npz"
  --music_glob "$AUDIO_GLOB"
  --out_dir "$RUN_ROOT"
  --feature_dir "$FEATURE_DIR"
  --num_frames "${V22_NUM_FRAMES:-150}"
  --phrase_count "${V22_PHRASE_COUNT:-3}"
  --beam_size "${V22_BEAM_SIZE:-64}"
  --candidate_top_k "${V22_CANDIDATE_TOP_K:-2400}"
  --refine_rounds "${V22_REFINE_ROUNDS:-2}"
  --start_pose "$START_POSE"
  --start_anchor_blend "${V22_START_ANCHOR_BLEND:-8}"
  --style_weight "${V22_STYLE_WEIGHT:-1.45}"
  --quality_weight "${V22_QUALITY_WEIGHT:-0.65}"
  --safety_weight "${V22_SAFETY_WEIGHT:-0.35}"
  --music_weight "${V22_MUSIC_WEIGHT:-0.70}"
  --event_weight "${V22_EVENT_WEIGHT:-0.60}"
  --activity_weight "${V22_ACTIVITY_WEIGHT:-0.27}"
  --duration_weight "${V22_DURATION_WEIGHT:-0.35}"
  --transition_weight "${V22_TRANSITION_WEIGHT:-0.60}"
  --mmr_weight "${V22_MMR_WEIGHT:-0.36}"
  --family_repeat_weight "${V22_FAMILY_REPEAT_WEIGHT:-0.55}"
  --source_repeat_weight "${V22_SOURCE_REPEAT_WEIGHT:-0.15}"
  --batch_overlap_weight "${V22_BATCH_OVERLAP_WEIGHT:-0.25}"
  --batch_family_overlap_weight "${V22_BATCH_FAMILY_OVERLAP_WEIGHT:-0.18}"
  --batch_mmr_weight "${V22_BATCH_MMR_WEIGHT:-0.15}"
  --time_warp_weight "${V22_TIME_WARP_WEIGHT:-0.30}"
  --min_time_warp "${V22_MIN_TIME_WARP:-0.72}"
  --max_time_warp "${V22_MAX_TIME_WARP:-1.35}"
  --turn_speed_weight "${V22_TURN_SPEED_WEIGHT:-0.95}"
  --turn_speed_hard_ratio "${V22_TURN_SPEED_HARD_RATIO:-1.55}"
  --turn_refine_threshold_ratio "${V22_TURN_REFINE_THRESHOLD_RATIO:-1.08}"
  --turn_refine_window "${V22_TURN_REFINE_WINDOW:-72}"
  --turn_refine_context "${V22_TURN_REFINE_CONTEXT:-10}"
  --turn_refine_strength "${V22_TURN_REFINE_STRENGTH:-0.90}"
  --turn_refine_max_events "${V22_TURN_REFINE_MAX_EVENTS:-4}"
)

[[ -n "$ROUTER_CKPT" ]] && ARGS+=(--router_ckpt "$ROUTER_CKPT")
[[ -n "$TRANSITION_CKPT" ]] && ARGS+=(--transition_ckpt "$TRANSITION_CKPT")
[[ -n "$PACE_CKPT" ]] && ARGS+=(--pace_refiner_ckpt "$PACE_CKPT")
[[ "${V22_DISABLE_TURN_GATE:-0}" == "1" ]] && ARGS+=(--disable_turn_gate)
[[ "${V22_HARD_FAMILY_UNIQUE:-0}" == "1" ]] && ARGS+=(--hard_family_unique)

python tools/schedule_v22_multi_music.py "${ARGS[@]}"

if [[ "${V22_RENDER:-1}" == "1" ]]; then
  shopt -s nullglob
  for AUDIO in $AUDIO_GLOB; do
    STEM="$(basename "$AUDIO")"
    STEM="${STEM%.*}"
    NPY="$RUN_ROOT/${STEM}_v22.npy"
    [[ -f "$NPY" ]] || continue
    python render_from_npy.py \
      --motion "$NPY" \
      --audio "$AUDIO" \
      --output "$RUN_ROOT/${STEM}_v22_fixed.mp4" \
      --camera_mode fixed
    if [[ "${V22_RENDER_FOLLOW:-0}" == "1" ]]; then
      python render_from_npy.py \
        --motion "$NPY" \
        --audio "$AUDIO" \
        --output "$RUN_ROOT/${STEM}_v22_follow.mp4" \
        --camera_mode follow
    fi
  done
fi

python tools/evaluate_v22_multi_music.py --run_dir "$RUN_ROOT"
printf '\nDONE: %s\n' "$RUN_ROOT"
