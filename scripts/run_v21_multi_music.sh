#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INDEX_PREFIX="${V21_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"
AUDIO_GLOB="${V21_AUDIO_GLOB:-test_music_bank/*.wav}"
RUN_ROOT="${V21_RUN_ROOT:-output/v21_multi_music_$(date +%Y%m%d_%H%M%S)}"
FEATURE_DIR="${V21_MUSIC_FEATURE_DIR:-$RUN_ROOT/music_features}"
START_POSE="${V21_START_POSE:-data/dunhuang_dynamic_event_rag_physical/v20_common_start_pose.npy}"
ROUTER_CKPT="${V21_ROUTER_CKPT:-}"
TRANSITION_CKPT="${V21_TRANSITION_CKPT:-}"
mkdir -p "$RUN_ROOT"

ARGS=(
  --index_json "${INDEX_PREFIX}.json"
  --index_npz "${INDEX_PREFIX}.npz"
  --music_glob "$AUDIO_GLOB"
  --out_dir "$RUN_ROOT"
  --feature_dir "$FEATURE_DIR"
  --num_frames "${V21_NUM_FRAMES:-150}"
  --phrase_count "${V21_PHRASE_COUNT:-3}"
  --beam_size "${V21_BEAM_SIZE:-48}"
  --candidate_top_k "${V21_CANDIDATE_TOP_K:-1800}"
  --refine_rounds "${V21_REFINE_ROUNDS:-2}"
  --start_pose "$START_POSE"
  --start_anchor_blend "${V21_START_ANCHOR_BLEND:-8}"
  --style_weight "${V21_STYLE_WEIGHT:-1.35}"
  --quality_weight "${V21_QUALITY_WEIGHT:-0.65}"
  --safety_weight "${V21_SAFETY_WEIGHT:-0.35}"
  --music_weight "${V21_MUSIC_WEIGHT:-0.80}"
  --event_weight "${V21_EVENT_WEIGHT:-0.65}"
  --activity_weight "${V21_ACTIVITY_WEIGHT:-0.25}"
  --duration_weight "${V21_DURATION_WEIGHT:-0.25}"
  --transition_weight "${V21_TRANSITION_WEIGHT:-0.55}"
  --mmr_weight "${V21_MMR_WEIGHT:-0.38}"
  --family_repeat_weight "${V21_FAMILY_REPEAT_WEIGHT:-0.55}"
  --source_repeat_weight "${V21_SOURCE_REPEAT_WEIGHT:-0.15}"
  --batch_overlap_weight "${V21_BATCH_OVERLAP_WEIGHT:-0.30}"
  --batch_family_overlap_weight "${V21_BATCH_FAMILY_OVERLAP_WEIGHT:-0.20}" \
  --batch_mmr_weight "${V21_BATCH_MMR_WEIGHT:-0.18}"
  --time_warp_weight "${V21_TIME_WARP_WEIGHT:-0.25}"
  --min_time_warp "${V21_MIN_TIME_WARP:-0.65}"
  --max_time_warp "${V21_MAX_TIME_WARP:-1.45}"
)

if [[ -n "$ROUTER_CKPT" ]]; then
  ARGS+=(--router_ckpt "$ROUTER_CKPT")
fi
if [[ -n "$TRANSITION_CKPT" ]]; then
  ARGS+=(--transition_ckpt "$TRANSITION_CKPT")
fi
if [[ "${V21_HARD_FAMILY_UNIQUE:-0}" == "1" ]]; then
  ARGS+=(--hard_family_unique)
fi

python tools/schedule_v21_multi_music.py "${ARGS[@]}"

if [[ "${V21_RENDER:-1}" == "1" ]]; then
  shopt -s nullglob
  for AUDIO in $AUDIO_GLOB; do
    NAME="$(basename "$AUDIO")"
    STEM="${NAME%.*}"
    NPY="$RUN_ROOT/${STEM}_v21.npy"
    [[ -f "$NPY" ]] || continue
    python render_from_npy.py \
      --motion "$NPY" \
      --audio "$AUDIO" \
      --output "$RUN_ROOT/${STEM}_v21_fixed.mp4" \
      --camera_mode fixed
    python render_from_npy.py \
      --motion "$NPY" \
      --audio "$AUDIO" \
      --output "$RUN_ROOT/${STEM}_v21_follow.mp4" \
      --camera_mode follow
  done
fi

python tools/evaluate_v21_multi_music.py --run_dir "$RUN_ROOT"
printf '\nDONE: %s\n' "$RUN_ROOT"
