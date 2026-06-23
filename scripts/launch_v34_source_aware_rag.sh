#!/usr/bin/env bash
set -euo pipefail

# Source-aware V34 launcher for the motion-only Dunhuang BVH setting.
#
# The raw BVH library has no paired music.  Source names follow:
#   dancer_repeat_Take_category.bvh
# e.g. dyl_002_Take_003.bvh -> dancer=dyl, repeat=002, category=Take_003.
#
# This launcher builds an aligned JSON+NPZ index that is category/source aware,
# rebuilds the matching hierarchy index, then runs V34 retrieval with source
# repetition penalties enabled.  It is safe for pure inference and can also be
# used before retraining by setting V34_TRAIN=1.

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p output data

BASE_V34_INDEX_JSON="${BASE_V34_INDEX_JSON:-data/v34_shared_event_index.json}"
BASE_V34_INDEX_NPZ="${BASE_V34_INDEX_NPZ:-data/v26_music_dominant_duration_index.npz}"

SOURCE_AWARE_PREFIX="${SOURCE_AWARE_PREFIX:-data/v34_source_aware/v34_shared_event_index_source_aware}"
SOURCE_AWARE_JSON="${SOURCE_AWARE_JSON:-${SOURCE_AWARE_PREFIX}.json}"
SOURCE_AWARE_NPZ="${SOURCE_AWARE_NPZ:-${SOURCE_AWARE_PREFIX}.npz}"
SOURCE_AWARE_AUDIT="${SOURCE_AWARE_AUDIT:-${SOURCE_AWARE_PREFIX}.audit.json}"
SOURCE_AWARE_HIER_NPZ="${SOURCE_AWARE_HIER_NPZ:-data/v34_source_aware/v34_hierarchical_event_index_source_aware.npz}"
SOURCE_AWARE_HIER_JSON="${SOURCE_AWARE_HIER_JSON:-data/v34_source_aware/v34_hierarchical_event_index_source_aware.json}"

python -m py_compile \
  tools/v34_source_aware_rag.py \
  tools/build_v34_source_aware_index.py \
  tools/build_v21_shared_event_index.py \
  tools/v34_warp_aware_retrieval.py \
  tools/v34_motion_quality_postprocess.py \
  tools/build_v26_hierarchical_event_index.py

if [[ "${V34_SOURCE_AWARE_REBUILD:-1}" == "1" || ! -f "$SOURCE_AWARE_JSON" || ! -f "$SOURCE_AWARE_NPZ" ]]; then
  python tools/build_v34_source_aware_index.py \
    --index_json "$BASE_V34_INDEX_JSON" \
    --index_npz "$BASE_V34_INDEX_NPZ" \
    --out_json "$SOURCE_AWARE_JSON" \
    --out_npz "$SOURCE_AWARE_NPZ" \
    --audit_json "$SOURCE_AWARE_AUDIT" \
    --cap_per_source_uid "${V34_SOURCE_CAP_PER_SOURCE_UID:-64}" \
    --category_cap_factor "${V34_SOURCE_CATEGORY_CAP_FACTOR:-1.35}" \
    --repeat_cap_factor "${V34_SOURCE_REPEAT_CAP_FACTOR:-1.60}" \
    --dancer_cap_factor "${V34_SOURCE_DANCER_CAP_FACTOR:-1.50}" \
    --max_events "${V34_SOURCE_MAX_EVENTS:-0}"
fi

if [[ "${V34_SOURCE_AWARE_REBUILD_HIERARCHY:-1}" == "1" || ! -f "$SOURCE_AWARE_HIER_NPZ" ]]; then
  python tools/build_v26_hierarchical_event_index.py \
    --index_json "$SOURCE_AWARE_JSON" \
    --duration_index_npz "$SOURCE_AWARE_NPZ" \
    --out_npz "$SOURCE_AWARE_HIER_NPZ" \
    --out_json "$SOURCE_AWARE_HIER_JSON" \
    --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt}"
fi

export V34_INDEX_JSON="$SOURCE_AWARE_JSON"
export V26_DURATION_INDEX_NPZ="$SOURCE_AWARE_NPZ"
export V26_HIERARCHY_INDEX_NPZ="$SOURCE_AWARE_HIER_NPZ"
export V34_BUILD_EVENT_LIBRARY=0
export V34_LIBRARY_DIR="${V34_LIBRARY_DIR:-data/v34_event_library}"

# Retrieval-time source/category controls.
export V34_SOURCE_AWARE_RAG="${V34_SOURCE_AWARE_RAG:-1}"
export V34_SOURCE_AWARE_WEIGHT="${V34_SOURCE_AWARE_WEIGHT:-1.0}"
export V34_SOURCE_AWARE_WINDOW="${V34_SOURCE_AWARE_WINDOW:-8}"
export V34_SOURCE_UID_REPEAT_WEIGHT="${V34_SOURCE_UID_REPEAT_WEIGHT:-2.50}"
export V34_DANCER_CATEGORY_REPEAT_WEIGHT="${V34_DANCER_CATEGORY_REPEAT_WEIGHT:-1.20}"
export V34_CATEGORY_REPEAT_WEIGHT="${V34_CATEGORY_REPEAT_WEIGHT:-0.35}"
export V34_DANCER_REPEAT_WEIGHT="${V34_DANCER_REPEAT_WEIGHT:-0.25}"
export V34_REPEAT_ID_REPEAT_WEIGHT="${V34_REPEAT_ID_REPEAT_WEIGHT:-0.12}"
export V34_CATEGORY_PRIOR_BALANCE_WEIGHT="${V34_CATEGORY_PRIOR_BALANCE_WEIGHT:-0.20}"
export V34_DANCER_PRIOR_BALANCE_WEIGHT="${V34_DANCER_PRIOR_BALANCE_WEIGHT:-0.08}"
export V34_REPEAT_PRIOR_BALANCE_WEIGHT="${V34_REPEAT_PRIOR_BALANCE_WEIGHT:-0.06}"

# Keep rhythm repair enabled by default because source balancing alone cannot
# prevent quick-pose-change + long-hold collapse.
export V34_RHYTHM_DEGRADATION_PENALTY="${V34_RHYTHM_DEGRADATION_PENALTY:-1}"
export V34_RHYTHM_WEIGHT="${V34_RHYTHM_WEIGHT:-1.20}"
export V34_HOLD_FIRST1S_RATIO_LIMIT="${V34_HOLD_FIRST1S_RATIO_LIMIT:-0.55}"
export V34_HOLD_TAIL_ENERGY_LIMIT="${V34_HOLD_TAIL_ENERGY_LIMIT:-0.024}"
export V34_MIN_SLOT_MEAN_ENERGY="${V34_MIN_SLOT_MEAN_ENERGY:-0.018}"
export V34_LATE_ENERGY_RATIO_MIN="${V34_LATE_ENERGY_RATIO_MIN:-0.34}"
export V34_TAIL_MEAN_RATIO_MIN="${V34_TAIL_MEAN_RATIO_MIN:-0.55}"
export V34_LOW_ENERGY_FRACTION_MAX="${V34_LOW_ENERGY_FRACTION_MAX:-0.62}"
export V34_HOLD_PENALTY_WEIGHT="${V34_HOLD_PENALTY_WEIGHT:-4.00}"
export V34_FRONTLOAD_PENALTY_WEIGHT="${V34_FRONTLOAD_PENALTY_WEIGHT:-4.50}"
export V34_TAIL_RATIO_PENALTY_WEIGHT="${V34_TAIL_RATIO_PENALTY_WEIGHT:-3.00}"
export V34_COVERAGE_PENALTY_WEIGHT="${V34_COVERAGE_PENALTY_WEIGHT:-2.75}"
export V34_LOW_ENERGY_PENALTY_WEIGHT="${V34_LOW_ENERGY_PENALTY_WEIGHT:-2.00}"
export V34_DENSITY_PENALTY_WEIGHT="${V34_DENSITY_PENALTY_WEIGHT:-4.50}"
export V34_STATIC_STREAK_ALLOW="${V34_STATIC_STREAK_ALLOW:-1}"
export V34_STREAK_PENALTY_WEIGHT="${V34_STREAK_PENALTY_WEIGHT:-3.00}"
export V34_SELF_COLLISION_RETRIEVAL="${V34_SELF_COLLISION_RETRIEVAL:-1}"
export V34_SELF_COLLISION_RADIUS="${V34_SELF_COLLISION_RADIUS:-0.16}"
export V34_SELF_COLLISION_PENALTY_WEIGHT="${V34_SELF_COLLISION_PENALTY_WEIGHT:-1.75}"
export V34_BOUNDARY_COMPAT="${V34_BOUNDARY_COMPAT:-1}"
export V34_COMPAT_DENSE_SCORE="${V34_COMPAT_DENSE_SCORE:-1}"
export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"

# Output-only physical cleanup.  Keeps raw *_v26.npy files intact and creates
# *_v26_motion_quality.npy for metrics/rendering.
export V34_MOTION_QUALITY_POSTPROCESS="${V34_MOTION_QUALITY_POSTPROCESS:-1}"
export V34_ROOT_Y_PHYSICS="${V34_ROOT_Y_PHYSICS:-1}"
export V34_MIN_FLIGHT_FRAMES="${V34_MIN_FLIGHT_FRAMES:-6}"
export V34_PARABOLA_STRENGTH="${V34_PARABOLA_STRENGTH:-0.60}"
export V34_MIN_ARC_LIFT="${V34_MIN_ARC_LIFT:-0.012}"
export V34_MAX_ARC_LIFT="${V34_MAX_ARC_LIFT:-0.10}"
export V34_LANDING_FRAMES="${V34_LANDING_FRAMES:-8}"
export V34_LANDING_MAX_DROP="${V34_LANDING_MAX_DROP:-0.035}"
export V34_LANDING_STRENGTH="${V34_LANDING_STRENGTH:-0.75}"
export V34_COLLISION_IK="${V34_COLLISION_IK:-1}"
export V34_COLLISION_RADIUS="${V34_COLLISION_RADIUS:-0.16}"
export V34_COLLISION_STEPS="${V34_COLLISION_STEPS:-24}"
export V34_COLLISION_LR="${V34_COLLISION_LR:-0.025}"
export V34_COLLISION_WEIGHT="${V34_COLLISION_WEIGHT:-8.0}"
export V34_COLLISION_REG_WEIGHT="${V34_COLLISION_REG_WEIGHT:-0.45}"
export V34_COLLISION_TEMPORAL_WEIGHT="${V34_COLLISION_TEMPORAL_WEIGHT:-0.02}"
export V34_CONTACT_LOCK_POSTPROCESS="${V34_CONTACT_LOCK_POSTPROCESS:-1}"
export V34_CONTACT_LOCK_THRESHOLD="${V34_CONTACT_LOCK_THRESHOLD:-0.65}"
export V34_CONTACT_LOCK_MIN_FRAMES="${V34_CONTACT_LOCK_MIN_FRAMES:-8}"
export V34_CONTACT_LOCK_STRENGTH="${V34_CONTACT_LOCK_STRENGTH:-0.85}"
export V34_CONTACT_LOCK_SMOOTH_WINDOW="${V34_CONTACT_LOCK_SMOOTH_WINDOW:-11}"
export V34_CONTACT_LOCK_MAX_ROOT_CORRECTION="${V34_CONTACT_LOCK_MAX_ROOT_CORRECTION:-0.18}"
export V34_OUTPUT_SMOOTH="${V34_OUTPUT_SMOOTH:-0}"
export V34_OUTPUT_ROT_SMOOTH_WINDOW="${V34_OUTPUT_ROT_SMOOTH_WINDOW:-3}"
export V34_OUTPUT_SMOOTH_STRENGTH="${V34_OUTPUT_SMOOTH_STRENGTH:-0.20}"

# Fixed scientific render box.  Disable with EDGE_RENDER_FIXED_BOUNDS=0.
export EDGE_RENDER_FIXED_BOUNDS="${EDGE_RENDER_FIXED_BOUNDS:-1}"
export EDGE_RENDER_XLIM="${EDGE_RENDER_XLIM:--1.8,1.8}"
export EDGE_RENDER_YLIM="${EDGE_RENDER_YLIM:--1.8,1.8}"
export EDGE_RENDER_ZLIM="${EDGE_RENDER_ZLIM:--0.05,2.25}"

if [[ -z "${RUN_ID:-}" ]]; then
  export RUN_ID="v34_source_aware_rag_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_SOURCE_AWARE_RAG.txt

echo "[V34 SOURCE-AWARE RAG] RUN_ROOT=$RUN_ROOT"
echo "[V34 SOURCE-AWARE RAG] V34_INDEX_JSON=$V34_INDEX_JSON"
echo "[V34 SOURCE-AWARE RAG] V26_DURATION_INDEX_NPZ=$V26_DURATION_INDEX_NPZ"
echo "[V34 SOURCE-AWARE RAG] V26_HIERARCHY_INDEX_NPZ=$V26_HIERARCHY_INDEX_NPZ"

if [[ "${V34_TRAIN:-0}" == "1" ]]; then
  bash scripts/run_v34_full_research.sh "$@"
else
  bash scripts/launch_v34_rhythm_repair.sh "$@"
fi
