#!/usr/bin/env bash
set -euo pipefail

# V35 source-aware RAG database rebuild launcher.
#
# This script rebuilds the Event-RAG bank from the segmentation stage, not only
# by post-filtering an already-built index.  It is intended for the motion-only
# Dunhuang BVH setting where file names encode:
#   dancer_repeat_Take_category
#
# Required if the default input path is not correct:
#   export V35_MOTION_INPUT_DIR=/path/to/151d_motion_files
#
# Optional full retraining:
#   export V35_RUN_FULL_RESEARCH=1
#   export V34_TRAIN=1

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

mkdir -p data output

if [[ -z "${RUN_ID:-}" ]]; then
  export RUN_ID="v35_source_aware_segmentation_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V35_SOURCE_AWARE_SEGMENTATION.txt

if [[ -z "${V35_MOTION_INPUT_DIR:-}" ]]; then
  candidate_dirs=(
    "data/dunhuang_dynamic_event_rag_physical"
    "data/dunhuang_151d"
    "data/dunhuang_motion_151d"
    "data/motions"
  )
  for candidate in "${candidate_dirs[@]}"; do
    if [[ -d "$candidate" ]]; then
      export V35_MOTION_INPUT_DIR="$candidate"
      break
    fi
  done
fi

if [[ -z "${V35_MOTION_INPUT_DIR:-}" || ! -d "$V35_MOTION_INPUT_DIR" ]]; then
  echo "[ERROR] Set V35_MOTION_INPUT_DIR to the directory containing 151D .pkl/.npy/.npz motion files." >&2
  exit 2
fi

V35_ROOT="${V35_ROOT:-data/v35_source_aware}"
V35_EVENT_DB_DIR="${V35_EVENT_DB_DIR:-$V35_ROOT/dynamic_event_rag}"
V35_EVENT_INDEX_JSON="${V35_EVENT_INDEX_JSON:-$V35_EVENT_DB_DIR/index_dynamic_event_source_aware.json}"
V35_SHARED_PREFIX="${V35_SHARED_PREFIX:-$V35_ROOT/v21_shared_event_index_source_aware}"
V35_SHARED_JSON="${V35_SHARED_JSON:-$V35_SHARED_PREFIX.json}"
V35_SHARED_NPZ="${V35_SHARED_NPZ:-$V35_SHARED_PREFIX.npz}"
V35_DURATION_NPZ="${V35_DURATION_NPZ:-$V35_ROOT/v26_music_dominant_duration_index_source_aware.npz}"
V35_DURATION_JSON="${V35_DURATION_JSON:-$V35_ROOT/v26_music_dominant_duration_index_source_aware.json}"
V35_HIER_NPZ="${V35_HIER_NPZ:-$V35_ROOT/v34_hierarchical_event_index_source_aware.npz}"
V35_HIER_JSON="${V35_HIER_JSON:-$V35_ROOT/v34_hierarchical_event_index_source_aware.json}"
V35_CONTACT_CACHE="${V35_CONTACT_CACHE:-$V35_ROOT/v33_event_contact_cache_source_aware.npz}"
V35_CONTACT_REPORT="${V35_CONTACT_REPORT:-$V35_ROOT/v33_event_contact_cache_source_aware.json}"
V35_TRANSITION_DATA="${V35_TRANSITION_DATA:-$V35_ROOT/v33_transition_dataset_source_aware.npz}"

python -m py_compile \
  tools/v34_source_aware_rag.py \
  tools/v35_source_aware_segmentation.py \
  tools/build_v21_shared_event_index.py \
  tools/build_v26_duration_index.py \
  tools/build_v26_hierarchical_event_index.py \
  tools/build_v33_event_contact_cache.py \
  tools/build_v27_transition_diffusion_dataset.py

echo "[V35 SOURCE-AWARE SEGMENTATION] RUN_ROOT=$RUN_ROOT"
echo "[V35 SOURCE-AWARE SEGMENTATION] input=$V35_MOTION_INPUT_DIR"
echo "[V35 SOURCE-AWARE SEGMENTATION] out=$V35_ROOT"

if [[ "${V35_REBUILD_EVENT_DB:-1}" == "1" || ! -f "$V35_EVENT_INDEX_JSON" ]]; then
  python tools/v35_source_aware_segmentation.py \
    --input_dir "$V35_MOTION_INPUT_DIR" \
    --out_dir "$V35_EVENT_DB_DIR" \
    --report "$V35_EVENT_DB_DIR/audit_source_aware_segmentation.json" \
    --min_len "${V35_SEG_MIN_LEN:-24}" \
    --ideal_len "${V35_SEG_IDEAL_LEN:-48}" \
    --max_len "${V35_SEG_MAX_LEN:-72}" \
    --complex_max_len "${V35_SEG_COMPLEX_MAX_LEN:-96}" \
    --boundary_min_gap "${V35_SEG_BOUNDARY_MIN_GAP:-18}" \
    --energy_smooth "${V35_SEG_ENERGY_SMOOTH:-7}" \
    --enable_context_window "${V35_ENABLE_CONTEXT_WINDOW:-1}" \
    --macro_energy_window "${V35_MACRO_ENERGY_WINDOW:-90}" \
    --macro_high_percentile "${V35_MACRO_HIGH_PERCENTILE:-72}" \
    --save_canonical_len "${V35_SEG_CANONICAL_LEN:-48}" \
    --fps "${V35_FPS:-30}" \
    --min_motion_density "${V35_MIN_MOTION_DENSITY:-0.0045}" \
    --near_duplicate_iou "${V35_NEAR_DUP_IOU:-0.72}" \
    --enable_motion_nms "${V35_ENABLE_MOTION_NMS:-1}" \
    --motion_nms_similarity "${V35_MOTION_NMS_SIMILARITY:-0.94}" \
    --motion_nms_recent_window "${V35_MOTION_NMS_RECENT_WINDOW:-6}" \
    --enable_density_resample "${V35_ENABLE_DENSITY_RESAMPLE:-1}" \
    --max_pose_hold_ratio "${V35_MAX_POSE_HOLD_RATIO:-0.12}" \
    --max_calm_flow_ratio "${V35_MAX_CALM_FLOW_RATIO:-0.22}" \
    --max_neutral_flow_ratio "${V35_MAX_NEUTRAL_FLOW_RATIO:-0.34}" \
    --min_keep_per_limited_type "${V35_MIN_KEEP_PER_LIMITED_TYPE:-24}" \
    --max_per_source_before_global_cap "${V35_MAX_PER_SOURCE_BEFORE_CAP:-160}" \
    --cap_per_source_uid "${V35_CAP_PER_SOURCE_UID:-96}" \
    --category_cap_factor "${V35_CATEGORY_CAP_FACTOR:-1.35}" \
    --repeat_cap_factor "${V35_REPEAT_CAP_FACTOR:-1.55}" \
    --dancer_cap_factor "${V35_DANCER_CAP_FACTOR:-1.45}" \
    --max_events "${V35_SEG_MAX_EVENTS:-0}" \
    --allow_event_files "${V35_ALLOW_EVENT_FILES:-0}" \
    --limit_files "${V35_LIMIT_FILES:-0}" \
    --quality_top_k "${V35_QUALITY_TOP_K:-0}"
fi

if [[ "${V35_REBUILD_SHARED_INDEX:-1}" == "1" || ! -f "$V35_SHARED_JSON" || ! -f "$V35_SHARED_NPZ" ]]; then
  python tools/build_v21_shared_event_index.py \
    --input_db "$V35_EVENT_INDEX_JSON" \
    --output_prefix "$V35_SHARED_PREFIX" \
    --max_events "${V21_MAX_EVENTS:-7000}" \
    --min_style_percentile "${V21_MIN_STYLE_PERCENTILE:-5}" \
    --min_quality "${V21_MIN_QUALITY:-0.0}" \
    --min_safety "${V21_MIN_SAFETY:-0.0}" \
    --family_span "${V21_FAMILY_SPAN:-600}" \
    --mmr_dim "${V21_MMR_DIM:-64}" \
    --source_aware 1 \
    --cap_per_source_uid "${V35_SHARED_CAP_PER_SOURCE_UID:-72}" \
    --category_cap_factor "${V35_SHARED_CATEGORY_CAP_FACTOR:-1.35}" \
    --repeat_cap_factor "${V35_SHARED_REPEAT_CAP_FACTOR:-1.55}" \
    --dancer_cap_factor "${V35_SHARED_DANCER_CAP_FACTOR:-1.45}"
fi

if [[ "${V35_BUILD_DURATION_INDEX:-1}" == "1" || ! -f "$V35_DURATION_NPZ" ]]; then
  python tools/build_v26_duration_index.py \
    --index_json "$V35_SHARED_JSON" \
    --index_npz "$V35_SHARED_NPZ" \
    --v23_checkpoint "${V23_DURATION_CKPT:-output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt}" \
    --out_npz "$V35_DURATION_NPZ" \
    --out_json "$V35_DURATION_JSON" \
    --device "${V35_DURATION_DEVICE:-cuda}"
else
  cp "$V35_SHARED_NPZ" "$V35_DURATION_NPZ"
fi

if [[ "${V35_BUILD_HIERARCHY:-1}" == "1" || ! -f "$V35_HIER_NPZ" ]]; then
  python tools/build_v26_hierarchical_event_index.py \
    --index_json "$V35_SHARED_JSON" \
    --duration_index_npz "$V35_DURATION_NPZ" \
    --out_npz "$V35_HIER_NPZ" \
    --out_json "$V35_HIER_JSON" \
    --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt}"
fi

if [[ "${V35_BUILD_CONTACT_CACHE:-1}" == "1" || ! -f "$V35_CONTACT_CACHE" ]]; then
  python tools/build_v33_event_contact_cache.py \
    --index_json "$V35_SHARED_JSON" \
    --duration_index_npz "$V35_DURATION_NPZ" \
    --out_npz "$V35_CONTACT_CACHE" \
    --out_json "$V35_CONTACT_REPORT" \
    --device "${V33_CONTACT_DEVICE:-cuda:0}" \
    --batch_size "${V33_CONTACT_BATCH_SIZE:-96}" \
    --fps "${V35_FPS:-30}" \
    --target_rates "${V33_CONTACT_TARGET_RATES:-0.42,0.42,0.38,0.38}" \
    --transition_penalty "${V33_CONTACT_TRANSITION_PENALTY:-1.40}" \
    --min_run "${V33_CONTACT_MIN_RUN:-2}" \
    --max_gap "${V33_CONTACT_MAX_GAP:-2}" \
    --probability_temperature "${V33_CONTACT_TEMPERATURE:-0.20}" \
    --existing_contact_policy "${V33_EXISTING_CONTACT_POLICY:-auto}" \
    --min_overall_rate "${V33_MIN_CONTACT_RATE:-0.05}" \
    --max_overall_rate "${V33_MAX_CONTACT_RATE:-0.75}" \
    --max_all_four_rate "${V33_MAX_ALL_FOUR_RATE:-0.55}"
fi

if [[ "${V35_BUILD_TRANSITION_DATASET:-1}" == "1" || ! -f "$V35_TRANSITION_DATA" ]]; then
  python tools/build_v27_transition_diffusion_dataset.py \
    --index_json "$V35_SHARED_JSON" \
    --duration_index_npz "$V35_DURATION_NPZ" \
    --out_npz "$V35_TRANSITION_DATA" \
    --event_contact_cache "$V35_CONTACT_CACHE" \
    --require_event_contacts 1 \
    --assert_contact_consistency 1 \
    --max_len "${V32_MAX_LEN:-120}" \
    --min_len "${V32_MIN_LEN:-8}" \
    --samples_per_event "${V32_SAMPLES_PER_EVENT:-6}" \
    --real_masks_per_boundary "${V32_REAL_MASKS_PER_BOUNDARY:-4}" \
    --source_pairs_per_event "${V32_SOURCE_PAIRS_PER_EVENT:-0.75}" \
    --condition_dropout "${V32_DATA_CONDITION_DROPOUT:-0.08}" \
    --pseudo_max_pose_deg "${V32_PSEUDO_MAX_POSE_DEG:-28}" \
    --pseudo_max_velocity_deg_s "${V32_PSEUDO_MAX_VELOCITY_DEG_S:-160}" \
    --pseudo_max_root_y "${V32_PSEUDO_MAX_ROOT_Y:-0.10}" \
    --pseudo_max_contact_jump "${V32_PSEUDO_MAX_CONTACT_JUMP:-0.25}" \
    --seed "${V32_SEED:-20260610}"
fi

export V34_INDEX_JSON="$V35_SHARED_JSON"
export V26_DURATION_INDEX_NPZ="$V35_DURATION_NPZ"
export V26_HIERARCHY_INDEX_NPZ="$V35_HIER_NPZ"
export V33_EVENT_CONTACT_CACHE="$V35_CONTACT_CACHE"
export V32_DATASET="$V35_TRANSITION_DATA"
export V34_DATASET="$V35_TRANSITION_DATA"

export V34_SOURCE_AWARE_RAG="${V34_SOURCE_AWARE_RAG:-1}"
export V34_RHYTHM_DEGRADATION_PENALTY="${V34_RHYTHM_DEGRADATION_PENALTY:-1}"
export V34_BOUNDARY_COMPAT="${V34_BOUNDARY_COMPAT:-1}"
export V34_COMPAT_DENSE_SCORE="${V34_COMPAT_DENSE_SCORE:-1}"
export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_INPAINT_KINETIC_ADAPTIVE_GATE="${V34_INPAINT_KINETIC_ADAPTIVE_GATE:-1}"
export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"
export V34_EDGE_HUB_RESCUE="${V34_EDGE_HUB_RESCUE:-1}"
export V34_SOURCE_DIVERSE_RESCUE="${V34_SOURCE_DIVERSE_RESCUE:-1}"
export V34_MEMORY_IGNORE_RELAXED="${V34_MEMORY_IGNORE_RELAXED:-1}"
export V34_BUILD_EVENT_LIBRARY="${V34_BUILD_EVENT_LIBRARY:-1}"
export V34_OVERWRITE_EVENT_LIBRARY="${V34_OVERWRITE_EVENT_LIBRARY:-1}"

echo "[V35 SOURCE-AWARE SEGMENTATION] shared_json=$V34_INDEX_JSON"
echo "[V35 SOURCE-AWARE SEGMENTATION] duration_npz=$V26_DURATION_INDEX_NPZ"
echo "[V35 SOURCE-AWARE SEGMENTATION] hierarchy_npz=$V26_HIERARCHY_INDEX_NPZ"
echo "[V35 SOURCE-AWARE SEGMENTATION] contact_cache=$V33_EVENT_CONTACT_CACHE"
echo "[V35 SOURCE-AWARE SEGMENTATION] transition_dataset=$V32_DATASET"

if [[ "${V35_RUN_FULL_RESEARCH:-0}" == "1" ]]; then
  bash scripts/run_v34_full_research.sh "$@"
else
  echo "[DONE] V35 source-aware RAG database rebuilt. Set V35_RUN_FULL_RESEARCH=1 V34_TRAIN=1 to train after rebuild."
fi
