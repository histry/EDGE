#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-1}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-1}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_ENABLE_ROOT_Y_PHYSICS="${V46_ENABLE_ROOT_Y_PHYSICS:-1}"
export V46_ROOT_Y_MIN_FLIGHT_FRAMES="${V46_ROOT_Y_MIN_FLIGHT_FRAMES:-3}"
export V46_ROOT_Y_MAX_FLIGHT_SECONDS="${V46_ROOT_Y_MAX_FLIGHT_SECONDS:-1.2}"
export V46_ROOT_Y_DAMPING_MAX_SECONDS="${V46_ROOT_Y_DAMPING_MAX_SECONDS:-0.28}"
export V46_IK_SLIDE_RELEASE_M="${V46_IK_SLIDE_RELEASE_M:-0.12}"
export V46_IK_CLOUD_STEP_SPEED_MPS="${V46_IK_CLOUD_STEP_SPEED_MPS:-0.15}"
export V46_IK_SLIDING_ANCHOR_WINDOW="${V46_IK_SLIDING_ANCHOR_WINDOW:-10}"
export V46_IK_CLOUD_SPEED_CV_MAX="${V46_IK_CLOUD_SPEED_CV_MAX:-1.75}"
export V46_IK_CLOUD_ROOT_MIN_TRAVEL_M="${V46_IK_CLOUD_ROOT_MIN_TRAVEL_M:-0.045}"
export V46_IK_CLOUD_DIRECTION_COS_MIN="${V46_IK_CLOUD_DIRECTION_COS_MIN:-0.35}"
export V46_IK_CLOUD_ROOT_FOOT_REL_MAX_M="${V46_IK_CLOUD_ROOT_FOOT_REL_MAX_M:-0.18}"
export V46_IK_CHUNK_OVERLAP="${V46_IK_CHUNK_OVERLAP:-24}"
export V46_BVH_RESAMPLE_TO_CONFIG_FPS="${V46_BVH_RESAMPLE_TO_CONFIG_FPS:-1}"
export V46_SOURCE_GROUP_MODE="${V46_SOURCE_GROUP_MODE:-filename}"
export V46_FILENAME_SEMANTIC_ENABLE="${V46_FILENAME_SEMANTIC_ENABLE:-1}"
export V46_FILENAME_SEMANTIC_WEIGHT="${V46_FILENAME_SEMANTIC_WEIGHT:-0.35}"
export V46_FILENAME_SEMANTIC_RETRIEVAL_WEIGHT="${V46_FILENAME_SEMANTIC_RETRIEVAL_WEIGHT:-0.20}"
export V46_FILENAME_SEMANTIC_OT_WEIGHT="${V46_FILENAME_SEMANTIC_OT_WEIGHT:-0.35}"

export V46_CLASSIFICATION_SEMANTIC_ENABLE="${V46_CLASSIFICATION_SEMANTIC_ENABLE:-1}"
export V46_CLASSIFICATION_SEMANTIC_RATIO="${V46_CLASSIFICATION_SEMANTIC_RATIO:-0.70}"
export V46_CLASSIFICATION_RETRIEVAL_WEIGHT="${V46_CLASSIFICATION_RETRIEVAL_WEIGHT:-0.34}"
export V46_CLASSIFICATION_OT_WEIGHT="${V46_CLASSIFICATION_OT_WEIGHT:-0.45}"
export V46_CLASSIFICATION_RETRIEVAL_BONUS="${V46_CLASSIFICATION_RETRIEVAL_BONUS:-0.28}"
export V46_CLASSIFICATION_REPORT_TOPK="${V46_CLASSIFICATION_REPORT_TOPK:-8}"
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED="${V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_DIRS="${V46_EXTERNAL_MUSIC_SEMANTIC_DIRS:-music_semantics:external_music_semantics:output/music_semantics}"
export V46_EXTERNAL_MUSIC_SEMANTIC_CMD="${V46_EXTERNAL_MUSIC_SEMANTIC_CMD:-}"
export V46_EXTERNAL_MUSIC_SEMANTIC_CACHE_DIR="${V46_EXTERNAL_MUSIC_SEMANTIC_CACHE_DIR:-output/v46_external_music_semantic_cache}"
export V46_EXTERNAL_MUSIC_SEMANTIC_WEIGHT="${V46_EXTERNAL_MUSIC_SEMANTIC_WEIGHT:-0.78}"
export V46_EXTERNAL_MUSIC_SEMANTIC_TEMPERATURE="${V46_EXTERNAL_MUSIC_SEMANTIC_TEMPERATURE:-0.65}"
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY="${V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY:-1}"
export V46_UNPAIRED_AUDIO_ENABLE="${V46_UNPAIRED_AUDIO_ENABLE:-1}"
export V46_UNPAIRED_AUDIO_SLOT_SECONDS="${V46_UNPAIRED_AUDIO_SLOT_SECONDS:-4.0}"
export V46_UNPAIRED_POSITIVE_TOPK="${V46_UNPAIRED_POSITIVE_TOPK:-8}"
export V46_UNPAIRED_PAIRS_PER_AUDIO_SLOT="${V46_UNPAIRED_PAIRS_PER_AUDIO_SLOT:-4}"

CFG="configs/v46_motionrag_diff_config.json"
RUN_ROOT="output/v46_12_external_music_semantic_motionrag_diff_$(date +%Y%m%d_%H%M%S)"
DB_DIR="$RUN_ROOT/db"
mkdir -p "$RUN_ROOT" "$DB_DIR" output
echo "$RUN_ROOT" > output/LATEST_V46_MOTIONRAG_DIFF.txt

MOTION_DIRS=()
for d in change data/motions data/dunhuang_motion data/processed output/v34_event_library output/v38_source_aware_full_train_20260625_212818; do
  [[ -e "$d" ]] && MOTION_DIRS+=("$d")
done
if [[ ${#MOTION_DIRS[@]} -eq 0 ]]; then
  echo "[V46.12 ERROR] no motion directory found. Put Chang-E BVH files under ./change." >&2
  exit 2
fi

MANIFEST_ARGS=()
for m in change/manifest.csv change/manifest.tsv manifest.csv data/manifest.csv; do
  if [[ -f "$m" ]]; then MANIFEST_ARGS=(--manifest "$m"); break; fi
done

AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music change/music change/audio; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done
SEMANTIC_DIRS=()
for d in music_semantics external_music_semantics output/music_semantics; do
  [[ -e "$d" ]] && SEMANTIC_DIRS+=("$d")
done
AUDIO="test_music_bank/dunhuangwu2.wav"
[[ -f "$AUDIO" ]] || AUDIO="data/music/dunhuangwu2.wav"
[[ -f "$AUDIO" ]] || AUDIO="custom_music/dunhuangwu2.wav"
if [[ ! -f "$AUDIO" ]]; then
  echo "[V46.12 ERROR] dunhuangwu2 audio not found under test_music_bank/, data/music/, or custom_music/." >&2
  exit 2
fi

python tools/v46_motionrag_diff.py --config "$CFG" build-db \
  --motion_dirs "${MOTION_DIRS[@]}" \
  "${MANIFEST_ARGS[@]}" \
  --audio_dirs "${AUDIO_DIRS[@]}" \
  --out_db "$DB_DIR"

MUSIC_SEMANTIC_ARGS=()
if [[ ${#SEMANTIC_DIRS[@]} -gt 0 ]]; then
  MUSIC_SEMANTIC_ARGS=(--music_semantic_dirs "${SEMANTIC_DIRS[@]}")
fi
python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$DB_DIR/events.npz" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --out "$RUN_ROOT/v44_contrastive.pt"

python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
  --db "$DB_DIR/events.npz" \
  --out "$RUN_ROOT/v45_refiner.pt"

python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
  --db "$DB_DIR/events.npz" \
  --out "$RUN_ROOT/v46_diffusion.pt"

OUT="$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.npy"
JSON="$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.report.json"
MP4="$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.mp4"
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$DB_DIR/events.npz" \
  --contrastive "$RUN_ROOT/v44_contrastive.pt" \
  --refiner "$RUN_ROOT/v45_refiner.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion.pt" \
  --out "$OUT" \
  --json "$JSON" \
  --render_output "$MP4"

echo "[V46.12 DONE]"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[MOTION]   $OUT"
echo "[REPORT]   $JSON"
[[ -f "$MP4" ]] && echo "[MP4]      $MP4"
