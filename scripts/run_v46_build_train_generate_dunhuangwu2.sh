#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

# Runtime switches. Set any of them before launching to override config.
export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-1}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-1}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_ENABLE_ROOT_Y_PHYSICS="${V46_ENABLE_ROOT_Y_PHYSICS:-1}"
export V46_IK_SLIDE_RELEASE_M="${V46_IK_SLIDE_RELEASE_M:-0.12}"

CFG="configs/v46_motionrag_diff_config.json"
RUN_ROOT="output/v46_motionrag_diff_$(date +%Y%m%d_%H%M%S)"
DB_DIR="$RUN_ROOT/db"
mkdir -p "$RUN_ROOT" "$DB_DIR"
echo "$RUN_ROOT" > output/LATEST_V46_MOTIONRAG_DIFF.txt

# Prefer the new change dataset. Add old sources only if they exist.
MOTION_DIRS=()
for d in change data/motions data/dunhuang_motion data/processed output/v34_event_library output/v38_source_aware_full_train_20260625_212818; do
  if [[ -e "$d" ]]; then
    MOTION_DIRS+=("$d")
  fi
done
if [[ ${#MOTION_DIRS[@]} -eq 0 ]]; then
  echo "[V46 ERROR] no motion directory found. Put EDGE-151 .npy/.npz/.pkl files under ./change or pass a custom command." >&2
  exit 2
fi

AUDIO="test_music_bank/dunhuangwu2.wav"
if [[ ! -f "$AUDIO" ]]; then
  AUDIO="data/music/dunhuangwu2.wav"
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "[V46 ERROR] dunhuangwu2 audio not found under test_music_bank/ or data/music/." >&2
  exit 2
fi

python tools/v46_motionrag_diff.py --config "$CFG" build-db \
  --motion_dirs "${MOTION_DIRS[@]}" \
  --out_db "$DB_DIR"

python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$DB_DIR/events.npz" \
  --out "$RUN_ROOT/v44_contrastive.pt"

python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
  --db "$DB_DIR/events.npz" \
  --out "$RUN_ROOT/v45_refiner.pt"

python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
  --db "$DB_DIR/events.npz" \
  --out "$RUN_ROOT/v46_diffusion.pt"

OUT="$RUN_ROOT/dunhuangwu2_v46_MotionRAG_Diff.npy"
JSON="$RUN_ROOT/dunhuangwu2_v46_MotionRAG_Diff.report.json"
MP4="$RUN_ROOT/dunhuangwu2_v46_MotionRAG_Diff.mp4"
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --db "$DB_DIR/events.npz" \
  --contrastive "$RUN_ROOT/v44_contrastive.pt" \
  --refiner "$RUN_ROOT/v45_refiner.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion.pt" \
  --out "$OUT" \
  --json "$JSON" \
  --render_output "$MP4"

echo "[V46 DONE]"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[MOTION]   $OUT"
echo "[REPORT]   $JSON"
[[ -f "$MP4" ]] && echo "[MP4]      $MP4"
