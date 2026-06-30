#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
source scripts/v41_2_native_floor_planner_env.sh

RUN_ROOT="${RUN_ROOT:-output/v41_2_fk_verified_native_floor_planner_d4_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V41_2_NATIVE_FLOOR_PLANNER_D4.txt
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

echo "[V41.2 START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[INDEX JSON] $V41_SOURCE_INDEX_JSON"
echo "[INDEX NPZ ] $V41_SOURCE_INDEX_NPZ"
echo "[FOOT FK JOINTS] $V41_NATIVE_FLOOR_FOOT_JOINTS"
echo "[TAU_SAFE] $V41_NATIVE_FLOOR_TAU_SAFE_M"
echo "[TAU_DEAD] $V41_NATIVE_FLOOR_TAU_DEAD_M"
echo "[ALPHA] $V41_NATIVE_FLOOR_ALPHA"
echo "[BETA] $V41_NATIVE_FLOOR_BETA"
echo "[MISSING POLICY] $V41_NATIVE_FLOOR_MISSING_POLICY"

python -m py_compile \
  tools/v34_warp_aware_retrieval.py \
  tools/v41b_inject_min_foot_y_to_db.py \
  tools/apply_v41_2_fk_verified_hotfix.py

AUG_DIR="$RUN_ROOT/v41_2_augmented_index"
mkdir -p "$AUG_DIR"

python tools/v41b_inject_min_foot_y_to_db.py \
  --index_json "$V41_SOURCE_INDEX_JSON" \
  --out_json "$AUG_DIR/v41_2_native_floor_augmented_index.json" \
  --audit_json "$AUG_DIR/v41_2_native_floor_barrier_audit.json" \
  --search_root data/v34_source_aware \
  --search_root data \
  --search_root output \
  --foot_joints "$V41_NATIVE_FLOOR_FOOT_JOINTS" \
  --max_missing_fraction "$V41_NATIVE_FLOOR_MAX_MISSING_FRACTION" \
  --overwrite_existing

python - <<PY
import json, os
audit="$AUG_DIR/v41_2_native_floor_barrier_audit.json"
a=json.load(open(audit))
s=a.get("summary",{})
print("[V41.2 AUDIT SUMMARY]", json.dumps(s, ensure_ascii=False))
if s.get("raw_column_mode") is not False:
    raise SystemExit("[ERROR] raw_column_mode must be False")
if s.get("scored",0) + s.get("reused",0) <= 0:
    raise SystemExit("[ERROR] no FK native-floor features scored")
PY

export V26_INDEX_JSON="$AUG_DIR/v41_2_native_floor_augmented_index.json"
export V34_INDEX_JSON="$AUG_DIR/v41_2_native_floor_augmented_index.json"
export V26_DURATION_INDEX_NPZ="$V41_SOURCE_INDEX_NPZ"
export V26_MUSIC="test_music_bank/dunhuangwu4.wav"
export V32_KEYS="dunhuangwu4"
export V26_OUT_DIR="$RUN_ROOT/v41_2_native_floor_planner"

bash scripts/run_v34_whole_song.sh

RAW="$V26_OUT_DIR/dunhuangwu4_v26.npy"
if [[ ! -f "$RAW" ]]; then
  echo "[ERROR] missing raw generated motion: $RAW" >&2
  exit 3
fi

bash scripts/run_v40_reprocess_motion.sh "$RAW"

SUMMARY="$V26_OUT_DIR/dunhuangwu4_v26_v40_floor_aware.motion_quality_postprocess.v40.json"
MOTION="$V26_OUT_DIR/dunhuangwu4_v26_v40_floor_aware.npy"
[[ -f "$SUMMARY" ]] && cp "$SUMMARY" "$V26_OUT_DIR/dunhuangwu4_v41_2_native_floor_planner.motion_quality_postprocess.v41_2.json"
[[ -f "$MOTION" ]] && cp "$MOTION" "$V26_OUT_DIR/dunhuangwu4_v41_2_native_floor_planner.npy"

echo "[V41.2 DONE] $(date)"
echo "[OUTPUT] $V26_OUT_DIR"
