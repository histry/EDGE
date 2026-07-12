#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"

source configs/v46_50_event_heading.env

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-output/v46_50_event_heading_${RUN_TAG}}"
RETARGET_CACHE="${RETARGET_CACHE:-$OUT_ROOT/retarget_cache}"
DB_DIR="${DB_DIR:-$OUT_ROOT/event_heading_db}"
DB="${DB:-$DB_DIR/events.npz}"
DB_AESD="${DB_AESD:-$DB_DIR/events_aesd.npz}"
V44_CKPT="${V44_CKPT:-$OUT_ROOT/v44_heading_contrastive.pt}"
V45_CKPT="${V45_CKPT:-$OUT_ROOT/v45_heading_refiner.pt}"
V46_CKPT="${V46_CKPT:-$OUT_ROOT/v46_heading_diffusion.pt}"
FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/v46_50_final.npy}"
FINAL_REPORT="${FINAL_REPORT:-$OUT_ROOT/v46_50_final.report.json}"
FINAL_MP4="${FINAL_MP4:-$OUT_ROOT/v46_50_final.mp4}"

mkdir -p "$OUT_ROOT" "$DB_DIR"

if [[ ! -e "${SLOTS_JSON:-}" ]]; then
  echo "[FATAL] Formal V46.50 generation requires SLOTS_JSON pointing to a final MSSD/schedule." >&2
  echo "Example: export SLOTS_JSON=output/.../dunhuangwu2.final_mssd.json" >&2
  exit 2
fi

echo "========== V46.50 PATHS =========="
printf "OUT_ROOT=%s\nRETARGET_CACHE=%s\nDB=%s\nDB_AESD=%s\nAUDIO=%s\nSLOTS_JSON=%s\n" \
  "$OUT_ROOT" "$RETARGET_CACHE" "$DB" "$DB_AESD" "$AUDIO" "$SLOTS_JSON"

echo "========== 1. STRICT V46.49.4 RETARGET CACHE =========="
python tools/v46_50_build_retarget_cache.py \
  --in_dir "$CHANGE_BVH_DIR" \
  --out_dir "$RETARGET_CACHE" \
  --overwrite

echo "========== 2. GRAVITY AUDIT RETARGET CACHE =========="
python tools/v46_49_audit_gravity_contract.py \
  --input "$RETARGET_CACHE" \
  --out "$OUT_ROOT/retarget_cache.gravity.json" \
  --csv "$OUT_ROOT/retarget_cache.gravity.csv"

echo "========== 3. MOTION-ADAPTIVE HEADING-AWARE EVENT DB =========="
python tools/v46_50_build_event_heading_db.py \
  --config "$CONFIG" \
  --motion_dirs "$RETARGET_CACHE" \
  --out_db "$DB_DIR" \
  --overwrite

echo "========== 4. EVENT HEADING DB HARD AUDIT =========="
python tools/v46_50_audit_event_heading_db.py \
  --db "$DB" \
  --out "$OUT_ROOT/event_heading_db.audit.json" \
  --csv "$OUT_ROOT/event_heading_db.audit.csv"

echo "========== 5. AESD ENRICHMENT (PRESERVES V46.50 ARRAYS) =========="
python tools/v46_38_build_aesd_event_semantics.py \
  --db "$DB" \
  --out "$DB_AESD" \
  --json "$OUT_ROOT/aesd_build.json"

echo "========== 6. TRAIN V44 ON UNPAIRED NON-TEST MUSIC =========="
read -r -a MUSIC_DIR_ARRAY <<< "$MUSIC_DIRS"
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-contrastive \
  --db "$DB_AESD" \
  --out "$V44_CKPT" \
  --unpaired_audio_dirs "${MUSIC_DIR_ARRAY[@]}" \
  --epochs "${V44_EPOCHS:-120}"

echo "========== 7. TRAIN V45 ON CANONICAL EVENT CORES =========="
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-refiner \
  --db "$DB_AESD" \
  --out "$V45_CKPT" \
  --steps "${V45_STEPS:-8000}"

echo "========== 8. TRAIN V46 ON CANONICAL EVENT CORES =========="
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-diffusion \
  --db "$DB_AESD" \
  --out "$V46_CKPT" \
  --steps "${V46_STEPS:-15000}" \
  --diffusion_steps "${V46_DIFFUSION_STEPS:-50}"

echo "========== 9. HEADING-STATE CLOSED-LOOP GENERATION =========="
python tools/v46_50_heading_closed_loop.py \
  generate \
  --config "$CONFIG" \
  --audio "$AUDIO" \
  --slots_json "$SLOTS_JSON" \
  --db "$DB_AESD" \
  --contrastive "$V44_CKPT" \
  --refiner "$V45_CKPT" \
  --diffusion "$V46_CKPT" \
  --out "$FINAL_NPY" \
  --json "$FINAL_REPORT"

echo "========== 10. FINAL GRAVITY AUDIT =========="
python tools/v46_49_audit_gravity_contract.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.gravity.json" \
  --csv "$OUT_ROOT/final.gravity.csv"

echo "========== 11. FINAL PLANNER HEADING AUDIT =========="
python tools/v46_50_audit_generated_heading.py \
  --motion "$FINAL_NPY" \
  --report "$FINAL_REPORT" \
  --db "$DB_AESD" \
  --out "$OUT_ROOT/final.heading.json" \
  --csv "$OUT_ROOT/final.heading.csv"

echo "========== 12. FIXED-CAMERA RENDER =========="
python render_from_npy.py \
  --motion "$FINAL_NPY" \
  --audio "$AUDIO" \
  --output "$FINAL_MP4" \
  --camera_mode fixed \
  --render_smooth_window 1 \
  --gravity_audit_json "$OUT_ROOT/final.render_gravity.json"

echo "========== V46.50 COMPLETE =========="
ls -lh \
  "$DB" \
  "$DB_AESD" \
  "$V44_CKPT" \
  "$V45_CKPT" \
  "$V46_CKPT" \
  "$FINAL_NPY" \
  "$FINAL_REPORT" \
  "$FINAL_MP4"
