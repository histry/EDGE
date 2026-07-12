#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/v46_motionrag_diff_config.json}"
OUT_ROOT="${OUT_ROOT:-output/v46_49_retarget_full_$(date +%Y%m%d_%H%M%S)}"
CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-change}"
RETARGET_DIR="${RETARGET_DIR:-$OUT_ROOT/retargeted_change}"
DB_DIR="${DB_DIR:-$OUT_ROOT/db_retargeted}"
DB_AESD="${DB_AESD:-$OUT_ROOT/events_aesd.npz}"
SPLIT_DIR="${SPLIT_DIR:-$OUT_ROOT/source_disjoint_splits}"
AUDIO="${AUDIO:-test_music_bank/dunhuangwu2.wav}"

V44_CKPT="${V44_CKPT:-$OUT_ROOT/v44_retarget_contrastive.pt}"
V45_CKPT="${V45_CKPT:-$OUT_ROOT/v45_retarget_gravity_refiner.pt}"
V46_CKPT="${V46_CKPT:-$OUT_ROOT/v46_retarget_gravity_diffusion.pt}"

OUT_NPY="${OUT_NPY:-$OUT_ROOT/dunhuangwu2_v46_49.npy}"
OUT_JSON="${OUT_JSON:-$OUT_ROOT/dunhuangwu2_v46_49.report.json}"
OUT_MP4="${OUT_MP4:-$OUT_ROOT/dunhuangwu2_v46_49.mp4}"

mkdir -p "$OUT_ROOT"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

echo "========== V46.49 FULL RETARGET / RETRAIN =========="
echo "ROOT_DIR=$ROOT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "CHANGE_BVH_DIR=$CHANGE_BVH_DIR"
echo "RETARGET_DIR=$RETARGET_DIR"
echo "DB_DIR=$DB_DIR"
echo "AUDIO=$AUDIO"
echo "====================================================="

echo "[1/12] Compile V46.49 files and patch V45/V46 gravity losses"
python -m py_compile \
  tools/v46_49_gravity_contract.py \
  tools/chang_e_edge_retarget.py \
  tools/v46_49_build_retarget_cache.py \
  tools/v46_49_audit_gravity_contract.py \
  tools/v46_49_boundary_closed_loop.py \
  tools/v46_49_make_final_mssd.py \
  render_from_npy.py
python tools/apply_v46_49_gravity_training_patch.py
python -m py_compile tools/v46_motionrag_diff.py

echo "[2/12] Optimization retarget raw Chang-E BVHs"
python tools/v46_49_build_retarget_cache.py \
  --in_dir "$CHANGE_BVH_DIR" \
  --out_dir "$RETARGET_DIR" \
  --overwrite

echo "[3/12] Hard gravity audit before event slicing"
python tools/v46_49_audit_gravity_contract.py \
  --motion_dir "$RETARGET_DIR" \
  --out "$OUT_ROOT/retarget_gravity_audit.json" \
  --csv "$OUT_ROOT/retarget_gravity_audit.csv"

echo "[4/12] Rebuild Event-RAG DB only from valid retargeted NPY"
rm -rf "$DB_DIR"
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  build-db \
  --motion_dirs "$RETARGET_DIR" \
  --out_db "$DB_DIR"

echo "[5/12] Hard gravity audit after event slicing"
python tools/v46_49_audit_gravity_contract.py \
  --db "$DB_DIR" \
  --out "$OUT_ROOT/db_gravity_audit.json" \
  --csv "$OUT_ROOT/db_gravity_audit.csv"

echo "[6/12] Build AESD"
python tools/v46_38_build_aesd_event_semantics.py \
  --db "$DB_DIR/events.npz" \
  --out "$DB_AESD" \
  --json "$OUT_ROOT/events_aesd_audit.json"

echo "[7/12] Source-disjoint leakage audit/splits"
rm -rf "$SPLIT_DIR"
if python tools/make_v46_47_source_disjoint_splits.py --help | grep -q -- "--out_dir"; then
  python tools/make_v46_47_source_disjoint_splits.py \
    --db "$DB_DIR" \
    --out_dir "$SPLIT_DIR" \
    --folds "${V46_49_FOLDS:-3}"
else
  python tools/make_v46_47_source_disjoint_splits.py \
    --db "$DB_DIR" \
    --out "$SPLIT_DIR" \
    --folds "${V46_49_FOLDS:-3}"
fi

echo "[8/12] Train V44 with real unpaired target/classical music"
read -r -a MUSIC_DIR_ARRAY <<< "${MUSIC_DIRS:-test_music_bank proxy_music data/v21_router_music_valid25}"
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-contrastive \
  --db "$DB_AESD" \
  --out "$V44_CKPT" \
  --unpaired_audio_dirs "${MUSIC_DIR_ARRAY[@]}"

echo "[9/12] Train gravity-regularised V45"
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-refiner \
  --db "$DB_AESD" \
  --out "$V45_CKPT" \
  --steps "${V46_49_V45_STEPS:-8000}"

echo "[10/12] Train gravity-regularised V46"
python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  train-diffusion \
  --db "$DB_AESD" \
  --out "$V46_CKPT" \
  --steps "${V46_49_V46_STEPS:-15000}"

echo "[11/12] Resolve strict final MSSD and generate"
FINAL_MSSD="${FINAL_MSSD:-}"
if [[ -z "$FINAL_MSSD" && -n "${V26_REPORT:-}" ]]; then
  FINAL_MSSD="$OUT_ROOT/dunhuangwu2_v46_49_final.mssd.json"
  python tools/v46_49_make_final_mssd.py \
    --from_v26_report "$V26_REPORT" \
    --audio "$AUDIO" \
    --out "$FINAL_MSSD"
elif [[ -z "$FINAL_MSSD" && -n "${PREVIOUS_REPORT:-}" ]]; then
  echo "[WARN] Using previous schedule only for a controlled conversion ablation."
  FINAL_MSSD="$OUT_ROOT/dunhuangwu2_v46_49_controlled_final.mssd.json"
  python tools/v46_49_make_final_mssd.py \
    --from_previous_report "$PREVIOUS_REPORT" \
    --audio "$AUDIO" \
    --out "$FINAL_MSSD"
fi
if [[ -z "$FINAL_MSSD" || ! -f "$FINAL_MSSD" ]]; then
  echo "ERROR: Set FINAL_MSSD or V26_REPORT. PREVIOUS_REPORT is allowed only for controlled ablation." >&2
  exit 3
fi

python tools/v46_49_boundary_closed_loop.py \
  --config "$CONFIG" \
  generate \
  --audio "$AUDIO" \
  --slots_json "$FINAL_MSSD" \
  --db "$DB_AESD" \
  --contrastive "$V44_CKPT" \
  --refiner "$V45_CKPT" \
  --diffusion "$V46_CKPT" \
  --out "$OUT_NPY" \
  --json "$OUT_JSON" \
  --render_output "$OUT_MP4" \
  --render_script render_from_npy.py

echo "[12/12] Final independent gravity + physics audit"
python tools/v46_49_audit_gravity_contract.py \
  --input "$OUT_NPY" \
  --out "$OUT_ROOT/final_gravity_audit.json" \
  --csv "$OUT_ROOT/final_gravity_audit.csv"

python tools/v46_motionrag_diff.py \
  --config "$CONFIG" \
  audit \
  --input "$OUT_NPY" \
  --json "$OUT_ROOT/final_motion_audit.json"

ls -lh "$OUT_NPY" "$OUT_JSON" "$OUT_MP4"
echo "DONE: $OUT_ROOT"
