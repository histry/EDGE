#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${PYTHONPATH:=$ROOT_DIR}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

OUT_ROOT="${OUT_ROOT:-output/v46_47_chang_e_contract_main_$(date +%Y%m%d_%H%M%S)}"
DB="${DB:-$OUT_ROOT/db}"
SPLIT_DIR="${SPLIT_DIR:-$DB/splits}"
CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-change}"
AUDIO="${AUDIO:-dunhuangwu2.wav}"
mkdir -p "$OUT_ROOT" "$DB"

echo "[V46.47] ROOT=$ROOT_DIR"
echo "[V46.47] OUT_ROOT=$OUT_ROOT"
echo "[V46.47] CHANGE_BVH_DIR=$CHANGE_BVH_DIR"
echo "[V46.47] DB=$DB"
echo "[V46.47] ROOT_ROT_MODE=${V46_47_BVH_ROOT_ROT_MODE:-${V46_45_BVH_ROOT_ROT_MODE:-yaw}}"

if [[ "${APPLY_PATCH:-1}" == "1" ]]; then
  echo "[1/10] Applying V46.47 Chang-E contract patch"
  python tools/apply_v46_47_chang_e_contract_patch.py
  python -m py_compile tools/v46_motionrag_diff.py
fi

if [[ "${AUDIT_BVH:-1}" == "1" ]]; then
  echo "[2/10] Auditing raw/canonical Chang-E BVH through patched loader"
  python tools/audit_v46_47_chang_e_contract.py \
    --bvh_dir "$CHANGE_BVH_DIR" \
    --out "$OUT_ROOT/v46_47_bvh_contract_audit.json" \
    --csv "$OUT_ROOT/v46_47_bvh_contract_audit.csv"
fi

if [[ "${REBUILD_DB:-1}" == "1" ]]; then
  echo "[3/10] Rebuilding V46 Event-RAG DB from Chang-E data"
  rm -rf "$DB"
  python tools/v46_motionrag_diff.py build-db \
    --motion_dirs "$CHANGE_BVH_DIR" \
    --out_db "$DB"
fi

if [[ "${MAKE_SPLITS:-1}" == "1" ]]; then
  echo "[4/10] Building source-disjoint splits"
  python tools/make_v46_47_source_disjoint_splits.py \
    --db "$DB" \
    --out_dir "$SPLIT_DIR" \
    --folds "${V46_47_FOLDS:-3}" \
    --group_key "${V46_47_GROUP_KEY:-source_group}"
fi

if [[ "${AUDIT_DB:-1}" == "1" ]]; then
  echo "[5/10] Auditing rebuilt DB events"
  python tools/audit_v46_47_chang_e_contract.py \
    --db "$DB" \
    --out "$OUT_ROOT/v46_47_db_contract_audit.json" \
    --csv "$OUT_ROOT/v46_47_db_contract_audit.csv"
fi

if [[ "${BUILD_AESD:-1}" == "1" ]]; then
  echo "[6/10] Building AESD event semantics if script exists"
  if [[ -f tools/v46_38_build_aesd_event_semantics.py ]]; then
    python tools/v46_38_build_aesd_event_semantics.py \
      --db "$DB" \
      --out_db "$DB" || echo "[V46.47 WARN] AESD builder returned non-zero; continuing for compatibility"
  else
    echo "[V46.47 WARN] tools/v46_38_build_aesd_event_semantics.py not found; skipping"
  fi
fi

MSSD_JSON="${MSSD_JSON:-$OUT_ROOT/dunhuangwu2_v46_47_mssd_slots.json}"
if [[ "${BUILD_MSSD:-1}" == "1" ]]; then
  echo "[7/10] Building MSSD music slot descriptor if script exists"
  if [[ -f tools/v46_38_build_music_semantic_slot_descriptor.py && -f "$AUDIO" ]]; then
    python tools/v46_38_build_music_semantic_slot_descriptor.py \
      --audio "$AUDIO" \
      --out "$MSSD_JSON" \
      --router_ckpt "${V26_ROUTER_CKPT:-}" \
      --planner_ckpt "${V26_PLANNER_CKPT:-}" \
      --v23_ckpt "${V26_V23_CKPT:-}" || echo "[V46.47 WARN] MSSD builder returned non-zero; generation will fallback if allowed"
  else
    echo "[V46.47 WARN] MSSD builder/audio not found; skipping"
  fi
fi

CONTRASTIVE="${CONTRASTIVE:-$OUT_ROOT/v44_contrastive.pt}"
REFINER="${REFINER:-$OUT_ROOT/v45_refiner.pt}"
DIFFUSION="${DIFFUSION:-$OUT_ROOT/v46_diffusion.pt}"

if [[ "${TRAIN_V44:-1}" == "1" ]]; then
  echo "[8/10] Training V44 contrastive on rebuilt DB"
  python tools/v46_motionrag_diff.py train-contrastive \
    --db "$DB" \
    --out "$CONTRASTIVE" \
    --epochs "${V44_EPOCHS:-60}"
fi

if [[ "${TRAIN_V45:-1}" == "1" ]]; then
  echo "[9/10] Training V45 transition refiner on rebuilt DB"
  python tools/v46_motionrag_diff.py train-refiner \
    --db "$DB" \
    --out "$REFINER" \
    --steps "${V45_STEPS:-3000}"
fi

if [[ "${TRAIN_V46:-1}" == "1" ]]; then
  echo "[10/10] Training V46 masked diffusion on rebuilt DB"
  python tools/v46_motionrag_diff.py train-diffusion \
    --db "$DB" \
    --out "$DIFFUSION" \
    --steps "${V46_STEPS:-3000}" \
    --diffusion_steps "${V46_DIFFUSION_STEPS:-120}"
fi

if [[ "${GENERATE:-1}" == "1" ]]; then
  echo "[Generate] Running main whole-song generation"
  OUT_NPY="$OUT_ROOT/dunhuangwu2_v46_47_main.npy"
  OUT_JSON="$OUT_ROOT/dunhuangwu2_v46_47_main.report.json"
  OUT_MP4="$OUT_ROOT/dunhuangwu2_v46_47_main.mp4"
  if [[ "${USE_V46_46_CLOSED_LOOP:-1}" == "1" && -f tools/v46_46_boundary_closed_loop.py ]]; then
    echo "[Generate] Using V46.46 closed-loop wrapper"
    python tools/v46_46_boundary_closed_loop.py generate \
      --audio "$AUDIO" \
      --slots_json "$MSSD_JSON" \
      --db "$DB" \
      --contrastive "$CONTRASTIVE" \
      --refiner "$REFINER" \
      --diffusion "$DIFFUSION" \
      --out "$OUT_NPY" \
      --json "$OUT_JSON" \
      --render_output "$OUT_MP4" || {
        echo "[V46.47 WARN] V46.46 closed-loop failed; falling back to v46_motionrag_diff.py generate"
        python tools/v46_motionrag_diff.py generate \
          --audio "$AUDIO" \
          --slots_json "$MSSD_JSON" \
          --db "$DB" \
          --contrastive "$CONTRASTIVE" \
          --refiner "$REFINER" \
          --diffusion "$DIFFUSION" \
          --out "$OUT_NPY" \
          --json "$OUT_JSON" \
          --render_output "$OUT_MP4"
      }
  else
    echo "[Generate] Using native V46 generate"
    python tools/v46_motionrag_diff.py generate \
      --audio "$AUDIO" \
      --slots_json "$MSSD_JSON" \
      --db "$DB" \
      --contrastive "$CONTRASTIVE" \
      --refiner "$REFINER" \
      --diffusion "$DIFFUSION" \
      --out "$OUT_NPY" \
      --json "$OUT_JSON" \
      --render_output "$OUT_MP4"
  fi
fi

echo "[V46.47 DONE] OUT_ROOT=$OUT_ROOT"
