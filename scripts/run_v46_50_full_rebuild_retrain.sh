#!/usr/bin/env bash
# V46.53 direct replacement: full data rebuild, V44/V45/V46 retraining and
# geometry-aware whole-song closed loop.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"

[[ -f configs/v46_52_anatomy_research.env ]] || {
  echo "[FATAL] Missing V46.52 base profile: configs/v46_52_anatomy_research.env" >&2
  exit 2
}
[[ -f configs/v46_53_research.env ]] || {
  echo "[FATAL] Missing V46.53 profile: configs/v46_53_research.env" >&2
  exit 2
}
# shellcheck disable=SC1091
source configs/v46_52_anatomy_research.env
# shellcheck disable=SC1091
source configs/v46_53_research.env

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG
OUT_ROOT="${OUT_ROOT:-output/v46_53_research_${RUN_TAG}}"
export OUT_ROOT
export V46_53_GROUNDER_CKPT="${V46_53_GROUNDER_CKPT:-$OUT_ROOT/v46_53_dual_branch_grounder.pt}"

PY="${V46_51_PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
[[ -x "$PY" ]] || { echo "[FATAL] Python not executable: $PY" >&2; exit 2; }
[[ -f scripts/run_v46_50_full_rebuild_retrain_v46_51_base.sh ]] || {
  echo "[FATAL] Preserved V46.51 base launcher is missing." >&2
  exit 2
}
mkdir -p "$OUT_ROOT"

cat <<EOF
========== V46.53 RESEARCH PROFILE ==========
ROOT_DIR=$ROOT_DIR
OUT_ROOT=$OUT_ROOT
CHANGE_BVH_DIR=${CHANGE_BVH_DIR:-change}
FULL_REBUILD=$V46_53_FULL_REBUILD
REBUILD_RETARGET=${V46_51_REBUILD_RETARGET_CACHE:-0}
REBUILD_DB=${V46_51_REBUILD_EVENT_DB:-0}
RETRAIN_V44/V45/V46=${V46_51_RETRAIN_V44:-0}/${V46_51_RETRAIN_V45:-0}/${V46_51_RETRAIN_V46:-0}
GROUNDER_CKPT=$V46_53_GROUNDER_CKPT
GLOBAL_ROUTE=$V46_53_GLOBAL_ROUTE_ENABLE
BODY_PART_MASK=$V46_53_BODY_PART_MASK_ENABLE
=============================================
EOF

echo "========== 0. V46.53 CONTRACT TESTS =========="
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m unittest discover -s tests -p 'test_v46_53_*.py' -v

echo "========== 1. PRESERVED V46.51/V46.52 PIPELINE + V46.53 WRAPPERS =========="
bash scripts/run_v46_50_full_rebuild_retrain_v46_51_base.sh "$@"

FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/v46_51_final.npy}"
[[ -s "$FINAL_NPY" ]] || {
  echo "[FATAL] Final motion missing: $FINAL_NPY" >&2
  exit 2
}

echo "========== 2. FINAL ANATOMY AUDIT =========="
"$PY" tools/v46_52_audit_motion_contract.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.v46_52_anatomy.json" \
  --csv "$OUT_ROOT/final.v46_52_anatomy.csv"

echo "========== 3. FINAL INTRINSIC MOTION AUDIT =========="
"$PY" tools/v46_53_boundary_contract.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.v46_53_intrinsic.json" \
  --fps "${V46_51_FPS:-30}"

cat <<EOF
========== V46.53 COMPLETE ==========
FINAL_NPY=$FINAL_NPY
ANATOMY_AUDIT=$OUT_ROOT/final.v46_52_anatomy.json
INTRINSIC_AUDIT=$OUT_ROOT/final.v46_53_intrinsic.json
DURATION_AUDIT=$FINAL_NPY.v46_53_duration.json
GROUNDER_CKPT=$V46_53_GROUNDER_CKPT
=====================================
EOF
