#!/usr/bin/env bash
# V46.52 direct replacement wrapper. The installer preserves the current script
# as scripts/run_v46_50_full_rebuild_retrain_v46_51_base.sh.
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"
source configs/v46_52_anatomy_research.env

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TAG
OUT_ROOT="${OUT_ROOT:-output/v46_52_anatomy_${RUN_TAG}}"
export OUT_ROOT

echo "========== V46.52 ANATOMY RESEARCH PROFILE =========="
echo "ROOT_DIR=$ROOT_DIR"
echo "OUT_ROOT=$OUT_ROOT"
echo "CHANGE_BVH_DIR=${CHANGE_BVH_DIR:-change}"
echo "PREFER_OFFICIAL_SMPL=$V46_52_PREFER_OFFICIAL_SMPL"
echo "REBUILD_RETARGET=$V46_51_REBUILD_RETARGET_CACHE"
echo "REBUILD_DB=$V46_51_REBUILD_EVENT_DB"
echo "RETRAIN_V44/V45/V46=$V46_51_RETRAIN_V44/$V46_51_RETRAIN_V45/$V46_51_RETRAIN_V46"

bash scripts/run_v46_50_full_rebuild_retrain_v46_51_base.sh "$@"

PY="${V46_51_PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
FINAL_NPY="${FINAL_NPY:-$OUT_ROOT/v46_51_final.npy}"
"$PY" tools/v46_52_audit_motion_contract.py \
  --input "$FINAL_NPY" \
  --out "$OUT_ROOT/final.v46_52_anatomy.json" \
  --csv "$OUT_ROOT/final.v46_52_anatomy.csv"

echo "========== V46.52 COMPLETE =========="
echo "FINAL_NPY=$FINAL_NPY"
echo "ANATOMY_AUDIT=$OUT_ROOT/final.v46_52_anatomy.json"
