#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export PYTHONUNBUFFERED=1

# Use V46.44 root-scale / rot6d contract fix plus V46.45 upright-root guard.
export V46_45_BVH_ROOT_ROT_MODE="${V46_45_BVH_ROOT_ROT_MODE:-yaw}"
export V46_43_ENABLE_HN_DPO_FINETUNE="${V46_43_ENABLE_HN_DPO_FINETUNE:-0}"

# Delegate to the V46.44 fixed retraining script if present. It must use
# output/change_rot_only_meter_bvh_v46_44 as --motion_dirs.
if [[ ! -f scripts/run_v46_43_after_v46_44_contract_fix.sh ]]; then
  echo "[ERROR] scripts/run_v46_43_after_v46_44_contract_fix.sh not found. Install V46.44 package first." >&2
  exit 1
fi

python tools/apply_v46_45_upright_root_patch.py
python tools/audit_v46_45_root_upright.py --bvh_dir output/change_rot_only_meter_bvh_v46_44
bash scripts/run_v46_43_after_v46_44_contract_fix.sh
