#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export PYTHONUNBUFFERED=1

RUN_ROOT="output/v46_44_edge_contract_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT" output

echo "[1/6] Apply V46.44 EDGE contract patch"
python tools/apply_v46_44_edge_contract_patch.py
python -m py_compile tools/v46_motionrag_diff.py render_from_npy.py tools/canonicalize_chang_e_bvh_rot_only_meter_v2.py

echo "[2/6] Canonicalize Chang-E BVH: scale root + offsets, remove non-root positions"
python tools/canonicalize_chang_e_bvh_rot_only_meter_v2.py \
  --in_dir change \
  --out_dir output/change_rot_only_meter_bvh_v46_44 \
  --root_scale 0.01

echo "[3/6] Verify loaded root ranges are meter-scale, not 0.01m-scale"
python - <<'PY'
from pathlib import Path
import numpy as np
from tools.v46_motionrag_diff import load_bvh_file
bad=[]
for p in sorted(Path('output/change_rot_only_meter_bvh_v46_44').glob('*.bvh')):
    m=load_bvh_file(p)[0]
    root=m[:,[4,5,6]]
    xz=root[:,[0,2]]
    xz_range=float(np.linalg.norm(xz.max(axis=0)-xz.min(axis=0)))
    y_range=float(root[:,1].max()-root[:,1].min())
    print(p.name, 'root_xz_range_loaded=', round(xz_range,4), 'root_y_range_loaded=', round(y_range,4))
    if xz_range < 0.05 and 'mediation' not in p.name:
        bad.append((p.name,xz_range))
if bad:
    raise SystemExit('Loaded root range is still too small for non-meditation files: '+str(bad))
PY

echo "[4/6] Patch V46.43 script to use canonicalized v46_44 BVH directory if script exists"
if [ -f scripts/run_v46_43_physics_consistent_full_retrain.sh ]; then
  python - <<'PY'
from pathlib import Path
p=Path('scripts/run_v46_43_physics_consistent_full_retrain.sh')
s=p.read_text(encoding='utf-8')
s=s.replace('--motion_dirs output/change_rot_only_meter_bvh \\', '--motion_dirs output/change_rot_only_meter_bvh_v46_44 \\')
s=s.replace('--motion_dirs change \\', '--motion_dirs output/change_rot_only_meter_bvh_v46_44 \\')
p.write_text(s, encoding='utf-8')
print('[PATCHED]', p)
PY
  bash -n scripts/run_v46_43_physics_consistent_full_retrain.sh
fi

echo "[5/6] Start V46.43 full retrain under corrected contract"
echo "Run root marker: $RUN_ROOT"
# Delegate to your existing V46.43 script after path patching.
bash scripts/run_v46_43_physics_consistent_full_retrain.sh

echo "[6/6] Done"
