# V46.45 Upright Root Fix

This patch fixes full-body rolling/flipping caused by mapping Chang-E BVH `Hips` full pitch/roll directly into EDGE/SMPL root joint rotation.

Apply after V46.44 EDGE contract fix.

```bash
cd /home/disk/lsm/storage/EDGE
cp /mnt/data/v46_45_upright_solution/tools/*.py tools/
cp /mnt/data/v46_45_upright_solution/scripts/*.sh scripts/
chmod +x tools/apply_v46_45_upright_root_patch.py tools/audit_v46_45_root_upright.py scripts/run_v46_43_after_v46_45_upright_fix.sh

export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export V46_45_BVH_ROOT_ROT_MODE=yaw
python tools/apply_v46_45_upright_root_patch.py
python -m py_compile tools/v46_motionrag_diff.py
python tools/audit_v46_45_root_upright.py --bvh_dir output/change_rot_only_meter_bvh_v46_44
```

Then rebuild DB and retrain. Old DB/checkpoints are not valid because the BVH loader output has changed.

```bash
nohup bash scripts/run_v46_43_after_v46_45_upright_fix.sh \
  > output/v46_45_upright_root_fix_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
