# V46.47 Chang-E Contract / Split / Audit Solution

This package targets the latest `histry/EDGE` code path, especially `tools/v46_motionrag_diff.py`.
It is designed for the situation where Chang-E data has been re-cut and placed under `EDGE/change`.

## Why this package exists

The latest GitHub code already has V46.38 routing, V46.33 transition-budget stitching,
V46.41/KBO safety, and V46.44 root-scale / Rot6D contract fixes. However, the current
GitHub `tools/v46_motionrag_diff.py` does not expose the V46.45 upright-root environment
switch described in the experiment notes. If Chang-E Hips full root rotation is kept raw,
root pitch/roll can flip or roll the whole EDGE/SMPL-like skeleton.

This package adds the missing contract layer and the required dataset-split/audit tools:

- `tools/apply_v46_47_chang_e_contract_patch.py`
  - patches `tools/v46_motionrag_diff.py` in place;
  - adds `V46_45_BVH_ROOT_ROT_MODE` / `V46_47_BVH_ROOT_ROT_MODE` with modes `yaw`, `identity`, `raw`;
  - adds source-disjoint metadata arrays into `events.npz`.

- `tools/audit_v46_47_chang_e_contract.py`
  - audits raw/canonical BVH through the patched loader;
  - audits rebuilt Event-RAG DB events;
  - checks root XZ scale, root tilt, contact ratio, foot skate, jerk, finite FK.

- `tools/make_v46_47_source_disjoint_splits.py`
  - creates source-disjoint folds/splits from `events.npz`;
  - prevents random event-level leakage from the same BVH source.

- `scripts/run_v46_47_chang_e_contract_main.sh`
  - one-command workflow: patch, audit BVH, rebuild DB, make splits, audit DB, build AESD/MSSD, train V44/V45/V46, generate.

- `configs/v46_47_chang_e_contract.env.example`
  - environment switches for patch/rebuild/train/generate.

## Installation

From the EDGE root:

```bash
unzip -o /mnt/data/v46_47_chang_e_contract_solution.zip -d /mnt/data/
cp /mnt/data/v46_47_chang_e_contract_solution/tools/*.py tools/
cp /mnt/data/v46_47_chang_e_contract_solution/scripts/*.sh scripts/
cp /mnt/data/v46_47_chang_e_contract_solution/configs/*.example configs/
chmod +x tools/apply_v46_47_chang_e_contract_patch.py \
         tools/audit_v46_47_chang_e_contract.py \
         tools/make_v46_47_source_disjoint_splits.py \
         scripts/run_v46_47_chang_e_contract_main.sh
```

## Main experiment

```bash
cd /home/disk/lsm/storage/EDGE
source configs/v46_47_chang_e_contract.env.example
export CHANGE_BVH_DIR=change
export AUDIO=dunhuangwu2.wav
export V46_47_BVH_ROOT_ROT_MODE=yaw
export V46_45_BVH_ROOT_ROT_MODE=yaw
bash scripts/run_v46_47_chang_e_contract_main.sh \
  > output/v46_47_chang_e_contract_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Root rotation ablation

Main:

```bash
export V46_47_BVH_ROOT_ROT_MODE=yaw
```

Ablation without root rotation:

```bash
export V46_47_BVH_ROOT_ROT_MODE=identity
```

Diagnostic raw root rotation, likely to reproduce flips:

```bash
export V46_47_BVH_ROOT_ROT_MODE=raw
```

## Required checks after DB rebuild

```bash
python tools/audit_v46_47_chang_e_contract.py \
  --db "$DB" \
  --out output/v46_47_db_contract_audit.json \
  --csv output/v46_47_db_contract_audit.csv

python tools/make_v46_47_source_disjoint_splits.py \
  --db "$DB" \
  --out_dir "$DB/splits" \
  --folds 3 \
  --group_key source_group
```

Expected:

- `root_xz_range_m` should not collapse to centimeter scale.
- `root_tilt_p95_rad` should be near zero under `yaw` / `identity` mode.
- source-disjoint overlap audit should report `ok: true`.
- contact ratio should not be all zero or all one.
- old checkpoints should not be used as final main experiment after changing contract.

## Notes

This package intentionally does not replace the entire long `v46_motionrag_diff.py` file.
It patches only the missing root-upright guard and adds DB metadata. This is safer because
current `v46_motionrag_diff.py` contains multiple late-stage patches (V46.38, V46.41,
V46.43, V46.44, V46.45-local, V46.46-local in some environments). Replacing the whole file
would risk deleting local fixes.
