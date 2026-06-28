# V40 Floor-Aware Leg-IK + Native-Floor RAG Patch

## What changes

1. `tools/v34_motion_quality_postprocess.py`
   - Adds floor-aware lower-leg IK over hip/knee/ankle 6D rotations.
   - Adds anatomical knee guard.
   - Adds swing-foot ankle pitch clamping for non-contact toe penetration.
   - Preserves V39B footplant gating and residual-only Butterworth filter.

2. `tools/v34_warp_aware_retrieval.py`
   - Adds V40 native-floor prior in retrieval scoring.
   - Penalizes snippets whose own FK feet are pathologically below their local floor estimate.

3. `tools/v40_native_floor_audit.py`
   - Audits a JSON event index and writes native floor fields.
   - Can quality-penalize or remove pathologic source events.

## Install

```bash
cd /mnt/data/V40_FloorAware_LegIK_NativeRAG_Patch
mkdir -p /tmp/V40_FloorAware_LegIK_NativeRAG_Patch
cp -r * /tmp/V40_FloorAware_LegIK_NativeRAG_Patch/
bash install_v40_floor_aware_patch.sh /home/disk/lsm/storage/EDGE
```

## Reprocess one output

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v40_floor_aware_env.sh
bash scripts/run_v40_reprocess_motion.sh output/v38_source_aware_full_train_20260625_212818/v34_contact_inr/dunhuangwu2_v26.npy
```

## Build native-floor clean index

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v40_floor_aware_env.sh
bash scripts/run_v40_native_floor_audit.sh data/v34_source_aware/v34_shared_event_index_source_aware.json
```

## Overnight retraining

```bash
cd /home/disk/lsm/storage/EDGE
nohup bash scripts/run_v40_overnight_source_aware_full_train.sh > output/v40_overnight_launcher.log 2>&1 &
```

