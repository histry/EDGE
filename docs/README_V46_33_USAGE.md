# V46.33 Reference-Conditioned Transition-Masked MotionRAG-Diff

This package contains the code changes for the Dunhuang/Chang-E low-resource whole-song pipeline:

1. RAG retrieves real Dunhuang motion events.
2. Each music slot receives an exact target frame budget.
3. The event core is only lightly resampled.
4. Transition budget is reserved between adjacent events.
5. Root Hermite + rotation SLERP generate motion-space inbetweens.
6. Transition spans are saved as the seam/transition mask.
7. V45 refiner and V46 diffusion treat the RAG+inbetween output as a strong reference trajectory and only edit transition-mask regions by default.
8. V43 IK finalizes foot-ground contact.

## Files

- `tools/v46_33_reference_transition_patch.py`
- `tools/v46_33_relabel_change_event_db.py`
- `scripts/run_v46_33_reference_transition_overnight.sh`

## Install

Copy the files into the EDGE repo:

```bash
cd /home/disk/lsm/storage/EDGE
cp /path/to/tools/v46_33_reference_transition_patch.py tools/
cp /path/to/tools/v46_33_relabel_change_event_db.py tools/
cp /path/to/scripts/run_v46_33_reference_transition_overnight.sh scripts/
chmod +x tools/v46_33_reference_transition_patch.py tools/v46_33_relabel_change_event_db.py scripts/run_v46_33_reference_transition_overnight.sh
```

## Run overnight full pipeline

```bash
cd /home/disk/lsm/storage/EDGE
CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
V46_DEVICE=cuda \
V46_CONTRASTIVE_EPOCHS=160 \
V46_REFINER_TRAIN_STEPS=10000 \
V46_DIFFUSION_TRAIN_STEPS=20000 \
V46_DIFFUSION_STEPS=50 \
V46_TRANSITION_BUDGET_ENABLE=1 \
V46_TRANSITION_INBETWEEN_ENABLE=1 \
V46_TRANSITION_MIN_FRAMES=10 \
V46_TRANSITION_MAX_FRAMES=28 \
V46_TRANSITION_RATIO=0.18 \
V46_TRANSITION_MASK_HALO=6 \
V46_TRANSITION_MIN_CORE_FRAMES=30 \
V46_CORE_WARP_MIN=0.72 \
V46_CORE_WARP_MAX=1.38 \
V46_REFINER_CORE_STRENGTH=0.02 \
V46_DIFFUSION_CORE_STRENGTH=0.00 \
V46_DIFFUSION_TRANSITION_STRENGTH=0.72 \
bash scripts/run_v46_33_reference_transition_overnight.sh
```

The script creates:

- `output/v46_33_reference_transition_alltrain_*/train_all_db/events.npz`
- `v44_contrastive_v46_33.pt`
- `v45_refiner_v46_33.pt`
- `v46_diffusion_v46_33.pt`
- `dunhuangwu2_v46_33_ref_transition_refiner_ik.mp4`
- `dunhuangwu2_v46_33_ref_transition_diffusion_ik.mp4`
- `*.motion_ref.npy`
- `*.transition_mask.npy`
- `V46_33_FINAL_SUMMARY.json`

## Monitor

```bash
RUN_ROOT=$(ls -td output/v46_33_reference_transition_alltrain_* | head -1)
tail -f "$RUN_ROOT"/logs/v46_33_reference_transition_overnight_*.log
ps -p $(cat "$RUN_ROOT/logs/v46_33_reference_transition_overnight.pid")
```

## Important switches

- `V46_TRANSITION_BUDGET_ENABLE=1`: use V46.33 transition-budget reference builder.
- `V46_TRANSITION_INBETWEEN_ENABLE=1`: use root-Hermite + rotation-SLERP motion-space inbetweening.
- `V46_REFINER_CORE_STRENGTH=0.02`: almost lock core motion in refiner.
- `V46_DIFFUSION_CORE_STRENGTH=0.00`: fully lock core motion in diffusion.
- `V46_DIFFUSION_TRANSITION_STRENGTH=0.72`: allow diffusion residual generation inside transition mask.

