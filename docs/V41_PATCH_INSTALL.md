# V41 Beat-Decoupled Support Stabilizer

Use this patch when V40 floor-aware IK passes penetration checks but the rendered
video still shows beat-synchronous micro jitter or subtle oily foot sliding.

## Install

```bash
bash install_v41_beat_support_patch.sh /home/disk/lsm/storage/EDGE
```

## Reprocess one motion

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v41_beat_support_env.sh
bash scripts/run_v41_reprocess_motion.sh output/.../dunhuangwu2_v26_v40_floor_aware.npy
```

Prefer using the V40 accepted file as input when you only want to remove visual
micro-sliding.  Use the raw `*_v26.npy` as input when you want V40+V41 in one pass.

## Key outputs

Check `beat_decoupled_support_stabilizer` in the JSON summary:

- `support_frame_ratio`
- `root_filter.root_delta_max`
- `final_footplant_relock.segments`
- `mean_contact_speed_before/after`

The goal is lower contact speed and lower visual micro jitter without increasing
`foot_penetration_min_m` or `foot_skate_p95_mpf`.
