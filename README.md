# V42.2 FINAL EDGE Physics Patch

V42.2 FINAL is a conservative physics post-processing and IK-target-generation patch for EDGE-style 151D motion.

It fixes the known fatal issues:

1. **Origin Snap Bug**: IK targets are initialized from native FK foot positions, never zeros.
2. **Floating Anchor Bug**: footplant anchors are sampled only inside contact segments, never from pre-contact flight frames.
3. **Root Y index alignment**: EDGE 151D root is `motion[:, 4:7]`; root Y is `motion[:, 5]`.
4. **No fake foot XYZ writeback**: output `.npy` only modifies legal root-level channels `[4,5,6]`; foot IK targets are saved separately as `.npz`.
5. **C1-safe root-Y weighting**: flight parabola uses a bell-shaped gate with zero weight at flight boundaries.
6. **Damping fuse-off**: landing damping stops immediately if the motion re-enters flight.
7. **Rollback-if-worse**: if safety metrics worsen, V42.2 writes the original motion and records rollback metadata.

## Install

```bash
cd /mnt/data/V42_2_FINAL_EDGE_PHYSICS_PATCH
bash install_v42_2_final_edge_physics_patch.sh /home/disk/lsm/storage/EDGE
```

## Run one-pass fix for dunhuangwu2

```bash
cd /home/disk/lsm/storage/EDGE
bash scripts/run_v42_2_fix_dunhuangwu2.sh
```

## Run sweep

```bash
cd /home/disk/lsm/storage/EDGE
nohup bash scripts/run_v42_2_dunhuangwu2_sweep.sh > output/v42_2_dunhuangwu2_sweep_launcher.log 2>&1 &
tail -f output/v42_2_dunhuangwu2_sweep_launcher.log
```

## Important output files

```text
$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.npy
$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_physics.json
$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_targets.npz
$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.mp4
```
