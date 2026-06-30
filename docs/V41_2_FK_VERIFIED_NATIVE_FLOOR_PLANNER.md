# V41.2 FK-Verified Native-Floor-Aware Planner

## Why V41.2 exists

V41.2 fixes the 151D index-alignment trap.  In this EDGE project, `[T,151]`
is not a direct global-position table.  Channels `[7:151]` are 24 joints in
6D rotation representation, so raw columns such as `7` or `10` are not foot-Y.

## Correct physical extraction

V41.2 uses:

```text
root = motion[:, [4,5,6]]
local rotations = motion[:, 7:151].reshape(T, 24, 6)
FK(root, rotations) -> joints[T,24,3]
foot_y = joints[:, [7,8,10,11], 1]
native_floor_penetration = quantile(foot_y, q) + margin - min(foot_y)
```

The FK joint ids `[7,8,10,11]` are skeleton joints, not raw 151D feature channels.

## Planner rule

Safe candidates pay zero penalty.  Barrier candidates pay a finite non-convex
penalty.  Dead candidates are removed from strict search, but if a slot would
become disconnected, V41.2 restores them with a large finite rescue penalty.

## Install

```bash
cd /mnt/data/V41_2_FK_Verified_Native_Floor_Planner_Patch
bash install_v41_2_native_floor_planner_patch.sh /home/disk/lsm/storage/EDGE
```

## Single dunhuangwu4 run

```bash
cd /home/disk/lsm/storage/EDGE
bash scripts/run_v41_2_native_floor_planner_d4.sh
```

## Overnight sweep

```bash
cd /home/disk/lsm/storage/EDGE
nohup bash scripts/run_v41_2_native_floor_planner_overnight_d4_sweep.sh \
  > output/v41_2_native_floor_planner_overnight_d4_launcher.log 2>&1 &
```

## Important switches

```bash
export V41_NATIVE_FLOOR_FOOT_JOINTS=7,8,10,11
export V41_NATIVE_FLOOR_TAU_SAFE_M=0.012
export V41_NATIVE_FLOOR_TAU_DEAD_M=0.052
export V41_NATIVE_FLOOR_RELAX_ON_EMPTY=1
export V41_NATIVE_FLOOR_DEAD_RESCUE_PENALTY=25.0
export V41_NATIVE_FLOOR_FAIL_ON_MISSING=1
```
