# V40B Native-Floor Reroute Patch

## Goal

V40B stops increasing postprocess IK and instead cuts native floor-toxic events
out of the Event-RAG dictionary before planning.  The V34 graph planner is then
forced to reroute through physically healthier bridge motions.

## Important coordinate note

In this EDGE representation, foot Y is often around `-1.0`, so absolute
`foot_y < -0.08` is not a safe default.  V40B uses event-local native floor:

```text
floor_y = quantile(all lower-foot Y, q)
native_penetration = max(0, floor_y + margin - min(lower-foot Y))
```

Default hard remove threshold: `0.08 m`.

## Install

```bash
cd /mnt/data/V40B_Native_Floor_Reroute_Patch
bash install_v40b_native_floor_reroute_patch.sh /home/disk/lsm/storage/EDGE
```

## Run dunhuangwu4 reroute

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v40b_native_floor_env.sh

export V40B_SOURCE_INDEX_JSON=/path/to/v34_shared_event_index.json
export V40B_SOURCE_INDEX_NPZ=/path/to/v34_shared_event_index.npz
export V40B_MUSIC="test_music_bank/dunhuangwu4.wav"
export V40B_KEYS="dunhuangwu4"

bash scripts/run_v40b_native_floor_reroute.sh
```

If the source paths are omitted, the script tries common V34/V26 locations and
existing `V34_INDEX_JSON`, `V26_INDEX_JSON`, and `V26_DURATION_INDEX_NPZ`.

## Threshold sweep

Conservative:

```bash
export V40B_NATIVE_FLOOR_REMOVE_THRESHOLD=0.10
```

Default:

```bash
export V40B_NATIVE_FLOOR_REMOVE_THRESHOLD=0.08
```

Aggressive:

```bash
export V40B_NATIVE_FLOOR_REMOVE_THRESHOLD=0.06
```

If pruning aborts because too many events are removed, increase the threshold.
Use `V40B_FORCE_PRUNE=1` only after reading the audit JSON.

## Outputs

Latest run root:

```bash
cat output/LATEST_V40B_NATIVE_FLOOR_REROUTE.txt
```

Important files:

```text
<RUN_ROOT>/v40b_pruned_index/v40b_native_floor_prune_audit.json
<RUN_ROOT>/v40b_pruned_index/v40b_removed_event_ids.txt
<RUN_ROOT>/v40b_native_floor_reroute/dunhuangwu4_v26.npy
<RUN_ROOT>/v40b_native_floor_reroute/dunhuangwu4_v40b_native_floor_reroute.npy
<RUN_ROOT>/v40b_native_floor_reroute/dunhuangwu4_v40b_native_floor_reroute.motion_quality_postprocess.v40b.json
```

## Acceptance target

```text
accepted = True
foot_penetration_min_m > -0.03
foot_skate_p95_mpf <= 0.018
has_v41 = False
```
