# V46.50 Event-Level Heading State for EDGE

## Baseline

This package targets repository commit:

```text
da61593a5d5a0bbfa1e793f70e6044eea2b6a133
```

It assumes the validated V46.49.4 retargeter is already present.

## Why V46.50 is the correct next layer

V46.49.4 establishes a faithful target root orientation:

```text
corrected source heading -> target root orientation -> final EDGE151D
```

The remaining long turns belong to localized source intervals. They must not be
globally erased because Chang-E includes intentional rapid circular movements.
They must instead be represented as event semantics.

The current V46 database builder has motion-aware start candidates but still
slices fixed-length windows. The current closed-loop assembler aligns every
incoming event to the previous endpoint yaw. Neither component stores an event
entry heading, relative yaw delta, turn intent, or semantic yaw budget.

V46.50 introduces:

```text
V46.49.4 strict retarget cache
        ↓
motion-adaptive event segmentation
        ↓
event entry heading canonicalization
        ↓
turn intent / reset-drift classification
        ↓
semantic yaw budget
        ↓
heading-aware Event-RAG NPZ + AESD
        ↓
planner-owned stage heading state
        ↓
refiner/diffusion/IK planned-heading guard
```

## Files

```text
tools/v46_50_heading_contract.py
tools/v46_50_build_retarget_cache.py
tools/v46_50_build_event_heading_db.py
tools/v46_50_audit_event_heading_db.py
tools/v46_50_heading_closed_loop.py
tools/v46_50_audit_generated_heading.py
configs/v46_50_event_heading.env
scripts/run_v46_50_full_rebuild_retrain.sh
tests/test_v46_50_heading_contract.py
install_v46_50.sh
```

No wholesale replacement of `tools/v46_motionrag_diff.py` is performed. This
preserves the current V46.38 MSSD-AESD routing, V46.41–43 stability code,
V46.46 closed-loop boundary system, and V46.49.4 retarget fixes.

## Installation

```bash
unzip -o v46_50_event_heading_solution.zip -d /tmp
cd /tmp/v46_50_event_heading_solution
bash install_v46_50.sh /home/disk/lsm/storage/EDGE
```

## Formal environment

```bash
cd /home/disk/lsm/storage/EDGE
source configs/v46_50_event_heading.env
```

Set the final schedule descriptor:

```bash
export SLOTS_JSON="output/.../dunhuangwu2.final_mssd.json"
```

The V44 music pool intentionally excludes `test_music_bank`.

## One-shot rebuild, retrain and generation

```bash
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="output/v46_50_event_heading_${RUN_TAG}" \
nohup bash scripts/run_v46_50_full_rebuild_retrain.sh \
  > "output/v46_50_${RUN_TAG}.log" 2>&1 &

tail -f "output/v46_50_${RUN_TAG}.log"
```

## Hard acceptance gates

Retarget cache:

```text
nonroot_position_mode = ignore
heading mode = stabilize
root orientation mode = absolute_reference_lock
fit_rmse_p95 < 0.18 m
gravity pass
```

Event DB:

```text
entry heading p95 <= 5°
no invalid heading event saved
no non-turn event over its yaw budget
at least two source_uids
all event files exist
```

Whole-song output:

```text
planned-vs-final root heading error p95 <= 2°
no explicit spin in non-turn anchor slots
no non-turn core over its semantic yaw budget
gravity pass
boundary audit pass
```

Global `mechanical_spin_fail` is not used as a whole-song veto because an
intentional Sogdian-whirl event may legitimately contain a long turn. The hard
gate is event-semantic compatibility.
