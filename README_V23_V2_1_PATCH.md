# V23-v2.1 Full-Body Natural-Duration Patch

## Why the previous dataset failed

The previous builder produced only 2,104 or 4,811 samples, and most samples were
in the shortest duration bin. Two mechanisms caused this:

1. the detector retained only a narrow high-speed root-yaw core, then expanded
   many events to exactly the minimum duration;
2. the online equal-bin collector discarded common short-bin records while
   sparse long bins could not fill their quota.

Lowering the validation assertion would hide the problem and should not be done.

## What v2.1 changes

- detects a full turn phrase using root yaw + upper/torso/lower activity + root
  movement + contact changes;
- includes preparation and recovery around the root-yaw core;
- rejects genuinely too-short phrases instead of clamping them all to one value;
- creates adaptive duration bins from observed unique durations (`auto:6`);
- uses a two-pass allocation scheme and always builds exactly `V23_MAX_SAMPLES`;
- controls the final identity fraction exactly;
- bounds the *effective* compression factor, including rounding for short events;
- retains the inference-safe 17D condition from observed motion only.

## Install

```bash
bash install_patch.sh /home/disk/lsm/storage/EDGE
```

## Remove the failed dataset

```bash
cd /home/disk/lsm/storage/EDGE
rm -f \
  data/v23_v2_natural_duration_dataset.npz \
  data/v23_v2_natural_duration_dataset.metadata.json
```

## Recommended 6k validation run

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export CUDA_VISIBLE_DEVICES=0
export V23_REBUILD_DATASET=1
export V23_MAX_SAMPLES=6000
export V23_EPOCHS=80
export V23_PATIENCE=30
bash scripts/launch_v23_v2_full.sh
```

The dataset stage must report exactly 6,000 samples and pass before training.

## Recommended full run

```bash
unset V23_MAX_SAMPLES V23_EPOCHS V23_PATIENCE
export V23_REBUILD_DATASET=1
export V23_MAX_SAMPLES=12000
export V23_EPOCHS=600
export V23_PATIENCE=100
bash scripts/launch_v23_v2_full.sh
```

## Dataset acceptance criteria

- exact requested sample count;
- identity ratio between 18% and 32%;
- at least 10 distinct target durations;
- at least four non-empty adaptive bins;
- largest selected bin no more than 50%;
- target duration P90 - P10 at least 8 frames;
- no non-finite or non-monotonic target tau.

## Useful environment switches

```bash
export V23_MIN_TARGET_DURATION=10
export V23_MAX_TARGET_DURATION=56
export V23_DURATION_BINS=auto:6
export V23_ACTIVITY_THRESHOLD_RATIO=0.18
export V23_BOUNDARY_YAW_RATIO=0.06
export V23_QUIET_RUN=4
export V23_PHRASE_MARGIN=3
export V23_BALANCE_POWER=0.35
export V23_MAX_BIN_FRACTION=0.45
export V23_IDENTITY_FRACTION=0.25
export V23_MIN_SPEED_FACTOR=1.15
export V23_MAX_SPEED_FACTOR=3.0
```

Do not reduce the dataset assertions merely to let training start. If the new
full-body detector still yields fewer than four duration bins, inspect the raw
event duration statistics printed before materialization.
