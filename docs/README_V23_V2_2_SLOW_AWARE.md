# V23-v2.2 Slow-Aware Phase Splitter

This patch changes only event detection / dataset construction.  The V23-v2.1
model and training code remain compatible.

## Why

Dunhuang dance often contains slow preparation, sustained extension and gradual
recovery.  Instantaneous velocity thresholds either fragment these phases or
merge several nearby phases until the hard maximum duration.  V23-v2.2 adds:

- multi-scale pose progression for slow movement;
- accumulated-yaw proposals for low-speed turns;
- recursive splitting at sustained valleys, direction reversals and dual peaks;
- rejection of unsplittable over-cap events instead of clipping them to the cap.

## Install

```bash
bash install_patch.sh /home/disk/lsm/storage/EDGE
```

## Recommended diagnostic dataset

```bash
cd /home/disk/lsm/storage/EDGE
export V23_DATASET=data/v23_v2_2_slowaware_w120_d88_6k.npz
export V23_MAX_SAMPLES=6000
export V23_WINDOW_LEN=120
export V23_MIN_TARGET_DURATION=12
export V23_MAX_TARGET_DURATION=88
export V23_DURATION_BINS='auto:6'
export V23_IDENTITY_FRACTION=0.25
export V23_MIN_SPEED_FACTOR=1.15
export V23_MAX_SPEED_FACTOR=3.0
export V23_MIN_PEAK_DPS=14
export V23_MIN_TURN_ANGLE=10
export V23_SLOW_POSE_SPAN=10
export V23_SLOW_ANGLE_WINDOW=24
export V23_QUIET_RUN=8
export V23_SPLIT_SCORE_THRESHOLD=0.68
export V23_LONG_SPLIT_SCORE_THRESHOLD=0.42
bash scripts/run_v23_build_dataset.sh
```

## Accept the dataset only if

- raw P90 is strictly below 88;
- the raw last-bin fraction is below 35%;
- no single exact duration occupies more than 20%;
- at least four adaptive bins are non-empty;
- all 72 sources remain represented after materialization.
