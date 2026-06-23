# EDGE V23-v2 Natural Duration Replacement Package

This package replaces the V23-v1 synthetic full-window duration setup with an
inference-safe natural-duration pipeline.

## Main corrections

1. Natural turn boundaries use adaptive peak detection and 5%-95% cumulative
   yaw-angle cropping, with a configurable 8-56 frame duration range.
2. The 17D condition is built from the observed/corrupted motion only. Target
   motion dynamics are never used as network input.
3. The final dataset is stratified by natural-duration bin and contains about
   25% exact identity samples.
4. Corruption factors default to 1.15-3.0 rather than extreme 9x compression.
5. The model predicts bounded duration, monotonic tau, and edit probability.
6. Training includes duration ranking, identity preservation, context,
   velocity, activity, yaw-peak and edit-classification objectives.
7. Best checkpoint selection uses a composite safety/generalization score.
8. A standalone runtime applies V23-v2 to V21 output with conservative fallback.

## Files

Replace or add all files in this package while preserving their relative paths.
The installer backs up existing files automatically.

```bash
cd /path/to/extracted/EDGE_V23_v2_replace
bash install_into_EDGE.sh /home/disk/lsm/storage/EDGE
```

## Full rebuild and three-seed training

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export CUDA_VISIBLE_DEVICES=0
bash scripts/launch_v23_v2_full.sh
```

The launcher rebuilds:

```text
data/v23_v2_natural_duration_dataset.npz
```

and creates:

```text
output/v23_v2_natural_duration_YYYYMMDD_HHMMSS/
```

## Environment switches

```bash
export V23_REBUILD_DATASET=1
export V23_MAX_SAMPLES=18000
export V23_MIN_SPEED_FACTOR=1.15
export V23_MAX_SPEED_FACTOR=3.0
export V23_IDENTITY_FRACTION=0.25
export V23_MIN_TARGET_DURATION=8
export V23_MAX_TARGET_DURATION=56
export V23_EPOCHS=600
export V23_BATCH_SIZE=64
export V23_PATIENCE=100
```

For a quick smoke run:

```bash
export V23_MAX_SAMPLES=6000
export V23_EPOCHS=80
export V23_PATIENCE=30
bash scripts/launch_v23_v2_full.sh
```

## Dataset acceptance criteria

The overnight script checks these automatically:

- at least 5,000 samples;
- 17D inference-safe condition;
- identity fraction between 18% and 32%;
- at least 12 distinct natural durations;
- at least four non-empty duration bins and no single bin above 55%;
- finite and monotonic target tau.

Recommended inspection:

```bash
python - <<'PY'
import numpy as np
p='data/v23_v2_natural_duration_dataset.npz'
with np.load(p, allow_pickle=True) as z:
    d=z['target_duration_frames']
    print('samples', len(d))
    print('duration', np.percentile(d,[0,10,25,50,75,90,100]))
    print('identity', (z['is_identity']>0.5).mean())
    print('bins', np.bincount(z['duration_bin'].astype(int)))
    print('factor', np.percentile(z['speed_factor'],[0,10,50,90,100]))
PY
```

## Training acceptance criteria

Use the generated `heldout_eval_best/V23_V2_HELDOUT_EVALUATION.json`.
Recommended minimums before V21 integration:

```text
duration_correlation          >= 0.85
rare_duration_mae             <= 4 frames
tau_mae                       <= 0.01
motion_mse_improvement        >= 50%
yaw_mae_improvement           >= 50%
peak_error_improvement        >= 25%
activity_preservation_ratio   >= 0.80
pose_range_preservation_ratio >= 0.90
identity_tau_mae              <= 0.01
identity_motion_drift         small and stable
edit_accuracy                 >= 0.90
```

## Apply to V21 outputs

```bash
cd /home/disk/lsm/storage/EDGE
RUN=$(ls -td output/v23_v2_natural_duration_* | head -1)
BEST=$(cat "$RUN/BEST_V23_CKPT.txt")

python tools/apply_v23_monotonic_duration.py \
  --motion_glob 'output/V21_RESULT_DIR/**/*.npy' \
  --checkpoint "$BEST" \
  --out_dir "$RUN/v21_runtime_test"
```

The runtime only accepts an edit when:

- edit probability is high;
- predicted duration expands the observed event;
- pose range and activity remain above safety thresholds;
- rotation jump does not increase excessively; and
- yaw peak improves or is already under the allowed limit.

Rejected events retain the original V21 motion.

## Render corrected motions

```bash
python render_from_npy.py \
  --motion "$RUN/v21_runtime_test/example_v23v2.npy" \
  --audio test_music_bank/dunhuangwu2.wav \
  --output "$RUN/v21_runtime_test/example_v23v2_fixed.mp4" \
  --camera_mode fixed
```

## Important compatibility note

V23-v1 checkpoints are not compatible with the V23-v2 model because V23-v2
adds bounded duration configuration and an edit-probability head. Rebuild the
new dataset and retrain from scratch.
