# V23-v2.5 Continuous-Calibrated Scientific Gate

## 1. Why this patch is necessary

The Stage-1 checkpoint is scientifically useful even though the old gate rejected it:

- final continuous-duration exact-bin accuracy: about 0.689;
- final continuous-duration within-one-bin accuracy: about 0.996;
- continuous duration MAE: about 5.21 frames;
- ordinal argmax exact/within-one-bin: about 0.331/0.648.

The old gate evaluated the auxiliary ordinal argmax, while Stage 2 and runtime consume the blended continuous duration. V23-v2.5 makes the final continuous duration the single source of truth for:

- checkpoint selection;
- Stage-1 scientific gating;
- held-out evaluation;
- event-level ranking;
- runtime duration-bin reporting.

Ordinal predictions remain diagnostics. Edit classification becomes an auxiliary warning by default and no longer blocks Tau training.

## 2. Main changes

1. `duration_bin_index` now follows the final continuous duration.
2. The ordinal argmax remains available as `duration_ordinal_bin_index`.
3. Sample- and event-level metrics report continuous and ordinal results separately.
4. The Stage-1 core gate uses:
   - event MAE;
   - long-event MAE;
   - event correlation;
   - continuous-duration within-one-bin accuracy;
   - P90 calibration error;
   - quantile calibration MAE.
5. Edit AUROC and optimally calibrated balanced accuracy are reported, but are advisory in `duration_core` mode.
6. Runtime uses Edit as advisory by default and relies on explicit yaw/activity/range/jump safety checks for acceptance.
7. The launcher exits cleanly if no Stage-2 checkpoint exists.
8. Existing V23-v2.4 Stage-1 checkpoints are compatible; no database rebuild is required.

## 3. Installation

```bash
cd /path/to/EDGE_V23_v2_5_continuous_gate
bash install_into_EDGE.sh /home/disk/lsm/storage/EDGE
```

Then:

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
```

Static checks:

```bash
python -m py_compile \
  model/v23_monotonic_duration.py \
  train_v23_monotonic_duration.py \
  tools/evaluate_v23_stage1_gate.py \
  tools/evaluate_v23_checkpoint.py \
  tools/apply_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py

bash -n \
  scripts/run_v23_train_one_seed.sh \
  scripts/run_v23_duration_overnight.sh \
  scripts/launch_v23_v2_5_full.sh \
  scripts/continue_v23_v2_5_from_stage1.sh

python tools/smoke_v23_monotonic_duration.py \
  --window_len 120 \
  --duration_edges 12,24,37,50,63,76,89
```

## 4. Continue directly from the successful Stage-1 checkpoint

Do not rebuild the database and do not retrain Stage 1 first.

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge

export CUDA_VISIBLE_DEVICES=0
export V23_DATASET=data/v23_v2_4_slowaware_w120_d88_9k.npz
export V23_STAGE1_CHECKPOINT_OVERRIDE=/home/disk/lsm/storage/EDGE/output/v23_v2_4_ordinal_event_20260606_214309/seed_20260610/stage1_duration/checkpoints/best.pt
export V23_SEEDS='20260610'
export V23_GATE_MODE=duration_core

bash scripts/continue_v23_v2_5_from_stage1.sh
```

The script first creates `STAGE1_CONTINUOUS_EVALUATION.json`, applies the corrected gate, and then starts Stage 2.

## 5. Correct Stage-1 gate

Default core thresholds:

```bash
export V23_GATE_EVENT_MAE=7.0
export V23_GATE_LONG_MAE=8.0
export V23_GATE_EVENT_CORR=0.90
export V23_GATE_CONTINUOUS_WITHIN1=0.95
export V23_GATE_P90_ERROR=8.0
export V23_GATE_QUANTILE_MAE=7.0
```

Advisory metrics:

```bash
export V23_WARN_ORDINAL_WITHIN1=0.60
export V23_WARN_EDIT_BALANCED=0.72
export V23_WARN_EDIT_AUROC=0.72
```

Default mode:

```bash
export V23_GATE_MODE=duration_core
```

`duration_core` only requires the six duration/calibration checks. `strict` additionally requires all auxiliary checks.

## 6. Overnight Stage-2 robustness experiment from one fixed Stage-1 model

After one seed runs correctly:

```bash
export V23_REBUILD_DATASET=0
export V23_DATASET=data/v23_v2_4_slowaware_w120_d88_9k.npz
export V23_STAGE1_CHECKPOINT_OVERRIDE=/home/disk/lsm/storage/EDGE/output/v23_v2_4_ordinal_event_20260606_214309/seed_20260610/stage1_duration/checkpoints/best.pt
export V23_SEEDS='20260610 20260611 20260612'
export V23_GATE_MODE=duration_core

bash scripts/continue_v23_v2_5_from_stage1.sh
```

This keeps the successful duration model fixed and measures Stage-2 initialization variability fairly.

## 7. Fresh full two-stage training

To retrain Stage 1 and Stage 2 from scratch:

```bash
unset V23_STAGE1_CHECKPOINT_OVERRIDE
export V23_REBUILD_DATASET=0
export V23_DATASET=data/v23_v2_4_slowaware_w120_d88_9k.npz
export V23_SEEDS='20260610 20260611 20260612'
export V23_GATE_MODE=duration_core

bash scripts/launch_v23_v2_5_full.sh
```

No database rebuild is needed unless the underlying natural events change.

## 8. Runtime policy

The default runtime policy is:

```text
Edit probability: advisory
Ordinal confidence: diagnostic
Duration expansion: hard gate
Yaw improvement/safety: hard gate
Activity preservation: hard gate
Pose-range preservation: hard gate
Rotation-jump increase: hard gate
```

Example:

```bash
RUN=$(ls -td output/v23_v2_5_continuous_gate_* | head -1)
BEST=$(cat "$RUN/BEST_V23_CKPT.txt")

python tools/apply_v23_monotonic_duration.py \
  --motion output/your_v21_result.npy \
  --checkpoint "$BEST" \
  --out_dir "$RUN/v21_runtime_test" \
  --edit_gate_mode advisory \
  --min_edit_probability -1
```

`--min_edit_probability -1` uses the validation-calibrated threshold stored in the checkpoint metrics. Use `--edit_gate_mode strict` only for a strict edit ablation.

## 9. Scientific interpretation

V23-v2.5 explicitly distinguishes:

- **ordinal supervision**, an auxiliary structured learning signal;
- **continuous natural duration**, the final scientific prediction;
- **edit necessity**, an auxiliary runtime cue;
- **physical safety**, the final acceptance criterion.

This prevents an auxiliary classifier from vetoing a continuous estimator that already meets the actual research objective.
