# V23-v2.4 Ordinal Event-Consistent Natural Duration

## Purpose

This package replaces the failed V23-v2.3 Stage-1 protocol while retaining the validated slow-aware W120/D88 event segmentation and Stage-2 monotonic SO(3) time warp.

The failure addressed here is not insufficient training time. In V23-v2.3, the same natural event produced many identity/compressed augmentations, but every augmentation was treated as an independent sample. The shared Stage-1 encoder was simultaneously asked to:

- remain invariant to compression when predicting natural duration; and
- remain sensitive to compression when predicting whether editing is needed.

This produced rapid memorisation and best validation epochs around 3-4.

V23-v2.4 changes the scientific formulation:

1. **One training item = one natural event with two views.**
2. **Duration prediction is view-invariant.**
3. **Edit prediction uses a separate pace-sensitive head.**
4. **Duration bins are ordinal, not unrelated categorical classes.**
5. **Long-duration underestimation is penalised explicitly.**
6. **Validation and checkpoint selection are event-level.**

## Files replaced

- `model/v23_monotonic_duration.py`
- `train_v23_monotonic_duration.py`
- `tools/v23_duration_utils.py`
- `tools/build_v23_monotonic_duration_dataset.py`
- `tools/evaluate_v23_checkpoint.py`
- `tools/apply_v23_monotonic_duration.py`
- `tools/smoke_v23_monotonic_duration.py`
- `scripts/run_v23_build_dataset.sh`
- `scripts/run_v23_train_one_seed.sh`
- `scripts/run_v23_duration_overnight.sh`
- `scripts/launch_v23_v2_4_full.sh`
- compatibility wrappers `launch_v23_v2_3_full.sh`, `launch_v23_v2_full.sh`

## Important incompatibility

The database must be rebuilt because V23-v2.4 requires:

- `event_uid`
- `augmentation_id`

Old V23-v2.2/v2.3 NPZ files are rejected intentionally.

## Install

```bash
cd /path/to/EDGE_V23_v2_4_ordinal_event
bash install_into_EDGE.sh /home/disk/lsm/storage/EDGE
```

## Static checks

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python -m py_compile \
  model/v23_monotonic_duration.py \
  train_v23_monotonic_duration.py \
  tools/v23_duration_utils.py \
  tools/build_v23_monotonic_duration_dataset.py \
  tools/evaluate_v23_checkpoint.py \
  tools/apply_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py

bash -n \
  scripts/run_v23_build_dataset.sh \
  scripts/run_v23_train_one_seed.sh \
  scripts/run_v23_duration_overnight.sh \
  scripts/launch_v23_v2_4_full.sh

python tools/smoke_v23_monotonic_duration.py \
  --window_len 120 \
  --duration_edges 12,24,37,50,63,76,89
```

## Recommended first run: one seed

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export CUDA_VISIBLE_DEVICES=0

export V23_REBUILD_DATASET=1
export V23_DATASET=data/v23_v2_4_slowaware_w120_d88_9k.npz
export V23_MAX_SAMPLES=9000
export V23_WINDOW_LEN=120
export V23_MIN_TARGET_DURATION=12
export V23_MAX_TARGET_DURATION=88
export V23_DURATION_BINS='auto:6'
export V23_IDENTITY_FRACTION=0.25
export V23_MIN_SPEED_FACTOR=1.15
export V23_MAX_SPEED_FACTOR=3.0

export V23_SEEDS='20260610'
export V23_SPLIT_SEED=20260620
export V23_SPLIT_TRIALS=4096

export V23_HIDDEN_DIM=96
export V23_DROPOUT=0.24
export V23_WEIGHT_DECAY=1e-3
export V23_STAGE1_EVENT_BATCH=40
export V23_STAGE2_BATCH_SIZE=40

export V23_STAGE1_EPOCHS=240
export V23_STAGE1_PATIENCE=60
export V23_STAGE1_LR=3e-5
export V23_STAGE1_WARMUP=10

export V23_STAGE2_EPOCHS=240
export V23_STAGE2_PATIENCE=60
export V23_STAGE2_LR=8e-5
export V23_STAGE2_WARMUP=8
export V23_TF_DECAY_EPOCHS=90

# Stage 2 only starts if Stage 1 passes these gates.
export V23_STAGE1_REQUIRE=1
export V23_GATE_EVENT_MAE=8.0
export V23_GATE_LONG_MAE=10.0
export V23_GATE_EVENT_CORR=0.82
export V23_GATE_WITHIN1=0.78
export V23_GATE_EDIT=0.80

bash scripts/launch_v23_v2_4_full.sh
```

## Full overnight after the one-seed gate passes

```bash
unset V23_SEEDS
export V23_SEEDS='20260610 20260611 20260612'
export V23_REBUILD_DATASET=0
export V23_DATASET=data/v23_v2_4_slowaware_w120_d88_9k.npz

bash scripts/launch_v23_v2_4_full.sh
```

## Stage-1 outputs

Each seed writes:

```text
seed_<seed>/stage1_duration/
  checkpoints/best.pt
  history.json
  split.json
seed_<seed>/STAGE1_GATE.json
```

The root run directory writes:

```text
STAGE1_RANKING.tsv
BEST_DURATION_CKPT.txt
```

## Stage-1 acceptance targets

Primary event-level metrics:

- `event_duration_mae_frames <= 8`
- `event_duration_long_mae <= 10`
- `event_duration_correlation >= 0.82`
- `event_duration_within_one_bin_accuracy >= 0.78`
- `edit_accuracy >= 0.80`

Stronger paper-ready targets:

- Event MAE `<= 6`
- Long-event MAE `<= 8`
- Correlation `>= 0.88`
- Within-one-bin accuracy `>= 0.90`
- Edit accuracy `>= 0.88`

## Key environment switches

### Model and optimisation

```bash
export V23_HIDDEN_DIM=96
export V23_DROPOUT=0.24
export V23_WEIGHT_DECAY=1e-3
export V23_STAGE1_LR=3e-5
export V23_STAGE1_WARMUP=10
export V23_STAGE1_EMA_DECAY=0.995
export V23_CONDITION_NOISE_STD=0.01
```

### Ordinal/event-consistency losses

```bash
export V23_LAMBDA_ORDINAL=1.0
export V23_LAMBDA_RESIDUAL=0.8
export V23_LAMBDA_RELATIVE=1.0
export V23_LAMBDA_LOG_DURATION=0.6
export V23_LAMBDA_DIRECT=0.35
export V23_LAMBDA_UNDERESTIMATE=0.8
export V23_LAMBDA_DURATION_RANK=0.18
export V23_LAMBDA_PAIR_DURATION=0.9
export V23_LAMBDA_PAIR_DISTRIBUTION=0.35
export V23_LAMBDA_MOMENT=0.30
export V23_LAMBDA_EDIT=0.25
export V23_LONG_DURATION_WEIGHT=1.5
```

If long events remain underestimated:

```bash
export V23_LAMBDA_UNDERESTIMATE=1.1
export V23_LONG_DURATION_WEIGHT=1.9
export V23_LAMBDA_MOMENT=0.40
```

Do not raise all three aggressively in the first run.

## Scientific interpretation

V23-v2.4 implements:

- slow-aware phase segmentation;
- event-grouped multi-view supervision;
- pose-invariant pace representation;
- ordinal duration estimation;
- intra-bin residual calibration;
- separate edit sensitivity;
- two-stage monotonic temporal reparameterisation.

The intended paper claim is not merely “a better duration regressor”. It is:

> Natural event duration is invariant to synthetic pace corruption, whereas edit necessity is pace-sensitive. Separating these objectives and training on paired event views improves low-resource duration generalisation.
