# EDGE V23 — Monotonic Duration & Time-Warp Network

V23 replaces the failed pose-residual pace refiner with a model that directly learns:

1. how many output frames a turn should occupy; and
2. a monotonic source-time map `tau` used to resample the original motion.

The model never predicts a new pose. The output motion is created only by SO(3)-aware temporal resampling, so it cannot flatten body motion in the same way as V22.

## Dependency

V22 utilities must already exist in the EDGE project:

- `tools/v22_turn_utils.py`
- `tools/v21_common.py`

## Install

```bash
cd /home/disk/lsm/storage/EDGE
unzip -o EDGE_V23_MonotonicDuration_patch.zip
EDGE_ROOT=/home/disk/lsm/storage/EDGE \
  bash EDGE_V23_MonotonicDuration_patch/install_v23.sh
```

## Smoke test

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tools/smoke_v23_monotonic_duration.py
```

## Overnight training

```bash
cd /home/disk/lsm/storage/EDGE
RUN=output/v23_monotonic_duration_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

tmux kill-session -t v23_duration 2>/dev/null || true

tmux new-session -d -s v23_duration \
"cd /home/disk/lsm/storage/EDGE && \
 export PATH=/home/disk/lsm/conda_envs/edge/bin:\$PATH && \
 export PYTHONPATH=/home/disk/lsm/storage/EDGE:\${PYTHONPATH:-} && \
 export CUDA_VISIBLE_DEVICES=0 && \
 export V23_RUN_ROOT='$RUN' && \
 export V23_MOTION_GLOB='data/dunhuang_151d_physical/**/*' && \
 export V23_DATASET='data/v23_monotonic_duration_dataset.npz' && \
 export V23_MAX_SAMPLES=16000 && \
 export V23_AUGMENTATIONS=10 && \
 export V23_MAX_SPEED_FACTOR=8.0 && \
 export V23_EPOCHS=700 && \
 export V23_BATCH_SIZE=64 && \
 export V23_REBUILD_DATASET=1 && \
 bash scripts/run_v23_duration_overnight.sh"

sleep 3
echo "RUN=$RUN"
tmux ls
tail -80 "$RUN/overnight.log"
```

The script trains three seeds and writes `BEST_V23_CKPT.txt`.
