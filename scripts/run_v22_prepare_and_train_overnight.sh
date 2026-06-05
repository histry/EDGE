#!/usr/bin/env bash
set -Eeuo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN="${V22_OVERNIGHT_ROOT:-output/v22_turn_pace_overnight_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN"
export V22_TURN_DATASET="${V22_TURN_DATASET:-data/v22_turn_pace_dataset.npz}"
export V22_TURN_TRAIN_OUT="$RUN/train"

exec > >(tee -a "$RUN/overnight.log") 2>&1

echo "Stage 1/3: annotate shared index"
bash scripts/run_v22_annotate_index.sh

echo "Stage 2/3: build turn-pace dataset"
bash scripts/run_v22_build_turn_dataset.sh

echo "Stage 3/3: train turn-pace refiner"
bash scripts/run_v22_train_turn_pace.sh

cp "$RUN/train/BEST_TURN_PACE_CKPT.txt" "$RUN/BEST_TURN_PACE_CKPT.txt"
printf '\nDONE: %s\n' "$RUN"
