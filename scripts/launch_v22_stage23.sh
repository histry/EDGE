#!/usr/bin/env bash
set -Eeuo pipefail

RUN="${1:?Usage: launch_v22_stage23.sh RUN_DIR}"

cd /home/disk/lsm/storage/EDGE

export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="/home/disk/lsm/storage/EDGE:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

export V22_OVERNIGHT_ROOT="$RUN"

export V22_TURN_DATASET=data/v22_turn_pace_dataset.npz
export V22_TURN_TRAIN_OUT="$RUN/train"

export V22_MOTION_GLOB='data/dunhuang_151d_physical/*.npy'
export V22_DATA_MAX_SAMPLES=12000

export V22_TURN_EPOCHS=320
export V22_TURN_BATCH_SIZE=96
export V22_TURN_WORKERS=4
export V22_TURN_AMP=1

bash scripts/run_v22_stage23_resume.sh
