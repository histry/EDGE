#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
: "${V23_STAGE1_CHECKPOINT_OVERRIDE:?Set V23_STAGE1_CHECKPOINT_OVERRIDE to a successful Stage-1 best.pt}"
export V23_REBUILD_DATASET=0
export V23_DATASET="${V23_DATASET:-data/v23_v2_4_slowaware_w120_d88_9k.npz}"
export V23_SEEDS="${V23_SEEDS:-20260610}"
export V23_GATE_MODE="${V23_GATE_MODE:-duration_core}"
exec bash scripts/launch_v23_v2_5_full.sh
