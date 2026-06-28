#!/usr/bin/env bash
set -euo pipefail
cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
source scripts/v40_floor_aware_env.sh
export EDGE_ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$EDGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
mkdir -p output
export V26_INDEX_JSON="${V26_INDEX_JSON:-data/v34_shared_event_index.json}"
export V26_DURATION_INDEX_NPZ="${V26_DURATION_INDEX_NPZ:-data/v26_music_dominant_duration_index.npz}"
export V33_EVENT_CONTACT_CACHE="${V33_EVENT_CONTACT_CACHE:-data/v34_source_aware/v33_event_contact_cache_source_aware.npz}"
export V26_ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
export V26_V23_CKPT="${V26_V23_CKPT:-output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt}"
export V26_PLANNER_CKPT="${V26_PLANNER_CKPT:-output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt}"
export V34_TRAIN="${V34_TRAIN:-1}"
export V34_AE_EPOCHS="${V34_AE_EPOCHS:-180}"
export V34_CONTACT_EPOCHS="${V34_CONTACT_EPOCHS:-70}"
export V34_DIFFUSION_EPOCHS="${V34_DIFFUSION_EPOCHS:-280}"
export RUN_ID="${RUN_ID:-v40_floor_aware_full_train_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
echo "$RUN_ROOT" > output/LATEST_V40_FLOOR_AWARE_FULL_TRAIN.txt
echo "[RUN_ROOT] $RUN_ROOT"
bash scripts/launch_v40_source_aware_rag.sh "$@"
