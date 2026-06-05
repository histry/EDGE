#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PATH="/home/disk/lsm/conda_envs/edge/bin:$PATH"
export PYTHONPATH="/home/disk/lsm/storage/EDGE:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

export V21_ROUTER_MUSIC_DIR=data/v21_router_music_999/splits/train
export V21_ROUTER_FEATURE_DIR=data/v21_music_features_router_985_train
export V21_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index

export V21_ROUTER_RUN_ROOT="${V21_ROUTER_RUN_ROOT:?V21_ROUTER_RUN_ROOT is required}"

export V21_ROUTER_EPOCHS=180
export V21_ROUTER_BATCH_SIZE=512
export V21_ROUTER_LR=1e-4
export V21_ROUTER_WEIGHT_DECAY=1e-4

export V21_ROUTER_HIDDEN=192
export V21_ROUTER_LATENT=96
export V21_ROUTER_DROPOUT=0.15
export V21_ROUTER_MARGIN_WEIGHT=0.80

export V21_ROUTER_POSITIVES=6
export V21_ROUTER_NEGATIVES=4
export V21_ROUTER_PHRASES=3
export V21_ROUTER_NUM_FRAMES=150

export V21_FORCE_REEXTRACT=0

bash scripts/run_v21_router_calibrated_overnight.sh
