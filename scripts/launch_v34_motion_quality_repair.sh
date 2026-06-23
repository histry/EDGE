#!/usr/bin/env bash
set -euo pipefail

# Source-aware + rhythm-density + contact-lock V34 launcher.
# Default mode is pure inference.  Set V34_TRAIN=1 for full retraining.

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

export V34_TRAIN="${V34_TRAIN:-0}"
export V34_SOURCE_AWARE_REBUILD="${V34_SOURCE_AWARE_REBUILD:-0}"
export V34_SOURCE_AWARE_REBUILD_HIERARCHY="${V34_SOURCE_AWARE_REBUILD_HIERARCHY:-0}"

export SOURCE_AWARE_JSON="${SOURCE_AWARE_JSON:-data/v34_source_aware/v34_shared_event_index_source_aware.json}"
export SOURCE_AWARE_NPZ="${SOURCE_AWARE_NPZ:-data/v34_source_aware/v34_shared_event_index_source_aware.npz}"
export SOURCE_AWARE_HIER_NPZ="${SOURCE_AWARE_HIER_NPZ:-data/v34_source_aware/v34_hierarchical_event_index_source_aware.npz}"
export SOURCE_AWARE_HIER_JSON="${SOURCE_AWARE_HIER_JSON:-data/v34_source_aware/v34_hierarchical_event_index_source_aware.json}"
export V33_EVENT_CONTACT_CACHE="${V33_EVENT_CONTACT_CACHE:-data/v34_source_aware/v33_event_contact_cache_source_aware.npz}"
export V34_DATASET="${V34_DATASET:-data/v34_source_aware/v33_transition_dataset_source_aware.npz}"

export RUN_ID="${RUN_ID:-v34_motion_quality_repair_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_MOTION_QUALITY_REPAIR.txt

echo "[V34 MOTION QUALITY REPAIR] RUN_ROOT=$RUN_ROOT"
echo "[V34 MOTION QUALITY REPAIR] V34_TRAIN=$V34_TRAIN"
echo "[V34 MOTION QUALITY REPAIR] SOURCE_AWARE_JSON=$SOURCE_AWARE_JSON"

bash scripts/launch_v34_source_aware_rag.sh "$@"
