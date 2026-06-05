#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

export EDGE_ENABLE_V21_EVENT_ADAPTER=1
export EDGE_V21_EVENT_MANIFEST="${EDGE_V21_EVENT_MANIFEST:-data/v21_event_adapter_manifest.json}"
export EDGE_V21_EVENT_DROP_PROB="${EDGE_V21_EVENT_DROP_PROB:-0.15}"
export EDGE_V21_TRAIN_STAGE="${EDGE_V21_TRAIN_STAGE:-adapter}"
export EDGE_V21_ADAPTER_TRAIN_DECODER="${EDGE_V21_ADAPTER_TRAIN_DECODER:-0}"

python train_v21_event_adapter.py "$@"
