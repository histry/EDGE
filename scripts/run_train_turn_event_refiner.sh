#!/usr/bin/env bash
# Backward-compatible entrypoint: now runs v2 multi-target event-weighted refiner.
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
exec bash scripts/run_train_turn_event_refiner_v2.sh "$@"
