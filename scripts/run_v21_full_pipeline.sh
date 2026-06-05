#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

bash scripts/run_v21_build_index.sh
bash scripts/run_v21_extract_music.sh

if [[ "${V21_TRAIN_ROUTER:-0}" == "1" ]]; then
  bash scripts/run_v21_train_router.sh
fi

if [[ "${V21_TRAIN_TRANSITION:-0}" == "1" ]]; then
  bash scripts/run_v21_train_transition.sh
fi

bash scripts/run_v21_multi_music.sh
