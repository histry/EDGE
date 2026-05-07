#!/usr/bin/env bash
# Optional smoke-test launcher. This does NOT replace the full training launcher.

set -euo pipefail
cd /home/disk/lsm/storage/EDGE

EPOCHS="${EPOCHS:-1}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-20}" \
EXP_NAME="${EXP_NAME:-v10_text_context_rag_adapter_smoke_fix2}" \
bash scripts/run_train_text_context_rag_adapter_fix2.sh
