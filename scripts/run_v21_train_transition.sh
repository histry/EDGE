#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

EVENT_DB="${V21_TRANSITION_EVENT_DB:-data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_balanced_strict.json}"
RUN_ROOT="${V21_TRANSITION_RUN_ROOT:-output/v21_transition_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

python tools/build_v21_transition_dataset.py \
  --event_db "$EVENT_DB" \
  --out "$RUN_ROOT/transition_dataset.npz" \
  --max_pairs "${V21_TRANSITION_MAX_PAIRS:-30000}" \
  --max_gap "${V21_TRANSITION_MAX_GAP:-8}"

python train_v21_transition.py \
  --data "$RUN_ROOT/transition_dataset.npz" \
  --out_dir "$RUN_ROOT/train" \
  --epochs "${V21_TRANSITION_EPOCHS:-600}" \
  --batch_size "${V21_TRANSITION_BATCH_SIZE:-64}" \
  --lr "${V21_TRANSITION_LR:-2e-4}" \
  --dpn_hidden_dim "${V21_DPN_HIDDEN:-192}" \
  --refiner_hidden_dim "${V21_REFINER_HIDDEN:-256}" \
  --residual_scale "${V21_RESIDUAL_SCALE:-0.18}"

printf '\nTRANSITION_CKPT=%s\n' "$RUN_ROOT/train/checkpoints/best.pt"
