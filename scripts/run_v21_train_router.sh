#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

INDEX_PREFIX="${V21_INDEX_PREFIX:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index}"
MUSIC_GLOB="${V21_MUSIC_FEATURE_GLOB:-data/v21_music_features/*_v21_music.npy}"
RUN_ROOT="${V21_ROUTER_RUN_ROOT:-output/v21_music_router_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

python tools/build_v21_router_dataset.py \
  --index_json "${INDEX_PREFIX}.json" \
  --index_npz "${INDEX_PREFIX}.npz" \
  --music_glob "$MUSIC_GLOB" \
  --out "$RUN_ROOT/router_dataset.npz" \
  --phrases "${V21_PHRASE_COUNT:-3}" \
  --positives_per_phrase "${V21_ROUTER_POSITIVES:-4}" \
  --negatives_per_positive "${V21_ROUTER_NEGATIVES:-2}"

python train_v21_music_router.py \
  --data "$RUN_ROOT/router_dataset.npz" \
  --out_dir "$RUN_ROOT/train" \
  --epochs "${V21_ROUTER_EPOCHS:-250}" \
  --batch_size "${V21_ROUTER_BATCH_SIZE:-256}" \
  --lr "${V21_ROUTER_LR:-2e-4}" \
  --hidden_dim "${V21_ROUTER_HIDDEN:-128}" \
  --latent_dim "${V21_ROUTER_LATENT:-64}"

printf '\nROUTER_CKPT=%s\n' "$RUN_ROOT/train/checkpoints/best.pt"
