#!/usr/bin/env bash
set -euo pipefail

cd "${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

EVENT_DB="${V21_STYLE_EVENT_DB:-data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_balanced_strict.json}"
POS_GLOB_1="${V21_STYLE_POS_GLOB_1:-output/night_v17b_conservative_20260604_170625/refined/*s0_32_64_96_ew0.45*emotion_refined.npy}"
POS_GLOB_2="${V21_STYLE_POS_GLOB_2:-output/night_v17b_conservative_20260604_170625/refined/*s0_35_70_105_ew0.45*emotion_refined.npy}"
RUN_ROOT="${V21_STYLE_RUN_ROOT:-output/v21_style_ranker_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

python tools/build_v21_style_ranker_dataset.py \
  --positive_glob "$POS_GLOB_1" \
  --positive_glob "$POS_GLOB_2" \
  --event_db "$EVENT_DB" \
  --out "$RUN_ROOT/style_pairs.npz" \
  --num_pairs "${V21_STYLE_NUM_PAIRS:-20000}" \
  --low_style_percentile "${V21_STYLE_NEG_PERCENTILE:-35}"

python train_v21_style_ranker.py \
  --data "$RUN_ROOT/style_pairs.npz" \
  --out_dir "$RUN_ROOT/train" \
  --epochs "${V21_STYLE_EPOCHS:-300}" \
  --batch_size "${V21_STYLE_BATCH_SIZE:-256}" \
  --lr "${V21_STYLE_LR:-2e-4}" \
  --hidden_dim "${V21_STYLE_HIDDEN:-128}"

printf '\nSTYLE_RANKER_CKPT=%s\n' "$RUN_ROOT/train/checkpoints/best.pt"
