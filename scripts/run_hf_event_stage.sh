#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh"
fi

conda activate edge || conda activate /home/disk/lsm/conda_envs/edge || true

RUN_ID="${RUN_ID:-hf_event_day01}"
OUT_DIR="${OUT_DIR:-checkpoints/hf_event_contrastive/${RUN_ID}}"
DATA_DIR="${DATA_DIR:-data/dunhuang_bvh/footwork_v3h_u45}"
MUSIC_GLOB="${MUSIC_GLOB:-test_music_bank/*.wav}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p logs "$OUT_DIR"

LOG="logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "RUN_ID=$RUN_ID"
echo "DATA_DIR=$DATA_DIR"
echo "MUSIC_GLOB=$MUSIC_GLOB"
echo "OUT_DIR=$OUT_DIR"
echo "============================================================"

python scripts/train_hf_event_contrastive.py \
  --data_dir "$DATA_DIR" \
  --music_glob "$MUSIC_GLOB" \
  --out_dir "$OUT_DIR" \
  --seq_len 45 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr 1e-3 \
  --hidden_dim 128 \
  --emb_dim 64 \
  --temperature 0.1 \
  --num_workers 2

IN_DB="${IN_DB:-data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz}"
OUT_DB="${OUT_DB:-data/dunhuang_choreo_unit_rag/index_v14_hf_event_u45_s15.npz}"

python scripts/export_hf_event_rag_db.py \
  --in_db "$IN_DB" \
  --out_db "$OUT_DB" \
  --encoder_ckpt "$OUT_DIR/hf_event_encoder.pt" \
  --seq_len 45 \
  --batch_size 512 \
  --copy_arrays 1

echo "============================================================"
echo "HF event stage done."
echo "encoder=$OUT_DIR/hf_event_encoder.pt"
echo "rag_db=$OUT_DB"
echo "log=$LOG"
echo "============================================================"
