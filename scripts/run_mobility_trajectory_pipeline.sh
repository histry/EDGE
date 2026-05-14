#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
mkdir -p logs output/mobility_trajectory output/mobility_trajectory/videos

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"

CKPT="${CKPT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
RAW_DB="${RAW_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"
MOB_DB="${MOB_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz}"

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_end.npy}"
MUSIC="${MUSIC:-test_music_bank/dunhuangwu2.wav}"
TRAJECTORY="${TRAJECTORY:-0,0;0.5,0.7;-0.3,1.2;0,1.6}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "$MOB_DB" ]; then
  "$PY" mobility_unit_labels.py \
    --input "$RAW_DB" \
    --output "$MOB_DB" \
    --report "${MOB_DB%.npz}.report.json"
fi

CASE="mobile_units_s_curve"
OUT="output/mobility_trajectory/${CASE}.npy"

"$PY" generate_mobility_aware.py \
  --mode trajectory \
  --checkpoint "$CKPT" \
  --music "$MUSIC" \
  --start_pose "$START_POSE" \
  --end_pose "$END_POSE" \
  --out "$OUT" \
  --mobility_db "$MOB_DB" \
  --trajectory "$TRAJECTORY" \
  --mid_count 2 \
  --mid_keyframe_strength 0.10 \
  --endpoint_keyframe_strength 1.0 \
  --energy_scale 0.5 \
  --context_scale 0.5 \
  --unit_prior_strength 0.012 \
  --unit_prior_features upper+torso \
  --render \
  --log "logs/${CASE}.log"

cp -f "output/mobility_trajectory/${CASE}_follow.mp4" "output/mobility_trajectory/videos/${CASE}_follow.mp4" 2>/dev/null || true
cp -f "output/mobility_trajectory/${CASE}_fixed.mp4" "output/mobility_trajectory/videos/${CASE}_fixed.mp4" 2>/dev/null || true

echo "✅ Mobility-aware trajectory run done."
