#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

mkdir -p logs
mkdir -p output/mobility_stationary
mkdir -p output/mobility_stationary/videos

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"

CKPT="${CKPT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
RAW_DB="${RAW_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"
MOB_DB="${MOB_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz}"

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_start.npy}"
MUSIC_NAMES="${MUSIC_NAMES:-dunhuangwu2 dunhuangwu3 dunhuangwu4}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 1. build mobility labels
if [ ! -f "$MOB_DB" ]; then
  echo "==== Building mobility labels ===="
  "$PY" mobility_unit_labels.py \
    --input "$RAW_DB" \
    --output "$MOB_DB" \
    --report "${MOB_DB%.npz}.report.json" \
    2>&1 | tee logs/build_mobility_labels.log
else
  echo "Mobility DB exists: $MOB_DB"
fi

# 2. stationary root-locked generation
for M in $MUSIC_NAMES; do
  AUDIO="test_music_bank/${M}.wav"
  if [ ! -f "$AUDIO" ]; then
    echo "skip missing music: $AUDIO"
    continue
  fi

  for MID_COUNT in 0 1 2; do
    for PRIOR in 0.000 0.012 0.020; do
      if [ "$MID_COUNT" = "0" ] && [ "$PRIOR" != "0.000" ]; then
        continue
      fi

      CASE="${M}_stationary_mid${MID_COUNT}_prior${PRIOR}"
      OUT="output/mobility_stationary/${CASE}.npy"

      echo ""
      echo "==== Generate $CASE ===="

      "$PY" generate_mobility_aware.py \
        --mode stationary \
        --checkpoint "$CKPT" \
        --music "$AUDIO" \
        --start_pose "$START_POSE" \
        --end_pose "$END_POSE" \
        --out "$OUT" \
        --mobility_db "$MOB_DB" \
        --mid_count "$MID_COUNT" \
        --mid_keyframe_strength 0.08 \
        --endpoint_keyframe_strength 0.25 \
        --energy_scale 0.5 \
        --context_scale 0.5 \
        --unit_prior_strength "$PRIOR" \
        --unit_prior_features upper \
        --root_lock_after \
        --render \
        --log "logs/${CASE}.log" || true

      # move rendered videos to videos folder as convenience symlinks/copies
      ROOTLOCK="${OUT%.npy}_rootlock"
      [ -f "${ROOTLOCK}_follow.mp4" ] && cp -f "${ROOTLOCK}_follow.mp4" "output/mobility_stationary/videos/${CASE}_follow.mp4"
      [ -f "${ROOTLOCK}_fixed.mp4" ] && cp -f "${ROOTLOCK}_fixed.mp4" "output/mobility_stationary/videos/${CASE}_fixed.mp4"
    done
  done

  # Beat only after content exists; still root-locked.
  for BEAT in 0.01 0.03; do
    CASE="${M}_stationary_mid1_prior012_beat${BEAT}"
    OUT="output/mobility_stationary/${CASE}.npy"

    "$PY" generate_mobility_aware.py \
      --mode stationary \
      --checkpoint "$CKPT" \
      --music "$AUDIO" \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --out "$OUT" \
      --mobility_db "$MOB_DB" \
      --mid_count 1 \
      --mid_keyframe_strength 0.08 \
      --endpoint_keyframe_strength 0.25 \
      --beat_weight "$BEAT" \
      --energy_scale 0.5 \
      --context_scale 0.5 \
      --unit_prior_strength 0.012 \
      --unit_prior_features upper \
      --root_lock_after \
      --render \
      --log "logs/${CASE}.log" || true

    ROOTLOCK="${OUT%.npy}_rootlock"
    [ -f "${ROOTLOCK}_follow.mp4" ] && cp -f "${ROOTLOCK}_follow.mp4" "output/mobility_stationary/videos/${CASE}_follow.mp4"
    [ -f "${ROOTLOCK}_fixed.mp4" ] && cp -f "${ROOTLOCK}_fixed.mp4" "output/mobility_stationary/videos/${CASE}_fixed.mp4"
  done
done

# 3. body-centered metrics
"$PY" mobility_motion_utils.py batch-eval \
  --glob "output/mobility_stationary/*_rootlock.npy" \
  --output_csv output/mobility_stationary/body_centered_metrics.csv

echo ""
echo "✅ Done."
echo "Metrics: output/mobility_stationary/body_centered_metrics.csv"
echo "Videos : output/mobility_stationary/videos"
