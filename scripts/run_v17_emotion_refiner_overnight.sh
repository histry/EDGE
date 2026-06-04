#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

if [ -f /home/disk/lsm/conda_envs/edge/bin/activate ]; then
  source /home/disk/lsm/conda_envs/edge/bin/activate
else
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate edge
fi

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATE=$(date +%Y%m%d_%H%M%S)
SRC_RUN_ROOT=${SRC_RUN_ROOT:-output/night_v16_visual_first_20260602_204417}
REPORT=${REPORT:-"$SRC_RUN_ROOT/visual_first_pool_report.json"}
AUDIO_LIST=${AUDIO_LIST:-"test_music_bank/dunhuangwu2.wav,test_music_bank/dunhuangwu3.wav,test_music_bank/dunhuangwu4.wav"}

RUN_ROOT=${RUN_ROOT:-output/night_v17_emotion_refiner_${DATE}}
LOG_ROOT=${LOG_ROOT:-logs/night_v17_emotion_refiner_${DATE}}
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V17 Emotion-Aware Dunhuang Refiner Overnight"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "SRC_RUN_ROOT=$SRC_RUN_ROOT"
echo "REPORT=$REPORT"
echo "AUDIO_LIST=$AUDIO_LIST"
echo "============================================================"

for f in \
  tools/extract_music_emotion_features.py \
  tools/build_emotion_refiner_dataset.py \
  tools/schedule_emotion_aware_phrase.py \
  model/emotion_conditioned_refiner.py \
  train_emotion_refiner.py \
  infer_emotion_refiner.py
do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    exit 2
  fi
done

if [ ! -f "$REPORT" ]; then
  echo "ERROR: missing report: $REPORT"
  exit 3
fi

echo "[1/6] Extract music emotion features"
MUSIC_NPYS=()
IFS=',' read -ra AUDS <<< "$AUDIO_LIST"
for a in "${AUDS[@]}"; do
  if [ ! -f "$a" ]; then
    echo "WARN: missing audio $a, skip"
    continue
  fi
  stem=$(basename "$a" .wav)
  out="$RUN_ROOT/music_${stem}.npy"
  python tools/extract_music_emotion_features.py \
    --audio "$a" \
    --out_npy "$out" \
    --out_json "$RUN_ROOT/music_${stem}.json" \
    --num_frames 150
  MUSIC_NPYS+=("$out")
done

if [ "${#MUSIC_NPYS[@]}" -eq 0 ]; then
  echo "ERROR: no valid audio features"
  exit 4
fi

MUSIC_NPY_LIST=$(IFS=, ; echo "${MUSIC_NPYS[*]}")

echo "[2/6] Generate emotion-aware scheduled phrases"
SCHED_ROOT="$RUN_ROOT/emotion_aware_schedules"
mkdir -p "$SCHED_ROOT"

START_LIST=("0,32,64,96" "0,35,70,105" "0,30,60,90")
EW_LIST=("0.20" "0.45" "0.70")

idx=0
for mn in "${MUSIC_NPYS[@]}"; do
  mstem=$(basename "$mn" .npy)
  for starts in "${START_LIST[@]}"; do
    for ew in "${EW_LIST[@]}"; do
      idx=$((idx+1))
      tag="${mstem}_s$(echo "$starts" | tr ',' '_')_ew${ew}"
      out_dir="$SCHED_ROOT/$tag"
      mkdir -p "$out_dir"
      python tools/schedule_emotion_aware_phrase.py \
        --report "$REPORT" \
        --music_npy "$mn" \
        --out "$out_dir/${tag}.npy" \
        --starts "$starts" \
        --emotion_weight "$ew" \
        --candidate_top_k 240 \
        --min_source_gap 90 \
        --blend_radius 14 \
        2>&1 | tee "$LOG_ROOT/schedule_${tag}.log" || true
    done
  done
done

echo "[3/6] Collect npy and build dataset"
COLLECT="$RUN_ROOT/collected_motion"
mkdir -p "$COLLECT"
find "$SCHED_ROOT" -name "*.npy" -type f | sort | while read -r f; do
  cp "$f" "$COLLECT/$(basename "$(dirname "$f")")_$(basename "$f")"
done

N_MOTION=$(find "$COLLECT" -name "*.npy" | wc -l | tr -d ' ')
echo "motion_count=$N_MOTION"
if [ "$N_MOTION" -lt 5 ]; then
  echo "ERROR: too few scheduled motions"
  exit 5
fi

DATASET="$RUN_ROOT/dataset"
python tools/build_emotion_refiner_dataset.py \
  --motion_dir "$COLLECT" \
  --music_npy "$MUSIC_NPY_LIST" \
  --out_dir "$DATASET" \
  --starts "0,32,64,96" \
  --seam_radius 8 \
  --noise_std 0.015 \
  --smooth_prob 0.50 \
  2>&1 | tee "$LOG_ROOT/build_dataset.log"

echo "[4/6] Train emotion-conditioned seam residual refiner"
TRAIN_OUT="$RUN_ROOT/train"
python train_emotion_refiner.py \
  --data_dir "$DATASET" \
  --out_dir "$TRAIN_OUT" \
  --epochs ${V17_EPOCHS:-800} \
  --batch_size ${V17_BATCH_SIZE:-8} \
  --lr ${V17_LR:-3e-4} \
  --lambda_music ${V17_LAMBDA_MUSIC:-0.04} \
  --lambda_preserve ${V17_LAMBDA_PRESERVE:-0.25} \
  --save_every 100 \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[5/6] Inference and render"
BEST="$TRAIN_OUT/checkpoints/best.pt"
FINAL="$TRAIN_OUT/checkpoints/final.pt"
CKPT="$BEST"
[ -f "$CKPT" ] || CKPT="$FINAL"

mkdir -p model
cp "$CKPT" model/emotion_conditioned_refiner.pt || true

REFINED="$RUN_ROOT/refined"
FIRST_AUDIO=$(echo "$AUDIO_LIST" | cut -d',' -f1)
INPUT="$COLLECT" AUDIO="$FIRST_AUDIO" CHECKPOINT="$CKPT" OUT_DIR="$REFINED" \
  bash scripts/run_emotion_refiner_inference.sh \
  2>&1 | tee "$LOG_ROOT/infer.log"

echo "[6/6] Write conclusion"
cat > "$RUN_ROOT/V17_EMOTION_REFINER_CONCLUSION.md" <<EOF
# V17 Emotion-Aware Refiner Overnight Conclusion

## Run
- RUN_ROOT: $RUN_ROOT
- LOG_ROOT: $LOG_ROOT
- Source report: $REPORT
- Audio list: $AUDIO_LIST
- Scheduled motions: $N_MOTION
- Dataset: $DATASET
- Checkpoint: $CKPT
- Refined outputs: $REFINED

## Loss design
- seam-weighted reconstruction
- velocity continuity
- acceleration/jitter
- root X/Z in-place lock
- outside-seam style preservation
- residual regularization
- weak music-emotion alignment

## First inspection
1. $LOG_ROOT/train.log
2. $REFINED/*.mp4
3. Compare with V16C scheduled outputs.

## Interpretation
If videos become too smooth or weaker, reduce V17_LAMBDA_MUSIC and increase lambda_preserve.
If boundary still jumps, increase seam_radius or lambda_vel.
EOF

zip -r "$RUN_ROOT/v17_emotion_refiner_package.zip" "$RUN_ROOT/V17_EMOTION_REFINER_CONCLUSION.md" "$LOG_ROOT" "$TRAIN_OUT/checkpoints" "$REFINED" 2>/dev/null || true

echo "============================================================"
echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CHECKPOINT=$CKPT"
echo "CONCLUSION=$RUN_ROOT/V17_EMOTION_REFINER_CONCLUSION.md"
echo "============================================================"
