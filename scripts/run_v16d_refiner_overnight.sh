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

RUN_ROOT=${RUN_ROOT:-output/night_v16d_refiner_${DATE}}
LOG_ROOT=${LOG_ROOT:-logs/night_v16d_refiner_${DATE}}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V16D Refiner Overnight Training"
echo "DATE=$DATE"
echo "SRC_RUN_ROOT=$SRC_RUN_ROOT"
echo "REPORT=$REPORT"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

if [ ! -f "$REPORT" ]; then
  echo "ERROR: visual_first_pool_report.json not found: $REPORT"
  exit 1
fi

echo "[1/8] Check required files"

for f in \
  tools/schedule_visual_first_phrase.py \
  tools/build_refiner_dataset.py \
  model/phrase_refiner.py \
  train_refiner.py \
  render_from_npy.py
do
  if [ ! -f "$f" ]; then
    echo "ERROR: missing $f"
    exit 2
  fi
done

echo "[2/8] Build many V16C phrase samples"

SCHED_ROOT="$RUN_ROOT/v16c_phrase_sources"
mkdir -p "$SCHED_ROOT"

AUDIO_LIST=(
  "test_music_bank/dunhuangwu2.wav"
  "test_music_bank/dunhuangwu3.wav"
  "test_music_bank/dunhuangwu4.wav"
)

START_LIST=(
  "0,32,64,96"
  "0,35,70,105"
  "0,30,60,90"
  "0,36,72,108"
)

GAP_LIST=(
  "70"
  "90"
  "120"
)

BLEND_LIST=(
  "12"
  "14"
  "16"
)

count=0

for audio in "${AUDIO_LIST[@]}"; do
  if [ ! -f "$audio" ]; then
    echo "WARN: missing audio $audio, skip"
    continue
  fi

  audio_stem=$(basename "$audio" .wav)

  for starts in "${START_LIST[@]}"; do
    for gap in "${GAP_LIST[@]}"; do
      for blend in "${BLEND_LIST[@]}"; do

        count=$((count + 1))
        tag="${audio_stem}_s$(echo "$starts" | tr ',' '_')_gap${gap}_b${blend}"
        out_dir="$SCHED_ROOT/$tag"
        mkdir -p "$out_dir"

        echo "------------------------------------------------------------"
        echo "Generate V16C sample #$count"
        echo "tag=$tag"
        echo "audio=$audio"
        echo "starts=$starts"
        echo "gap=$gap"
        echo "blend=$blend"
        echo "------------------------------------------------------------"

        python tools/schedule_visual_first_phrase.py \
          --report "$REPORT" \
          --out "$out_dir/${tag}_balanced.npy" \
          --num_frames 150 \
          --starts "$starts" \
          --candidate_top_k 240 \
          --transition_weight 0.65 \
          --entry_reset_weight 0.45 \
          --activity_weight 0.08 \
          --visual_weight 1.0 \
          --blend_radius "$blend" \
          --min_source_gap "$gap" \
          --reset_window 8 \
          --max_internal_offset 10 \
          --offset_step 2 \
          --min_remaining_frames 24 \
          2>&1 | tee "$LOG_ROOT/schedule_${tag}_balanced.log" || true

        python tools/schedule_visual_first_phrase.py \
          --report "$REPORT" \
          --out "$out_dir/${tag}_visual_heavy.npy" \
          --num_frames 150 \
          --starts "$starts" \
          --candidate_top_k 240 \
          --transition_weight 0.45 \
          --entry_reset_weight 0.35 \
          --activity_weight 0.10 \
          --visual_weight 1.25 \
          --blend_radius "$blend" \
          --min_source_gap "$gap" \
          --reset_window 8 \
          --max_internal_offset 10 \
          --offset_step 2 \
          --min_remaining_frames 24 \
          2>&1 | tee "$LOG_ROOT/schedule_${tag}_visual_heavy.log" || true

      done
    done
  done
done

echo "[3/8] Collect V16C npy files"

COLLECT_DIR="$RUN_ROOT/v16c_collected_npy"
mkdir -p "$COLLECT_DIR"

find "$SCHED_ROOT" -name "*.npy" -type f | sort | while read -r f; do
  base=$(basename "$f")
  parent=$(basename "$(dirname "$f")")
  cp "$f" "$COLLECT_DIR/${parent}_${base}"
done

N_NPY=$(find "$COLLECT_DIR" -name "*.npy" | wc -l | tr -d ' ')
echo "Collected npy count: $N_NPY"

if [ "$N_NPY" -lt 10 ]; then
  echo "ERROR: too few V16C phrase samples collected."
  exit 3
fi

echo "[4/8] Build V16D refiner dataset"

DATASET_DIR="$RUN_ROOT/v16d_refiner_dataset"

python tools/build_refiner_dataset.py \
  --v16c_input_dir "$COLLECT_DIR" \
  --motion_unit_dir data/dunhuang_choreo_unit_rag \
  --out_dir "$DATASET_DIR" \
  --phrase_len 45 \
  2>&1 | tee "$LOG_ROOT/build_refiner_dataset.log"

N_DATA=$(find "$DATASET_DIR" -name "*.npy" | wc -l | tr -d ' ')
echo "Dataset phrase count: $N_DATA"

if [ "$N_DATA" -lt 20 ]; then
  echo "WARN: dataset is small. Training will still run, but result may be weak."
fi

echo "[5/8] Train V16D phrase refiner"

mkdir -p "$RUN_ROOT/checkpoints"

# 当前模板 train_refiner.py 默认保存 phrase_refiner.pt 到当前目录。
# 这里在一个独立 workdir 内训练，避免覆盖旧模型。
TRAIN_WORKDIR="$RUN_ROOT/train_workdir"
mkdir -p "$TRAIN_WORKDIR"

(
  cd "$TRAIN_WORKDIR"
  export PYTHONPATH=/home/disk/lsm/storage/EDGE:${PYTHONPATH:-}

  python /home/disk/lsm/storage/EDGE/train_refiner.py \
    --data_dir "/home/disk/lsm/storage/EDGE/$DATASET_DIR" \
    --epochs ${V16D_EPOCHS:-800} \
    --batch_size ${V16D_BATCH_SIZE:-8} \
    --lr ${V16D_LR:-1e-3} \
    --use_onset ${EDGE_USE_ONSET:-0} \
    --scope phrase \
    2>&1 | tee "/home/disk/lsm/storage/EDGE/$LOG_ROOT/train_refiner.log"

  if [ -f phrase_refiner.pt ]; then
    cp phrase_refiner.pt "/home/disk/lsm/storage/EDGE/$RUN_ROOT/checkpoints/phrase_refiner_final.pt"
  fi
)

if [ ! -f "$RUN_ROOT/checkpoints/phrase_refiner_final.pt" ]; then
  echo "ERROR: refiner checkpoint not found."
  exit 4
fi

mkdir -p model
cp "$RUN_ROOT/checkpoints/phrase_refiner_final.pt" model/phrase_refiner.pt

echo "[6/8] Run V16D refiner inference"

REFINED_DIR="$RUN_ROOT/v16d_refined"
mkdir -p "$REFINED_DIR"

INPUT_DIR="$COLLECT_DIR" \
OUT_DIR="$REFINED_DIR" \
USE_ONSET=${EDGE_USE_ONSET:-0} \
PHRASE_REFINER_MODEL="model/phrase_refiner.pt" \
bash scripts/run_refiner_inference.sh \
  2>&1 | tee "$LOG_ROOT/refiner_inference.log"

echo "[7/8] Render selected refined outputs"

RENDER_DIR="$RUN_ROOT/renders"
mkdir -p "$RENDER_DIR"

render_count=0
for f in $(find "$REFINED_DIR" -name "*.npy" | sort | head -20); do
  render_count=$((render_count + 1))
  stem=$(basename "$f" .npy)

  python render_from_npy.py \
    --motion "$f" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$RENDER_DIR/${stem}_fixed.mp4" \
    --camera_mode fixed || true

  python render_from_npy.py \
    --motion "$f" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$RENDER_DIR/${stem}_follow.mp4" \
    --camera_mode follow || true
done

echo "Rendered selected count: $render_count"

echo "[8/8] Write conclusion"

cat > "$RUN_ROOT/V16D_OVERNIGHT_CONCLUSION.md" <<EOF
# V16D Refiner Overnight Conclusion

## Run

- RUN_ROOT: $RUN_ROOT
- LOG_ROOT: $LOG_ROOT
- Source V16C report: $REPORT
- Collected V16C npy count: $N_NPY
- Refiner dataset phrase count: $N_DATA
- Checkpoint: $RUN_ROOT/checkpoints/phrase_refiner_final.pt
- Refined outputs: $REFINED_DIR
- Render outputs: $RENDER_DIR

## Main idea

This run trains a phrase-level V16D Refiner from V16C stable scheduled phrases.

The goal is not to replace Visual-First selection, but to distill the V16C phrase composition prior into a lightweight refiner.

## What to inspect first

1. Training log:
   - $LOG_ROOT/train_refiner.log

2. Refined npy outputs:
   - $REFINED_DIR

3. Rendered videos:
   - $RENDER_DIR

## Current judgement rule

- If refined renders preserve V16C style but reduce seam artifacts, V16D Refiner is useful.
- If refined renders become weaker or over-smoothed, keep V16C scheduler as the main method and use V16D only as an ablation.
- If loss decreases but videos get worse, this simple MSE refiner is too weak and should be replaced by seam-mask / residual-only training.
EOF

zip -r "$RUN_ROOT/v16d_refiner_overnight_package.zip" \
  "$RUN_ROOT/V16D_OVERNIGHT_CONCLUSION.md" \
  "$RUN_ROOT/checkpoints" \
  "$LOG_ROOT" \
  "$RENDER_DIR" \
  2>/dev/null || true

echo "============================================================"
echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CHECKPOINT=$RUN_ROOT/checkpoints/phrase_refiner_final.pt"
echo "CONCLUSION=$RUN_ROOT/V16D_OVERNIGHT_CONCLUSION.md"
echo "PACKAGE=$RUN_ROOT/v16d_refiner_overnight_package.zip"
echo "============================================================"
