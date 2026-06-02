#!/usr/bin/env bash
# V16C scheduler-only rerun for existing Visual-First prior pool.
# Usage:
#   cd /home/disk/lsm/storage/EDGE
#   bash scripts/run_v16c_visual_first_scheduler.sh
#
# Optional:
#   RUN_ROOT=output/night_v16_visual_first_20260602_204417 bash scripts/run_v16c_visual_first_scheduler.sh
#   EDGE_V16C_STARTS="0,32,64,96" EDGE_V16C_MIN_SOURCE_GAP=90 bash scripts/run_v16c_visual_first_scheduler.sh

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

RUN_ROOT=${RUN_ROOT:-$(ls -td output/night_v16_visual_first_* | head -1)}
REPORT=${REPORT:-"$RUN_ROOT/visual_first_pool_report.json"}
AUDIO=${AUDIO:-test_music_bank/dunhuangwu2.wav}
OUT_DIR=${OUT_DIR:-"$RUN_ROOT/scheduled_demos_v16c"}
LOG_DIR=${LOG_DIR:-"$OUT_DIR/logs"}

mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -f "$REPORT" ]; then
  echo "ERROR: report not found: $REPORT"
  exit 1
fi

if [ ! -f "$AUDIO" ]; then
  echo "ERROR: audio not found: $AUDIO"
  exit 1
fi

export EDGE_V16C_STARTS=${EDGE_V16C_STARTS:-"0,32,64,96"}
export EDGE_V16C_CANDIDATE_TOP_K=${EDGE_V16C_CANDIDATE_TOP_K:-240}
export EDGE_V16C_MIN_SOURCE_GAP=${EDGE_V16C_MIN_SOURCE_GAP:-90}
export EDGE_V16C_BLEND_RADIUS=${EDGE_V16C_BLEND_RADIUS:-14}
export EDGE_V16C_RESET_WINDOW=${EDGE_V16C_RESET_WINDOW:-8}
export EDGE_V16C_MAX_INTERNAL_OFFSET=${EDGE_V16C_MAX_INTERNAL_OFFSET:-10}
export EDGE_V16C_OFFSET_STEP=${EDGE_V16C_OFFSET_STEP:-2}
export EDGE_V16C_MIN_REMAINING_FRAMES=${EDGE_V16C_MIN_REMAINING_FRAMES:-24}

echo "============================================================"
echo "V16C Visual-First Boundary-aware Scheduler"
echo "RUN_ROOT=$RUN_ROOT"
echo "REPORT=$REPORT"
echo "AUDIO=$AUDIO"
echo "OUT_DIR=$OUT_DIR"
echo "EDGE_V16C_STARTS=$EDGE_V16C_STARTS"
echo "EDGE_V16C_MIN_SOURCE_GAP=$EDGE_V16C_MIN_SOURCE_GAP"
echo "EDGE_V16C_BLEND_RADIUS=$EDGE_V16C_BLEND_RADIUS"
echo "============================================================"

echo "[1/3] Balanced schedule"
python tools/schedule_visual_first_phrase.py \
  --report "$REPORT" \
  --out "$OUT_DIR/dw2_v16c_balanced.npy" \
  --num_frames 150 \
  --starts "$EDGE_V16C_STARTS" \
  --candidate_top_k "$EDGE_V16C_CANDIDATE_TOP_K" \
  --transition_weight 0.65 \
  --entry_reset_weight 0.45 \
  --activity_weight 0.08 \
  --visual_weight 1.0 \
  --blend_radius "$EDGE_V16C_BLEND_RADIUS" \
  --min_source_gap "$EDGE_V16C_MIN_SOURCE_GAP" \
  --reset_window "$EDGE_V16C_RESET_WINDOW" \
  --max_internal_offset "$EDGE_V16C_MAX_INTERNAL_OFFSET" \
  --offset_step "$EDGE_V16C_OFFSET_STEP" \
  --min_remaining_frames "$EDGE_V16C_MIN_REMAINING_FRAMES" \
  2>&1 | tee "$LOG_DIR/schedule_balanced.log"

echo "[2/3] Visual-heavy schedule"
python tools/schedule_visual_first_phrase.py \
  --report "$REPORT" \
  --out "$OUT_DIR/dw2_v16c_visual_heavy.npy" \
  --num_frames 150 \
  --starts "$EDGE_V16C_STARTS" \
  --candidate_top_k "$EDGE_V16C_CANDIDATE_TOP_K" \
  --transition_weight 0.45 \
  --entry_reset_weight 0.35 \
  --activity_weight 0.10 \
  --visual_weight 1.25 \
  --blend_radius 16 \
  --min_source_gap "$EDGE_V16C_MIN_SOURCE_GAP" \
  --reset_window "$EDGE_V16C_RESET_WINDOW" \
  --max_internal_offset "$EDGE_V16C_MAX_INTERNAL_OFFSET" \
  --offset_step "$EDGE_V16C_OFFSET_STEP" \
  --min_remaining_frames "$EDGE_V16C_MIN_REMAINING_FRAMES" \
  2>&1 | tee "$LOG_DIR/schedule_visual_heavy.log"

echo "[3/3] Render follow/fixed videos"
for f in "$OUT_DIR"/dw2_v16c_*.npy; do
  stem=$(basename "$f" .npy)

  python render_from_npy.py \
    --motion "$f" \
    --audio "$AUDIO" \
    --output "$OUT_DIR/${stem}_follow.mp4" \
    --camera_mode follow || true

  python render_from_npy.py \
    --motion "$f" \
    --audio "$AUDIO" \
    --output "$OUT_DIR/${stem}_fixed.mp4" \
    --camera_mode fixed || true
done

cat > "$OUT_DIR/V16C_SCHEDULER_CONCLUSION.md" <<EOF
# V16C Scheduler Conclusion

## Inputs

- RUN_ROOT: $RUN_ROOT
- REPORT: $REPORT
- AUDIO: $AUDIO
- STARTS: $EDGE_V16C_STARTS
- MIN_SOURCE_GAP: $EDGE_V16C_MIN_SOURCE_GAP
- BLEND_RADIUS: $EDGE_V16C_BLEND_RADIUS

## Outputs to inspect first

- $OUT_DIR/dw2_v16c_balanced_fixed.mp4
- $OUT_DIR/dw2_v16c_balanced_follow.mp4
- $OUT_DIR/dw2_v16c_visual_heavy_fixed.mp4
- $OUT_DIR/dw2_v16c_visual_heavy_follow.mp4

## What changed

V16C keeps Visual-First prior selection and fixes phrase reset by adding:

1. source-index diversity;
2. entry-reset penalty;
3. trend-window transition cost;
4. internal unit offsets;
5. longer boundary cross-fade.

This is scheduler-only and does not require retraining.
EOF

echo "============================================================"
echo "DONE"
echo "OUT_DIR=$OUT_DIR"
echo "Conclusion: $OUT_DIR/V16C_SCHEDULER_CONCLUSION.md"
ls -lh "$OUT_DIR"/*.mp4 2>/dev/null || true
echo "============================================================"
