#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

CKPT="${CKPT:-runs/train_nextgen/stationary_v3f_bodycentered_x0w010_fk8_resp_e100/weights/train-100.pt}"

# fallback to V3B if V3F checkpoint is not ready
if [ ! -f "$CKPT" ]; then
  echo "⚠️ CKPT not found: $CKPT"
  echo "fallback to V3B checkpoint"
  CKPT="runs/train_nextgen/stationary_v3b_whitelist24_from_v16_x0w04_noDCT_energy/weights/train-300.pt"
fi

PRIOR="${PRIOR:-data/dunhuang_bvh/stationary_whitelist_v3_27units/v3_unit_023_unit_76207_gt45.pkl}"
AUDIO="${AUDIO:-test_music_bank/dunhuangwu2.wav}"
DATA_PATH="${DATA_PATH:-data/dunhuang_bvh/stationary_whitelist_v3_27units}"

OUT_ROOT="${OUT_ROOT:-output/v4_music_adaptive_20260518}"
LOG_ROOT="${LOG_ROOT:-logs/v4_music_adaptive_20260518}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

if [ ! -f "$CKPT" ]; then
  echo "❌ missing CKPT: $CKPT"
  exit 1
fi
if [ ! -f "$PRIOR" ]; then
  echo "❌ missing PRIOR: $PRIOR"
  exit 1
fi
if [ ! -f "$AUDIO" ]; then
  echo "❌ missing AUDIO: $AUDIO"
  exit 1
fi

echo "==== V4 music-adaptive eval ===="
echo "CKPT=$CKPT"
echo "PRIOR=$PRIOR"
echo "AUDIO=$AUDIO"
echo "OUT_ROOT=$OUT_ROOT"
echo "================================"

run_v4_one () {
  local tag="$1"
  local warp="$2"
  local min_speed="$3"
  local max_speed="$4"
  local strength="$5"
  local start_frac="$6"
  local gamma="$7"

  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"

  python tools/sample_v4_music_adaptive_prior.py \
    --checkpoint "$CKPT" \
    --prior "$PRIOR" \
    --audio "$AUDIO" \
    --out "$out_dir/${tag}.npy" \
    --batch_size 2 \
    --seq_len 45 \
    --strength "$strength" \
    --start_frac "$start_frac" \
    --gamma "$gamma" \
    --warp_strength "$warp" \
    --min_speed "$min_speed" \
    --max_speed "$max_speed" \
    --rhythm_min_gain 0.65 \
    --rhythm_max_gain 1.35 \
    --torso_scale 1.00 \
    --neck_head_scale 1.00 \
    --arms_scale 1.00 \
    2>&1 | tee "$LOG_ROOT/sample_${tag}.log"

  PYTHONPATH=. python tools/eval_fk_visible_motion.py \
    --pred "$out_dir/${tag}.npy" \
    --gt_dir "$DATA_PATH" \
    2>&1 | tee "$LOG_ROOT/eval_fk_${tag}.log"

  if [ -f tools/eval_body_centered_motion.py ]; then
    PYTHONPATH=. python tools/eval_body_centered_motion.py \
      --pred "$out_dir/${tag}.npy" \
      --gt_dir "$DATA_PATH" \
      2>&1 | tee "$LOG_ROOT/eval_bc_${tag}.log"
  fi

  python render_from_npy.py \
    --motion "$out_dir/${tag}.npy" \
    --audio "$AUDIO" \
    --output "$out_dir/${tag}_fixed.mp4" \
    --camera_mode fixed \
    2>&1 | tee "$LOG_ROOT/render_${tag}_fixed.log"

  python render_from_npy.py \
    --motion "$out_dir/${tag}_warped_prior.npy" \
    --audio "$AUDIO" \
    --output "$out_dir/${tag}_warped_prior_fixed.mp4" \
    --camera_mode fixed \
    2>&1 | tee "$LOG_ROOT/render_${tag}_warped_prior_fixed.log"
}

# Baseline-like: no temporal warping, but rhythm per-frame gain active.
run_v4_one "v4_nowarp_s035" 0.0 0.55 1.85 0.35 0.45 1.2

# Main V4: music-adaptive temporal warping.
run_v4_one "v4_warp100_s035" 1.0 0.55 1.85 0.35 0.45 1.2

# Strong prior variant for comparison.
run_v4_one "v4_warp100_s050" 1.0 0.55 1.85 0.50 0.35 1.0

# Conservative warp if main variant looks too elastic.
run_v4_one "v4_warp060_s035" 0.6 0.70 1.55 0.35 0.45 1.2

{
  echo "# V4 music adaptive summary"
  echo "date: $(date)"
  echo "CKPT=$CKPT"
  echo "PRIOR=$PRIOR"
  echo "AUDIO=$AUDIO"
  echo ""

  for f in "$LOG_ROOT"/eval_fk_*.log; do
    [ -f "$f" ] || continue
    echo "## $f"
    grep -E "wrists_hands_range_mean_ratio|wrists_hands_speed_mean_ratio|arms_range_mean_ratio|arms_speed_mean_ratio|upper_safe_plus_range_mean_ratio|upper_safe_plus_speed_mean_ratio|torso_range_mean|root_range_mean" "$f" || true
    echo ""
  done

  for f in "$LOG_ROOT"/eval_bc_*.log; do
    [ -f "$f" ] || continue
    echo "## $f"
    grep -E "torso_bc_range_mean_ratio|arms_bc_range_mean_ratio|hands_bc_range_mean_ratio|upper_bc_range_mean_ratio|torso_to_arms_range_ratio|root_xz_speed_mean_ratio" "$f" || true
    echo ""
  done
} | tee "$LOG_ROOT/v4_summary.md"

echo "summary: $LOG_ROOT/v4_summary.md"
