#!/usr/bin/env bash
set -uo pipefail

cd /home/disk/lsm/storage/EDGE

mkdir -p logs output/v3d_overnight_eval output/v3d_softprior_eval output/v3c2_eval

DATA_PATH="data/dunhuang_bvh/stationary_whitelist_v3_27units"
CKPT_V3B="runs/train_nextgen/stationary_v3b_whitelist24_from_v16_x0w04_noDCT_energy/weights/train-300.pt"
CKPT_V16="runs/train_nextgen/strict_single_unit45_recon_v16_smooth_dense8_from_v15/weights/train-5000.pt"

PRIOR_76207="data/dunhuang_bvh/stationary_whitelist_v3_27units/v3_unit_023_unit_76207_gt45.pkl"
PRIOR_5147="data/dunhuang_bvh/stationary_whitelist_v3_27units/v3_unit_022_unit_5147_gt45.pkl"

AUDIO="test_music_bank/dunhuangwu2.wav"

echo "==== Overnight V3D/V3C2 suite started at $(date) ===="
echo "DATA_PATH=$DATA_PATH"
echo "CKPT_V3B=$CKPT_V3B"
echo "CKPT_V16=$CKPT_V16"
echo "PRIOR_76207=$PRIOR_76207"
echo "PRIOR_5147=$PRIOR_5147"
echo "===================================================="

check_file() {
  if [ ! -f "$1" ]; then
    echo "❌ missing file: $1"
    return 1
  fi
  return 0
}

check_file "$CKPT_V3B" || exit 1
check_file "$CKPT_V16" || exit 1
check_file "$PRIOR_76207" || exit 1
check_file "$PRIOR_5147" || exit 1
check_file "$AUDIO" || exit 1

eval_and_render() {
  local npy="$1"
  local tag="$2"

  echo ""
  echo "==== EVAL $tag ===="
  PYTHONPATH=. python tools/eval_fk_visible_motion.py \
    --pred "$npy" \
    --gt_dir "$DATA_PATH" \
    2>&1 | tee "logs/eval_${tag}.log"

  echo ""
  echo "==== RENDER FIXED $tag ===="
  python render_from_npy.py \
    --motion "$npy" \
    --audio "$AUDIO" \
    --output "output/v3d_overnight_eval/${tag}_fixed.mp4" \
    --camera_mode fixed \
    2>&1 | tee "logs/render_${tag}_fixed.log"

  echo ""
  echo "==== RENDER FOLLOW $tag ===="
  python render_from_npy.py \
    --motion "$npy" \
    --audio "$AUDIO" \
    --output "output/v3d_overnight_eval/${tag}_follow.mp4" \
    --camera_mode follow \
    2>&1 | tee "logs/render_${tag}_follow.log"
}

sample_softprior() {
  local ckpt="$1"
  local prior="$2"
  local tag="$3"
  local strength="$4"
  local start_frac="$5"
  local gamma="$6"
  local torso_scale="$7"

  local out="output/v3d_overnight_eval/${tag}.npy"

  echo ""
  echo "==== SAMPLE $tag ===="
  echo "ckpt=$ckpt"
  echo "prior=$prior"
  echo "strength=$strength start_frac=$start_frac gamma=$gamma torso_scale=$torso_scale"

  EDGE_V3D_TORSO_PRIOR_SCALE="$torso_scale" \
  EDGE_V3D_NECK_HEAD_PRIOR_SCALE=1.00 \
  EDGE_V3D_ARMS_PRIOR_SCALE=1.00 \
  python tools/sample_v3d_upper_soft_prior.py \
    --checkpoint "$ckpt" \
    --prior "$prior" \
    --out "$out" \
    --batch_size 2 \
    --strength "$strength" \
    --start_frac "$start_frac" \
    --gamma "$gamma" \
    2>&1 | tee "logs/sample_${tag}.log"

  if [ -f "$out" ]; then
    eval_and_render "$out" "$tag"
  else
    echo "❌ sampling failed, missing $out"
  fi
}

# ------------------------------------------------------------
# Part A: V3B + retrieved upper_safe_plus soft prior grid
# ------------------------------------------------------------

echo ""
echo "================ PART A: V3B + V3D soft prior grid ================"

sample_softprior "$CKPT_V3B" "$PRIOR_76207" "v3d_v3b_76207_s022_torso65" 0.22 0.55 1.5 0.65
sample_softprior "$CKPT_V3B" "$PRIOR_76207" "v3d_v3b_76207_s035_torso65" 0.35 0.45 1.2 0.65
sample_softprior "$CKPT_V3B" "$PRIOR_76207" "v3d_v3b_76207_s035_torso100" 0.35 0.45 1.2 1.00
sample_softprior "$CKPT_V3B" "$PRIOR_76207" "v3d_v3b_76207_s050_torso100" 0.50 0.35 1.0 1.00

sample_softprior "$CKPT_V3B" "$PRIOR_5147" "v3d_v3b_5147_s022_torso65" 0.22 0.55 1.5 0.65
sample_softprior "$CKPT_V3B" "$PRIOR_5147" "v3d_v3b_5147_s035_torso100" 0.35 0.45 1.2 1.00

# ------------------------------------------------------------
# Part B: Train V3C2 visible-FK checkpoint
# ------------------------------------------------------------

echo ""
echo "================ PART B: Train V3C2 visible-FK model ================"

EXP_V3C2="stationary_v3c2_whitelist24_visibleFK_x0w02_fk25_range085_e300"

DATA_PATH="$DATA_PATH" \
EXP_NAME="$EXP_V3C2" \
CHECKPOINT="$CKPT_V16" \
BATCH_SIZE=2 \
EPOCHS=300 \
SAVE_INTERVAL=100 \
MAX_TRAIN_BATCHES=0 \
EDGE_X0_RECON_LOSS_WEIGHT=0.20 \
EDGE_V3C_VISIBLE_FK_WEIGHT=25.0 \
EDGE_V3C_ACTIVITY_FLOOR=0.85 \
EDGE_V3C_RANGE_FLOOR=0.85 \
EDGE_V3C_FK_RANGE_WEIGHT=8.0 \
EDGE_V3C_FK_HAND_RANGE_WEIGHT=12.0 \
EDGE_V3C_FK_ACTIVITY_WEIGHT=4.0 \
EDGE_V3C_FK_HAND_WEIGHT=1.5 \
EDGE_V3C_VISIBLE_FK_DEBUG=0 \
bash scripts/run_v3c_visible_fk.sh \
  2>&1 | tee "logs/train_${EXP_V3C2}_overnight_wrapper.log"

CKPT_V3C2="runs/train_nextgen/${EXP_V3C2}/weights/train-300.pt"

if [ -f "$CKPT_V3C2" ]; then
  echo "✅ V3C2 checkpoint found: $CKPT_V3C2"

  # Free sample from V3C2 auto render, if available.
  V3C2_SAMPLE=$(find "runs/train_nextgen/${EXP_V3C2}" -type f -name "*.npy" | head -1)
  if [ -n "$V3C2_SAMPLE" ] && [ -f "$V3C2_SAMPLE" ]; then
    eval_and_render "$V3C2_SAMPLE" "v3c2_free_sample"
  fi

  # V3C2 + soft prior, same prior as best current direction.
  sample_softprior "$CKPT_V3C2" "$PRIOR_76207" "v3d_v3c2_76207_s035_torso100" 0.35 0.45 1.2 1.00
  sample_softprior "$CKPT_V3C2" "$PRIOR_76207" "v3d_v3c2_76207_s022_torso100" 0.22 0.55 1.5 1.00
else
  echo "⚠️ V3C2 checkpoint not found, skipping V3C2 soft-prior sampling: $CKPT_V3C2"
fi

echo ""
echo "==== Overnight V3D/V3C2 suite finished at $(date) ===="

# Summarize key FK ratios into one file.
{
  echo "# Overnight V3D/V3C2 eval summary"
  echo "date: $(date)"
  echo ""
  for f in logs/eval_v3d_*.log logs/eval_v3c2*.log; do
    [ -f "$f" ] || continue
    echo "## $f"
    grep -E "wrists_hands_range_mean_ratio|arms_range_mean_ratio|upper_safe_plus_range_mean_ratio|wrists_hands_speed_mean_ratio|arms_speed_mean_ratio" "$f" || true
    echo ""
  done
} > logs/overnight_v3d_v3c2_summary.md

echo "summary: logs/overnight_v3d_v3c2_summary.md"
