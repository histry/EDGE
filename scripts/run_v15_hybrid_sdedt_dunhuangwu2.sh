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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# Disable heavy optional inference branches.
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENERGY_COND=0
export EDGE_DISABLE_TRAJ_COND=1

CKPT="runs/train_nextgen/v15_onset_phrase_safe_recon_si20_20260526_153307/weights/train-260.pt"
PRIOR="output/night_v15_onset_phrase_20260526_001018/dunhuangwu2/dunhuangwu2_v15_onset_phrase_prior.npy"
AUDIO="test_music_bank/dunhuangwu2.wav"

OUT_DIR="output/v15_hybrid_sdedt_dunhuangwu2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "OUT_DIR=$OUT_DIR"

# Conservative: mostly preserve style, lightly repair seams.
python hybrid_sdedt_refiner.py \
  --checkpoint "$CKPT" \
  --prior "$PRIOR" \
  --audio "$AUDIO" \
  --out "$OUT_DIR/dw2_hybrid_sdedt_conservative.npy" \
  --seams 35,74,105,108 \
  --base_t 15 \
  --seam_t 110 \
  --sampling_steps 24 \
  --core_radius 3 \
  --buffer_radius 12 \
  --global_floor 0.00 \
  --max_rewrite_weight 0.38 \
  --lower_scale 0.08 \
  --torso_scale 0.35 \
  --upper_scale 0.55 \
  --mixed_precision fp16 \
  --no_ema \
  2>&1 | tee "$OUT_DIR/conservative.log"

# Balanced: stronger seam repair, still style-preserving.
python hybrid_sdedt_refiner.py \
  --checkpoint "$CKPT" \
  --prior "$PRIOR" \
  --audio "$AUDIO" \
  --out "$OUT_DIR/dw2_hybrid_sdedt_balanced.npy" \
  --seams 35,74,105,108 \
  --base_t 20 \
  --seam_t 150 \
  --sampling_steps 30 \
  --core_radius 4 \
  --buffer_radius 14 \
  --global_floor 0.00 \
  --max_rewrite_weight 0.52 \
  --lower_scale 0.10 \
  --torso_scale 0.45 \
  --upper_scale 0.70 \
  --mixed_precision fp16 \
  --no_ema \
  2>&1 | tee "$OUT_DIR/balanced.log"

# Render prior and refined results.
python render_from_npy.py \
  --motion "$PRIOR" \
  --audio "$AUDIO" \
  --output "$OUT_DIR/dw2_prior_follow.mp4" \
  --camera_mode follow

for f in "$OUT_DIR"/*.npy; do
  stem=$(basename "$f" .npy)
  python render_from_npy.py \
    --motion "$f" \
    --audio "$AUDIO" \
    --output "$OUT_DIR/${stem}_follow.mp4" \
    --camera_mode follow

  python render_from_npy.py \
    --motion "$f" \
    --audio "$AUDIO" \
    --output "$OUT_DIR/${stem}_fixed.mp4" \
    --camera_mode fixed
done

echo "=== outputs ==="
find "$OUT_DIR" -maxdepth 1 -type f | sort
echo "DONE: $OUT_DIR"
