#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

DATA_DIR="data/dunhuang_bvh/stationary_expr_subset_e150"
MUSIC="test_music_bank/dunhuangwu2.wav"
OUT_DIR="output/single_clip_recon"
KF_DIR="test_keyframes/single_clip_recon"
LOG_DIR="logs"
RUN_NAME="strict_single_clip_recon_v11_xattn_3000steps_b1"

mkdir -p "$OUT_DIR/videos" "$KF_DIR" "$LOG_DIR"

echo "========== GPU BEFORE =========="
nvidia-smi || true

echo "========== STEP 1: export GT clip / keyframes =========="
python scripts/export_single_clip_recon_assets.py \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --keyframe_dir "$KF_DIR" \
  --seq_len 150 \
  2>&1 | tee "$LOG_DIR/${RUN_NAME}_export_assets.log"

source "$OUT_DIR/asset_paths.env"

echo "========== STEP 2: render GT first =========="
python render_from_npy.py \
  --motion "$GT_CLIP" \
  --audio "$MUSIC" \
  --output "$OUT_DIR/videos/gt_clip_fixed.mp4" \
  --camera_mode fixed \
  2>&1 | tee "$LOG_DIR/${RUN_NAME}_render_gt_fixed.log" || true

python render_from_npy.py \
  --motion "$GT_ROOTLOCK" \
  --audio "$MUSIC" \
  --output "$OUT_DIR/videos/gt_rootlock_follow.mp4" \
  --camera_mode follow \
  2>&1 | tee "$LOG_DIR/${RUN_NAME}_render_gt_rootlock_follow.log" || true

echo "========== STEP 3: choose base checkpoint =========="
BASE_CKPT="runs/train_nextgen/v11_inplace_expr_e3_b1_nockpt_quick/weights/train-3.pt"
if [ ! -f "$BASE_CKPT" ]; then
  BASE_CKPT="runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt"
fi
if [ ! -f "$BASE_CKPT" ]; then
  echo "❌ base checkpoint not found. Please set BASE_CKPT manually in this script."
  exit 1
fi
echo "BASE_CKPT=$BASE_CKPT"

echo "========== STEP 4: strict single-window overfit, 3000 effective steps =========="
unset EDGE_DIFF_CONTACT_LOSS || true
unset EDGE_BEAT_GUIDANCE || true
unset EDGE_UNIT_SOFT_PRIOR || true
unset EDGE_UNIT_PRIOR_REQUIRED || true

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1 \
EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1 \
EDGE_AUDIO_DEVICE=cpu \
EDGE_DYNAMIC_TRAJ_CFG=0 \
EDGE_GAIT_PHASE_COND=0 \
EDGE_GAIT_CONTACT_LOSS=0 \
EDGE_TRAJ_PHYSICS_FEATURES=0 \
EDGE_TRAJ_FOURIER_FEATURES=0 \
EDGE_TRAJ_SPARSE_WAYPOINT=0 \
EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
EDGE_ENABLE_RAG_SUMMARY_TOKEN=1 \
EDGE_V11_CROSS_ATTN_RAG=1 \
EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=0 \
EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0 \
EDGE_TEXT_CONTEXT_TRAIN_SELF=1 \
EDGE_TEXT_CONTEXT_REQUIRE_GRAD=1 \
EDGE_TEXT_CONTEXT_MIN_GRAD_NORM=1e-10 \
python train.py \
  --project runs/train_nextgen \
  --exp_name "$RUN_NAME" \
  --train_stage adapter \
  --adapter_train_decoder \
  --checkpoint "$BASE_CKPT" \
  --data_path "$DATA_DIR" \
  --feature_type hybrid \
  --audio_pairing_mode none \
  --batch_size 1 \
  --epochs 3000 \
  --learning_rate 1e-5 \
  --enable_rag_summary_token \
  --save_interval 250 \
  --val_batches 0 \
  --max_train_batches 1 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --mixed_precision bf16 \
  2>&1 | tee "$LOG_DIR/${RUN_NAME}_train.log"

echo "========== STEP 5: reconstruction generation on saved checkpoints =========="
METRICS="$OUT_DIR/${RUN_NAME}_metrics.csv"
rm -f "$METRICS"

# generate_controlled_v9.py accepts the same CLI as generate_controlled.py.
# We run both pose_space=normalized and pose_space=physical because the subset representation space must be verified.
for E in 250 500 1000 1500 2000 2500 3000; do
  CKPT="runs/train_nextgen/${RUN_NAME}/weights/train-${E}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "skip missing checkpoint: $CKPT"
    continue
  fi

  for POSE_SPACE in normalized physical; do
    TAG="e${E}_${POSE_SPACE}"
    OUT_NPY="$OUT_DIR/${RUN_NAME}_${TAG}.npy"
    GEN_LOG="$LOG_DIR/${RUN_NAME}_gen_${TAG}.log"

    echo "----- generating $TAG -----"

    set +e
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1 \
    EDGE_AUDIO_DEVICE=cpu \
    EDGE_DYNAMIC_TRAJ_CFG=0 \
    EDGE_GAIT_PHASE_COND=0 \
    EDGE_GAIT_CONTACT_LOSS=0 \
    EDGE_TRAJ_PHYSICS_FEATURES=0 \
    EDGE_TRAJ_FOURIER_FEATURES=0 \
    EDGE_TRAJ_SPARSE_WAYPOINT=0 \
    EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
    EDGE_ENABLE_RAG_SUMMARY_TOKEN=1 \
    EDGE_V11_CROSS_ATTN_RAG=1 \
    EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=0 \
    EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0 \
    EDGE_STRICT_EXPERIMENT_GUARD=1 \
    python generate_controlled_v9.py \
      --checkpoint "$CKPT" \
      --music "$MUSIC" \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --mid_poses "$MID_POSES" \
      --mid_pose_frames "$MID_FRAMES" \
      --out "$OUT_NPY" \
      --feature_type hybrid \
      --audio_dim 803 \
      --seq_len 150 \
      --num_frames 150 \
      --sampler ddim \
      --no_tto \
      --trajectory "0,0" \
      --uniform_trajectory_timing \
      --linear_trajectory \
      --endpoint_keyframe_strength 0.90 \
      --mid_keyframe_strength 0.25 \
      --infer_keyframe_width 0 \
      --pose_space "$POSE_SPACE" \
      --mixed_precision bf16 \
      2>&1 | tee "$GEN_LOG"
    GEN_STATUS=${PIPESTATUS[0]}
    set -e

    if grep -E "kept newly initialized keys|ignored unexpected keys" "$GEN_LOG" >/dev/null 2>&1; then
      echo "⚠️ WARNING: checkpoint architecture mismatch found in $GEN_LOG"
    fi

    if [ "$GEN_STATUS" -ne 0 ] || [ ! -f "$OUT_NPY" ]; then
      echo "⚠️ generation failed for $TAG"
      continue
    fi

    ROOTLOCK_NPY="$OUT_DIR/${RUN_NAME}_${TAG}_rootlock.npy"
    python scripts/rootlock_and_recon_metrics.py \
      --motion "$OUT_NPY" \
      --gt "$GT_CLIP" \
      --out_rootlock "$ROOTLOCK_NPY" \
      --metrics_csv "$METRICS" \
      --tag "$TAG" \
      2>&1 | tee "$LOG_DIR/${RUN_NAME}_metrics_${TAG}.log"

    python render_from_npy.py \
      --motion "$OUT_NPY" \
      --audio "$MUSIC" \
      --output "$OUT_DIR/videos/${RUN_NAME}_${TAG}_fixed.mp4" \
      --camera_mode fixed \
      2>&1 | tee "$LOG_DIR/${RUN_NAME}_render_${TAG}_fixed.log" || true

    python render_from_npy.py \
      --motion "$ROOTLOCK_NPY" \
      --audio "$MUSIC" \
      --output "$OUT_DIR/videos/${RUN_NAME}_${TAG}_rootlock_follow.mp4" \
      --camera_mode follow \
      2>&1 | tee "$LOG_DIR/${RUN_NAME}_render_${TAG}_rootlock_follow.log" || true
  done
done

echo "========== DONE =========="
echo "Videos:"
ls -lh "$OUT_DIR/videos" || true
echo
echo "Metrics:"
cat "$METRICS" || true
echo
echo "Check logs:"
echo "  $LOG_DIR/${RUN_NAME}_train.log"
echo "  $OUT_DIR/videos/"
echo "  $METRICS"
