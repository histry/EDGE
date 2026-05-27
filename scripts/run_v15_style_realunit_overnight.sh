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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="output/night_v15_style_realunit_${DATE}"
LOG_ROOT="logs/night_v15_style_realunit_${DATE}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V15 Style-RealUnit Overnight Training"
echo "DATE=$DATE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "============================================================"

echo "[1/6] Select source RAG / physics pool..."
SRC_NPZ=""
for p in \
  output/night_v15_onset_phrase_20260526_001018/physics_aware_prior_pool.npz \
  data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz
do
  if [ -f "$p" ]; then
    SRC_NPZ="$p"
    break
  fi
done

if [ -z "$SRC_NPZ" ]; then
  echo "ERROR: no source npz found."
  exit 1
fi

echo "SRC_NPZ=$SRC_NPZ"

echo "[2/6] Build visual/style-first real-unit trainset..."
python tools/build_v15_style_realunit_trainset.py \
  --npz "$SRC_NPZ" \
  --out_dir "$RUN_ROOT" \
  --target_len 45 \
  --top_k 1200 \
  --max_root_radius 0.20 \
  --max_rot_jump_p95 0.95 \
  --min_activity 0.035 \
  --min_upper_activity 0.025 \
  --min_torso_activity 0.010 \
  --audio_dim 803 \
  2>&1 | tee "$LOG_ROOT/build_style_trainset.log"

DATA_PATH="$RUN_ROOT/dunhuang_style_realunit_pkl"
PKL_COUNT=$(find "$DATA_PATH" -name "*.pkl" | wc -l | tr -d ' ')
echo "PKL_COUNT=$PKL_COUNT"

if [ "$PKL_COUNT" -lt 20 ]; then
  echo "ERROR: too few selected units. Loosen thresholds."
  exit 2
fi

echo "[3/6] Wait for shared RTX4090 GPU..."
GPU_ID=${GPU_ID:-0}
MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-19000}
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-300}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-12}

START_TS=$(date +%s)
MAX_WAIT_SEC=$((MAX_WAIT_HOURS * 3600))

while true; do
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))

  GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
  GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

  echo "[$(date '+%F %T')] GPU used=${GPU_USED_MB}MiB free=${GPU_FREE_MB}MiB elapsed=${ELAPSED}s"

  if [ "$GPU_FREE_MB" -ge "$MIN_GPU_FREE_MB" ]; then
    echo "✅ GPU memory enough."
    break
  fi

  if [ "$ELAPSED" -ge "$MAX_WAIT_SEC" ]; then
    echo "❌ timeout waiting GPU."
    exit 3
  fi

  echo "GPU busy. Do not kill other users. Sleep ${CHECK_INTERVAL_SEC}s..."
  sleep "$CHECK_INTERVAL_SEC"
done

sleep 10
GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
if [ "$GPU_FREE_MB" -lt "$MIN_GPU_FREE_MB" ]; then
  echo "GPU was occupied again. Exit safely."
  exit 4
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "[4/6] Setup training profile..."
export EDGE_DUNHUANG_STRICT_SPLIT=0
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_TRAJECTORY_PLANE=xz

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.55
export EDGE_V3_DCT_KEEP=8
export EDGE_V3_TEMPORAL_FEATURES=upper_torso
export EDGE_V3_VELOCITY_WEIGHT=0.04
export EDGE_V3_ACCEL_WEIGHT=0.008

EXP_NAME="v15_style_realunit_recon_${DATE}"

# Prefer a clean non-ck260 base checkpoint if available.
BASE_CKPT=""
for p in \
  runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt \
  runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt
do
  if [ -f "$p" ]; then
    BASE_CKPT="$p"
    break
  fi
done

EXTRA_CKPT_ARGS=()
if [ -n "$BASE_CKPT" ] && python train.py --help 2>&1 | grep -q -- "--checkpoint"; then
  EXTRA_CKPT_ARGS=(--checkpoint "$BASE_CKPT")
  echo "Using base checkpoint: $BASE_CKPT"
else
  echo "No compatible base checkpoint arg found; training without --checkpoint."
fi

echo "[5/6] Start overnight style-realunit reconstruction training..."
python train.py \
  "${EXTRA_CKPT_ARGS[@]}" \
  --project runs/train_nextgen \
  --exp_name "$EXP_NAME" \
  --data_path "$DATA_PATH" \
  --processed_data_dir "$RUN_ROOT/dataset_cache" \
  --render_dir "$RUN_ROOT/renders" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size 4 \
  --epochs 220 \
  --learning_rate 1.5e-4 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --cond_drop_prob 0.10 \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --keyframe_condition_prob 0.0 \
  --keyframe_loss_weight 0.0 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --beat_guidance_weight 0.0 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --contact_loss_weight 0.2 \
  --foot_loss_weight 0.5 \
  --traj_aug_prob 0.0 \
  --save_interval 20 \
  --val_batches 2 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --force_reload \
  --no_cache \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[6/6] DONE"
echo "EXP_NAME=$EXP_NAME"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CHECKPOINT_DIR=runs/train_nextgen/$EXP_NAME/weights"
