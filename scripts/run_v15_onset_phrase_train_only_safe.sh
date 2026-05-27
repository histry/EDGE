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

RUN_ROOT=output/night_v15_onset_phrase_20260526_001018
DATA_PATH="$RUN_ROOT/dunhuang_train_pkl_all"

DATE=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/night_v15_onset_phrase_safe_train_${DATE}"
mkdir -p "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V15 Onset Phrase SAFE Train Only"
echo "DATE=$DATE"
echo "DATA_PATH=$DATA_PATH"
echo "LOG_ROOT=$LOG_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

PKL_COUNT=$(find "$DATA_PATH" -name "*.pkl" | wc -l | tr -d ' ')
echo "PKL_COUNT=$PKL_COUNT"

if [ "$PKL_COUNT" -lt 1 ]; then
  echo "ERROR: no pkl files in $DATA_PATH"
  exit 1
fi

export EDGE_DUNHUANG_STRICT_SPLIT=0
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_TRAJECTORY_PLANE=xz

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.45
export EDGE_V3_DCT_KEEP=8
export EDGE_V3_TEMPORAL_FEATURES=upper_torso
export EDGE_V3_VELOCITY_WEIGHT=0.06
export EDGE_V3_ACCEL_WEIGHT=0.012

EXP_NAME="v15_onset_phrase_safe_recon_si20_${DATE}"

python train.py \
  --project runs/train_nextgen \
  --exp_name "$EXP_NAME" \
  --data_path "$DATA_PATH" \
  --processed_data_dir "$RUN_ROOT/dataset_cache_safe_${DATE}" \
  --render_dir "$RUN_ROOT/renders_safe_${DATE}" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --batch_size 1 \
  --epochs 300 \
  --learning_rate 2e-4 \
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
  --val_batches 1 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --force_reload \
  --no_cache \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "DONE"
echo "EXP_NAME=$EXP_NAME"
echo "LOG_ROOT=$LOG_ROOT"
echo "CHECKPOINT_DIR=runs/train_nextgen/$EXP_NAME/weights"
