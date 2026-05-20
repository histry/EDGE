#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh"
fi

conda activate edge || conda activate /home/disk/lsm/conda_envs/edge || true

RUN_ID="${RUN_ID:-v3i_hfevent_support_day08}"
OUT_DATA="${OUT_DATA:-data/dunhuang_bvh/footwork_hfevent_v3i_u45}"
BASE_CKPT="${BASE_CKPT:-runs/train_nextgen/v3h_footwork_day07_v3h_only_full_e300/weights/train-300.pt}"
SEQ_LEN="${SEQ_LEN:-45}"
EPOCHS="${EPOCHS:-240}"
SAVE_INTERVAL="${SAVE_INTERVAL:-40}"
BATCH_SIZE="${BATCH_SIZE:-12}"
BATCH_SIZE_SMOKE="${BATCH_SIZE_SMOKE:-6}"
LR="${LR:-4e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"

mkdir -p logs runs/train_nextgen
LOG="logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "RUN_ID=$RUN_ID"
echo "OUT_DATA=$OUT_DATA"
echo "BASE_CKPT=$BASE_CKPT"
echo "EPOCHS=$EPOCHS SAVE_INTERVAL=$SAVE_INTERVAL"
echo "BATCH_SIZE=$BATCH_SIZE LR=$LR"
echo "LOG=$LOG"
echo "============================================================"

if [ ! -d "$OUT_DATA" ]; then
  echo "ERROR: OUT_DATA not found: $OUT_DATA"
  exit 2
fi

if [ ! -f "$BASE_CKPT" ]; then
  echo "ERROR: BASE_CKPT not found: $BASE_CKPT"
  exit 3
fi

cat "$OUT_DATA/footwork_summary.json" || true

python - <<PY
from pathlib import Path
p=Path("$OUT_DATA")
files=list(p.glob("*.pkl"))
print("PKL count:", len(files))
if len(files) < 80:
    raise SystemExit("ERROR: too few pkl files")
PY

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0

export EDGE_V3_BASE_LOSS_STABILITY="${EDGE_V3_BASE_LOSS_STABILITY:-1}"
export EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES="${EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES:-1}"
export EDGE_V3_LOSS_STABILITY="${EDGE_V3_LOSS_STABILITY:-1}"
export EDGE_V3_CAP_TOTAL_LOSS="${EDGE_V3_CAP_TOTAL_LOSS:-0}"

export EDGE_X0_RECON_LOSS="${EDGE_X0_RECON_LOSS:-0}"
export EDGE_X0_RECON_LOSS_WEIGHT="${EDGE_X0_RECON_LOSS_WEIGHT:-0.12}"
export EDGE_V3_TEMPORAL_WEIGHT="${EDGE_V3_TEMPORAL_WEIGHT:-0.0}"
export EDGE_V3_DCT_KEEP="${EDGE_V3_DCT_KEEP:-8}"
export EDGE_V3_TEMPORAL_FEATURES="${EDGE_V3_TEMPORAL_FEATURES:-no_contact}"

export EDGE_V3C_VISIBLE_FK=0
export EDGE_V3F_BODY_CENTERED=0
export EDGE_V3H_SUPPORT_CHAIN_LOSS=1
export EDGE_V3H_SUPPORT_CHAIN_WEIGHT="${EDGE_V3H_SUPPORT_CHAIN_WEIGHT:-0.035}"

export EDGE_V3_MOTION_ENERGY_LOSS_CAP="${EDGE_V3_MOTION_ENERGY_LOSS_CAP:-80}"
export EDGE_V3H_SAMPLE_LOSS_CAP="${EDGE_V3H_SAMPLE_LOSS_CAP:-8}"
export EDGE_V3H_TERM_CAP="${EDGE_V3H_TERM_CAP:-8}"
export EDGE_V3H_PHYS_CLIP="${EDGE_V3H_PHYS_CLIP:-6.0}"
export EDGE_V3H_FK_POS_CLIP="${EDGE_V3H_FK_POS_CLIP:-8.0}"
export EDGE_V3H_DIFF_CLAMP="${EDGE_V3H_DIFF_CLAMP:-8.0}"

export EDGE_V3H_ROOT_Y_VEL_WEIGHT="${EDGE_V3H_ROOT_Y_VEL_WEIGHT:-0.18}"
export EDGE_V3H_ROOT_XZ_VEL_WEIGHT="${EDGE_V3H_ROOT_XZ_VEL_WEIGHT:-0.018}"
export EDGE_V3H_ROOT_Y_ABS_WEIGHT="${EDGE_V3H_ROOT_Y_ABS_WEIGHT:-0.01}"

export EDGE_V3H_PELVIS_ABS_WEIGHT="${EDGE_V3H_PELVIS_ABS_WEIGHT:-0.01}"
export EDGE_V3H_HIPS_ABS_WEIGHT="${EDGE_V3H_HIPS_ABS_WEIGHT:-0.01}"

export EDGE_V3H_PELVIS_ROT_VEL_WEIGHT="${EDGE_V3H_PELVIS_ROT_VEL_WEIGHT:-0.05}"
export EDGE_V3H_HIPS_ROT_VEL_WEIGHT="${EDGE_V3H_HIPS_ROT_VEL_WEIGHT:-0.04}"
export EDGE_V3H_KNEES_ROT_VEL_WEIGHT="${EDGE_V3H_KNEES_ROT_VEL_WEIGHT:-0.03}"
export EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT="${EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT:-0.025}"

export EDGE_V3H_LOWER_FK_SPEED_WEIGHT="${EDGE_V3H_LOWER_FK_SPEED_WEIGHT:-0.35}"
export EDGE_V3H_LOWER_FK_ACTIVITY_WEIGHT="${EDGE_V3H_LOWER_FK_ACTIVITY_WEIGHT:-0.35}"
export EDGE_V3H_LOWER_FK_RANGE_WEIGHT="${EDGE_V3H_LOWER_FK_RANGE_WEIGHT:-0.25}"
export EDGE_V3H_FOOT_FK_SPEED_WEIGHT="${EDGE_V3H_FOOT_FK_SPEED_WEIGHT:-0.15}"
export EDGE_V3H_FOOT_HEIGHT_WEIGHT="${EDGE_V3H_FOOT_HEIGHT_WEIGHT:-0.08}"
export EDGE_V3H_FOOT_RANGE_WEIGHT="${EDGE_V3H_FOOT_RANGE_WEIGHT:-0.15}"
export EDGE_V3H_LOWER_FK_DCT_WEIGHT="${EDGE_V3H_LOWER_FK_DCT_WEIGHT:-0.005}"

export EDGE_V3H_CONTACT_RECON_WEIGHT="${EDGE_V3H_CONTACT_RECON_WEIGHT:-0.005}"
export EDGE_V3H_CONTACT_SWITCH_WEIGHT="${EDGE_V3H_CONTACT_SWITCH_WEIGHT:-0.01}"
export EDGE_V3H_CONTACT_FOOT_VEL_WEIGHT="${EDGE_V3H_CONTACT_FOOT_VEL_WEIGHT:-0.005}"
export EDGE_V3H_CONTACT_FOOT_HEIGHT_WEIGHT="${EDGE_V3H_CONTACT_FOOT_HEIGHT_WEIGHT:-0.005}"

echo "============================================================"
echo "Smoke training..."
echo "============================================================"

export EDGE_V3H_DEBUG=1

python train.py \
  --project runs/train_nextgen \
  --exp_name "${RUN_ID}_smoke_e5" \
  --data_path "$OUT_DATA" \
  --checkpoint "$BASE_CKPT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE_SMOKE" \
  --epochs 5 \
  --save_interval 5 \
  --learning_rate "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
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
  --root_lower_coupling_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --val_batches 2 \
  --train_num_workers 2 \
  --val_num_workers 1

echo "============================================================"
echo "Full training..."
echo "============================================================"

export EDGE_V3H_DEBUG=0

python train.py \
  --project runs/train_nextgen \
  --exp_name "${RUN_ID}_full_e${EPOCHS}" \
  --data_path "$OUT_DATA" \
  --checkpoint "$BASE_CKPT" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len "$SEQ_LEN" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --save_interval "$SAVE_INTERVAL" \
  --learning_rate "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
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
  --root_lower_coupling_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --val_batches 4 \
  --train_num_workers 4 \
  --val_num_workers 2

echo "============================================================"
echo "Done."
find runs/train_nextgen -path "*${RUN_ID}_full_e${EPOCHS}*/weights/train-*.pt" | sort
echo "============================================================"
