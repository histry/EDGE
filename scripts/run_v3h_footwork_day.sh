#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

# Conda activate, robust fallback.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/disk/lsm/miniconda3/etc/profile.d/conda.sh"
fi

conda activate edge || conda activate /home/disk/lsm/conda_envs/edge || true

mkdir -p logs data/dunhuang_bvh/footwork_v3h_u45 runs/train_nextgen

RUN_ID="${RUN_ID:-v3h_footwork_day_$(date +%Y%m%d_%H%M%S)}"
LOG="logs/${RUN_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================================"
echo "RUN_ID=$RUN_ID"
echo "PWD=$(pwd)"
echo "DATE=$(date)"
echo "LOG=$LOG"
echo "============================================================"

# -----------------------------
# 0. Candidate DB selection
# -----------------------------
DB="${DB:-}"
if [ -z "$DB" ]; then
  for cand in \
    data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
    data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
    data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
    data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
    data/dunhuang_choreo_unit_rag/index_u45_s15_e10.npz
  do
    if [ -f "$cand" ]; then
      DB="$cand"
      break
    fi
  done
fi

if [ -z "$DB" ] || [ ! -f "$DB" ]; then
  echo "ERROR: no ChoreoRAG DB found. Set DB=/path/to/index.npz"
  exit 2
fi

echo "Using DB=$DB"

# If v12 footstep DB exists and v13 functional DB is missing, build it.
if [ ! -f data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz ] && \
   [ -f data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz ]; then
  echo "Building functional DB from v12 footstep DB..."
  python build_functional_choreo_rag_db.py \
    --in_db data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
    --out data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz
  DB="data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz"
fi

# -----------------------------
# 1. Export footwork-aware dataset
# -----------------------------
OUT_DATA="${OUT_DATA:-data/dunhuang_bvh/footwork_v3h_u45}"
SEQ_LEN="${SEQ_LEN:-45}"
PER_BUCKET="${PER_BUCKET:-120}"
MAX_TOTAL="${MAX_TOTAL:-500}"

echo "Exporting footwork-aware units..."
python scripts/export_footwork_aware_units.py \
  --db "$DB" \
  --out_dir "$OUT_DATA" \
  --seq_len "$SEQ_LEN" \
  --per_bucket "$PER_BUCKET" \
  --max_total "$MAX_TOTAL"

echo "Dataset summary:"
cat "$OUT_DATA/footwork_summary.json"

# Quick dataset count.
python - <<PY
from pathlib import Path
p=Path("$OUT_DATA")
files=list(p.glob("*.pkl"))
print("PKL count:", len(files))
if len(files) < 80:
    print("WARNING: selected units < 80; V3H may still be data-limited.")
PY

# -----------------------------
# 2. Base checkpoint
# -----------------------------
BASE_CKPT="${BASE_CKPT:-runs/train_nextgen/stationary_v3b_whitelist24_from_v16_x0w04_noDCT_energy/weights/train-300.pt}"

CKPT_ARGS=()
if [ -f "$BASE_CKPT" ]; then
  CKPT_ARGS=(--checkpoint "$BASE_CKPT")
  echo "Using BASE_CKPT=$BASE_CKPT"
else
  echo "WARNING: BASE_CKPT not found: $BASE_CKPT"
  echo "Training will start from random initialization unless you set BASE_CKPT=..."
fi

# -----------------------------
# 3. Shared V3H env
# -----------------------------
export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0

# Main V3 unit reconstruction settings.
export EDGE_X0_RECON_LOSS="${EDGE_X0_RECON_LOSS:-1}"
export EDGE_X0_RECON_LOSS_WEIGHT="${EDGE_X0_RECON_LOSS_WEIGHT:-0.30}"
export EDGE_V3_TEMPORAL_WEIGHT="${EDGE_V3_TEMPORAL_WEIGHT:-0.08}"
export EDGE_V3_DCT_KEEP="${EDGE_V3_DCT_KEEP:-8}"
export EDGE_V3_TEMPORAL_FEATURES="${EDGE_V3_TEMPORAL_FEATURES:-no_contact}"

# Isolate lower/support-chain first.
export EDGE_V3C_VISIBLE_FK=0
export EDGE_V3F_BODY_CENTERED=0
export EDGE_V3H_SUPPORT_CHAIN_LOSS=1
export EDGE_V3H_SUPPORT_CHAIN_WEIGHT="${EDGE_V3H_SUPPORT_CHAIN_WEIGHT:-4.0}"

# V3H support-chain weights.
export EDGE_V3H_ROOT_Y_VEL_WEIGHT="${EDGE_V3H_ROOT_Y_VEL_WEIGHT:-0.80}"
export EDGE_V3H_ROOT_XZ_VEL_WEIGHT="${EDGE_V3H_ROOT_XZ_VEL_WEIGHT:-0.35}"
export EDGE_V3H_ROOT_Y_ABS_WEIGHT="${EDGE_V3H_ROOT_Y_ABS_WEIGHT:-0.05}"

export EDGE_V3H_PELVIS_ROT_VEL_WEIGHT="${EDGE_V3H_PELVIS_ROT_VEL_WEIGHT:-0.35}"
export EDGE_V3H_HIPS_ROT_VEL_WEIGHT="${EDGE_V3H_HIPS_ROT_VEL_WEIGHT:-0.30}"
export EDGE_V3H_KNEES_ROT_VEL_WEIGHT="${EDGE_V3H_KNEES_ROT_VEL_WEIGHT:-0.24}"
export EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT="${EDGE_V3H_ANKLES_FEET_ROT_VEL_WEIGHT:-0.18}"

export EDGE_V3H_LOWER_ACTIVITY_FLOOR="${EDGE_V3H_LOWER_ACTIVITY_FLOOR:-0.65}"
export EDGE_V3H_LOWER_RANGE_FLOOR="${EDGE_V3H_LOWER_RANGE_FLOOR:-0.60}"
export EDGE_V3H_FOOT_RANGE_FLOOR="${EDGE_V3H_FOOT_RANGE_FLOOR:-0.55}"
export EDGE_V3H_LOWER_FK_DCT_WEIGHT="${EDGE_V3H_LOWER_FK_DCT_WEIGHT:-0.05}"
export EDGE_V3H_LOWER_FK_DCT_KEEP="${EDGE_V3H_LOWER_FK_DCT_KEEP:-8}"

# Training hyperparams.
BATCH_SIZE_SMOKE="${BATCH_SIZE_SMOKE:-16}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
EPOCHS="${EPOCHS:-900}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

echo "============================================================"
echo "ENV CHECK"
echo "EDGE_TRAIN_PROFILE=$EDGE_TRAIN_PROFILE"
echo "EDGE_V3H_SUPPORT_CHAIN_LOSS=$EDGE_V3H_SUPPORT_CHAIN_LOSS"
echo "EDGE_V3H_SUPPORT_CHAIN_WEIGHT=$EDGE_V3H_SUPPORT_CHAIN_WEIGHT"
echo "EDGE_X0_RECON_LOSS_WEIGHT=$EDGE_X0_RECON_LOSS_WEIGHT"
echo "EDGE_V3_TEMPORAL_WEIGHT=$EDGE_V3_TEMPORAL_WEIGHT"
echo "BATCH_SIZE=$BATCH_SIZE EPOCHS=$EPOCHS SAVE_INTERVAL=$SAVE_INTERVAL LR=$LR"
echo "============================================================"

# -----------------------------
# 4. Smoke training
# -----------------------------
echo "Starting V3H footwork smoke..."
export EDGE_V3H_DEBUG=1

python train.py \
  --project runs/train_nextgen \
  --exp_name "${RUN_ID}_smoke_e5" \
  --data_path "$OUT_DATA" \
  "${CKPT_ARGS[@]}" \
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

echo "Smoke done."

# -----------------------------
# 5. Full day training
# -----------------------------
echo "Starting V3H footwork full training..."
export EDGE_V3H_DEBUG=0

python train.py \
  --project runs/train_nextgen \
  --exp_name "${RUN_ID}_full_e${EPOCHS}" \
  --data_path "$OUT_DATA" \
  "${CKPT_ARGS[@]}" \
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
echo "V3H footwork day run finished."
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG"
echo "Latest checkpoints:"
find runs/train_nextgen -path "*${RUN_ID}_full_e${EPOCHS}*/weights/train-*.pt" | sort | tail -20
echo "============================================================"
