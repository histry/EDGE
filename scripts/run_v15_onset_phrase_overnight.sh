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

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="output/night_v15_onset_phrase_${DATE}"
LOG_ROOT="logs/night_v15_onset_phrase_${DATE}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V15 Onset-Physics Temporal-Phrase Overnight"
echo "DATE=$DATE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

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

export EDGE_ONSET_ALPHA_MUSIC=${EDGE_ONSET_ALPHA_MUSIC:-0.20}
export EDGE_ONSET_BETA_PHYSICS=${EDGE_ONSET_BETA_PHYSICS:-1.20}
export EDGE_ONSET_NOVELTY=${EDGE_ONSET_NOVELTY:-0.35}
export EDGE_ONSET_MAX_ROOT_Y_DELTA=${EDGE_ONSET_MAX_ROOT_Y_DELTA:-0.30}
export EDGE_ONSET_MAX_ROOT_SPEED_DELTA=${EDGE_ONSET_MAX_ROOT_SPEED_DELTA:-0.35}
export EDGE_ONSET_MAX_CONTACT_L1=${EDGE_ONSET_MAX_CONTACT_L1:-1.60}
export EDGE_ONSET_ROOT_DRIFT_KEEP=${EDGE_ONSET_ROOT_DRIFT_KEEP:-0.02}

echo "[1/7] Check patch files..."
ls -lh choreorag_physics_state.py generate_onset_phrase_prior.py tools/build_physics_aware_prior_pool.py

python -m py_compile \
  choreorag_physics_state.py \
  generate_onset_phrase_prior.py \
  tools/build_physics_aware_prior_pool.py

echo "[2/7] Select RAG DB..."
DB=""
for p in \
  data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
  data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz
do
  if [ -f "$p" ]; then
    DB="$p"
    break
  fi
done

if [ -z "$DB" ]; then
  echo "ERROR: no ChoreoRAG DB found."
  exit 1
fi

echo "Using DB=$DB"

echo "[3/7] Build physics-aware pool..."
POOL="$RUN_ROOT/physics_aware_prior_pool.npz"

# 兼容不同版本的 build 脚本参数：先尝试新版，再尝试旧版
if python tools/build_physics_aware_prior_pool.py --help 2>&1 | grep -q -- "--rag_npz"; then
  python tools/build_physics_aware_prior_pool.py \
    --rag_npz "$DB" \
    --out "$POOL" \
    2>&1 | tee "$LOG_ROOT/build_pool.log"
elif python tools/build_physics_aware_prior_pool.py --help 2>&1 | grep -q -- "--db"; then
  python tools/build_physics_aware_prior_pool.py \
    --db "$DB" \
    --out "$POOL" \
    2>&1 | tee "$LOG_ROOT/build_pool.log"
else
  python tools/build_physics_aware_prior_pool.py \
    --input "$DB" \
    --out "$POOL" \
    2>&1 | tee "$LOG_ROOT/build_pool.log"
fi

if [ ! -f "$POOL" ]; then
  echo "ERROR: pool was not created: $POOL"
  exit 1
fi

echo "POOL=$POOL"

echo "[4/7] Generate onset phrase priors with current CLI..."
COMBINED_PKL="$RUN_ROOT/train_pkl_all"
mkdir -p "$COMBINED_PKL"

MUSICS=(
  test_music_bank/dunhuangwu2.wav
  test_music_bank/dunhuangwu3.wav
  test_music_bank/dunhuangwu4.wav
)

for WAV in "${MUSICS[@]}"; do
  if [ ! -f "$WAV" ]; then
    echo "Skip missing music: $WAV"
    continue
  fi

  TAG=$(basename "$WAV" .wav)
  CASE_DIR="$RUN_ROOT/$TAG"
  mkdir -p "$CASE_DIR"

  echo "---- Generate prior for $WAV ----"

  python generate_onset_phrase_prior.py \
    --pool "$POOL" \
    --audio_wav "$WAV" \
    --out "$CASE_DIR/${TAG}_v15_onset_phrase_prior.npy" \
    --length 150 \
        --max_phrases 4 \
    --fps 30 \
    --transition_tolerance 18 \
    --min_blend 5 \
    --max_blend 18 \
    --min_tail_context 8 \
    --alpha_music "${EDGE_ONSET_ALPHA_MUSIC}" \
    --beta_physics "${EDGE_ONSET_BETA_PHYSICS}" \
    --novelty_weight "${EDGE_ONSET_NOVELTY}" \
    --root_drift_keep "${EDGE_ONSET_ROOT_DRIFT_KEEP}" \
    --inplace \
    --export_pkl_dir "$CASE_DIR/train_pkl" \
    2>&1 | tee "$LOG_ROOT/${TAG}_prior.log"

  if compgen -G "$CASE_DIR/train_pkl/*.pkl" > /dev/null; then
    cp "$CASE_DIR/train_pkl/"*.pkl "$COMBINED_PKL/" || true
  fi
done

PKL_COUNT=$(find "$COMBINED_PKL" -name "*.pkl" | wc -l | tr -d ' ')
echo "Combined pkl count: $PKL_COUNT"

if [ "$PKL_COUNT" -lt 1 ]; then
  echo "ERROR: no pkl generated for training."
  exit 1
fi

echo "[5/7] Prior reports..."
find "$RUN_ROOT" -name "*.report.json" -maxdepth 3 -print -exec tail -n 80 {} \; || true

echo "[6/7] Start training..."
EXP_NAME="v15_onset_phrase_recon_${DATE}"

python train.py \
  --project runs/train_nextgen \
  --exp_name "$EXP_NAME" \
  --data_path "$COMBINED_PKL" \
  --processed_data_dir "$RUN_ROOT/dataset_cache" \
  --render_dir "$RUN_ROOT/renders" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --batch_size 1 \
  --epochs 1600 \
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
  --save_interval 50 \
  --val_batches 2 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --force_reload \
  --no_cache \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[7/7] DONE"
echo "EXP_NAME=$EXP_NAME"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "CHECKPOINT_DIR=runs/train_nextgen/$EXP_NAME/weights"
