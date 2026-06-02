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

export EDGE_V16_VISUAL_FIRST=${EDGE_V16_VISUAL_FIRST:-1}
export EDGE_V16_USE_FK=${EDGE_V16_USE_FK:-1}
export EDGE_V16_FK_DEVICE=${EDGE_V16_FK_DEVICE:-auto}
export EDGE_V16_FK_BATCH=${EDGE_V16_FK_BATCH:-256}
export EDGE_V16_RENDER_TOP_K=${EDGE_V16_RENDER_TOP_K:-12}
export EDGE_V16_RUN_TRAIN=${EDGE_V16_RUN_TRAIN:-1}

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="output/night_v16_visual_first_${DATE}"
LOG_ROOT="logs/night_v16_visual_first_${DATE}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V16 Visual-First Prior Selection Overnight"
echo "DATE=$DATE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "EDGE_V16_USE_FK=$EDGE_V16_USE_FK"
echo "EDGE_V16_RUN_TRAIN=$EDGE_V16_RUN_TRAIN"
echo "============================================================"

echo "[1/9] Select source NPZ..."

SRC_NPZ=""
for p in \
  data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
  data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
  output/night_v15_onset_phrase_20260526_001018/physics_aware_prior_pool.npz
do
  if [ -f "$p" ]; then
    SRC_NPZ="$p"
    break
  fi
done

if [ -z "$SRC_NPZ" ]; then
  echo "ERROR: no source NPZ found."
  exit 1
fi

echo "SRC_NPZ=$SRC_NPZ"

echo "[2/9] Build visual-first prior pool..."

python tools/build_visual_first_prior_pool.py \
  --npz "$SRC_NPZ" \
  --out_dir "$RUN_ROOT" \
  --target_len 45 \
  --top_k 1200 \
  --render_top_k "$EDGE_V16_RENDER_TOP_K" \
  --use_fk "$EDGE_V16_USE_FK" \
  --fk_device "$EDGE_V16_FK_DEVICE" \
  --fk_batch_size "$EDGE_V16_FK_BATCH" \
  --max_root_radius 0.25 \
  --max_rot_jump_p95 1.10 \
  --min_activity 0.030 \
  --min_upper_activity 0.020 \
  --min_torso_activity 0.008 \
  --audio_dim 803 \
  2>&1 | tee "$LOG_ROOT/build_visual_pool.log"

REPORT="$RUN_ROOT/visual_first_pool_report.json"
DATA_PATH="$RUN_ROOT/dunhuang_visual_first_pkl"

PKL_COUNT=$(find "$DATA_PATH" -name "*.pkl" | wc -l | tr -d ' ')
echo "PKL_COUNT=$PKL_COUNT"

if [ "$PKL_COUNT" -lt 30 ]; then
  echo "ERROR: too few selected visual units."
  exit 2
fi

echo "[3/9] Render top visual units..."

VIS_DIR="$RUN_ROOT/top_visual_renders"
mkdir -p "$VIS_DIR"

count=0
for npy in "$RUN_ROOT"/top_visual_unit_npy/*.npy; do
  [ -f "$npy" ] || continue
  count=$((count + 1))
  stem=$(basename "$npy" .npy)

  python render_from_npy.py \
    --motion "$npy" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$VIS_DIR/${stem}_follow.mp4" \
    --camera_mode follow || true

  if [ "$count" -ge "$EDGE_V16_RENDER_TOP_K" ]; then
    break
  fi
done

echo "[4/9] Build visual-first scheduled 150-frame demo..."

SCHED_DIR="$RUN_ROOT/scheduled_demos"
mkdir -p "$SCHED_DIR"

python tools/schedule_visual_first_phrase.py \
  --report "$REPORT" \
  --out "$SCHED_DIR/dw2_visual_first_schedule_balanced.npy" \
  --num_frames 150 \
  --starts 0,35,74,108 \
  --candidate_top_k 160 \
  --transition_weight 0.20 \
  --activity_weight 0.15 \
  --visual_weight 1.0 \
  --blend_radius 6 \
  2>&1 | tee "$LOG_ROOT/schedule_balanced.log"

python tools/schedule_visual_first_phrase.py \
  --report "$REPORT" \
  --out "$SCHED_DIR/dw2_visual_first_schedule_visual_heavy.npy" \
  --num_frames 150 \
  --starts 0,35,74,108 \
  --candidate_top_k 80 \
  --transition_weight 0.08 \
  --activity_weight 0.10 \
  --visual_weight 1.3 \
  --blend_radius 4 \
  2>&1 | tee "$LOG_ROOT/schedule_visual_heavy.log"

echo "[5/9] Render scheduled demos..."

for f in "$SCHED_DIR"/*.npy; do
  stem=$(basename "$f" .npy)
  python render_from_npy.py \
    --motion "$f" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$SCHED_DIR/${stem}_follow.mp4" \
    --camera_mode follow || true

  python render_from_npy.py \
    --motion "$f" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$SCHED_DIR/${stem}_fixed.mp4" \
    --camera_mode fixed || true
done

echo "[6/9] Decide whether to train..."

if [ "$EDGE_V16_RUN_TRAIN" != "1" ]; then
  echo "EDGE_V16_RUN_TRAIN != 1, skip training."
else
  echo "[6A/9] Wait for shared GPU..."

  GPU_ID=${GPU_ID:-0}
  MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-18500}
  CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-300}
  MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-14}

  START_TS=$(date +%s)
  MAX_WAIT_SEC=$((MAX_WAIT_HOURS * 3600))

  while true; do
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))

    GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
    GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

    echo "[$(date '+%F %T')] GPU used=${GPU_USED_MB}MiB free=${GPU_FREE_MB}MiB elapsed=${ELAPSED}s"

    if [ "$GPU_FREE_MB" -ge "$MIN_GPU_FREE_MB" ]; then
      echo "GPU free enough."
      break
    fi

    if [ "$ELAPSED" -ge "$MAX_WAIT_SEC" ]; then
      echo "Timeout waiting GPU. Skip training but keep visual pool outputs."
      EDGE_V16_RUN_TRAIN=0
      break
    fi

    echo "GPU busy. Do not kill other users. Sleep ${CHECK_INTERVAL_SEC}s..."
    sleep "$CHECK_INTERVAL_SEC"
  done
fi

if [ "$EDGE_V16_RUN_TRAIN" = "1" ]; then
  sleep 10
  export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

  echo "[7/9] Train visual-first real-unit reconstruction..."

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

  EXP_NAME="v16_visual_first_realunit_recon_${DATE}"

  python train.py \
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

  EXP_DIR="runs/train_nextgen/$EXP_NAME"
else
  EXP_NAME="SKIPPED"
  EXP_DIR="SKIPPED"
fi

echo "[8/9] Write next morning conclusion..."

LAST_CKPT=""
if [ "$EXP_DIR" != "SKIPPED" ] && [ -d "$EXP_DIR/weights" ]; then
  LAST_CKPT=$(ls -1 "$EXP_DIR"/weights/train-*.pt 2>/dev/null | sort -V | tail -1 || true)
fi

BEST_VAL=""
if [ -f "$LOG_ROOT/train.log" ]; then
  BEST_VAL=$(grep -E "Validation \| Val Loss:" "$LOG_ROOT/train.log" | tail -20 | sort -t':' -k2,2n | head -1 || true)
fi

cat > "$RUN_ROOT/NEXT_MORNING_CONCLUSION.md" <<EOF
# V16 Visual-First Prior Selection Overnight Conclusion

## Run

- RUN_ROOT: $RUN_ROOT
- LOG_ROOT: $LOG_ROOT
- Source NPZ: $SRC_NPZ
- Selected PKL count: $PKL_COUNT
- Training exp: $EXP_NAME
- Training exp dir: $EXP_DIR
- Last checkpoint: $LAST_CKPT
- Best recent val line: $BEST_VAL

## Main Idea

This run implements Visual-First Prior Selection.

The search order is:

1. Visual tension first:
   - upper activity
   - torso activity
   - endpoint variance
   - FK extension / silhouette / three-bend proxy

2. Kinematic safety:
   - low root radius
   - bounded rotation jump
   - non-static motion

3. Onset phrase scheduling:
   - compose 150-frame demo from top visual units
   - render directly as prior-based demo

## Important outputs

### Pool report

$RUN_ROOT/visual_first_pool_report.json

### Top visual unit renders

$RUN_ROOT/top_visual_renders/

### Scheduled demos

$RUN_ROOT/scheduled_demos/

Recommended files to inspect first:

- scheduled_demos/dw2_visual_first_schedule_balanced_follow.mp4
- scheduled_demos/dw2_visual_first_schedule_visual_heavy_follow.mp4

### Training checkpoints

$EXP_DIR/weights/

## Tomorrow judgement

1. If top_visual_renders contain units that are visually stronger than ck260/SDEdit, the visual-first route is validated.
2. If scheduled demos look more Dunhuang-like than diffusion resampling, use scheduled prior render as group-meeting demo.
3. If reconstruction training also keeps upper/torso expressiveness, this checkpoint can become the next refiner candidate.
4. If scheduled demos have boundary artifacts, keep prior pool and improve scheduler rather than returning to full diffusion resampling.

## Current recommendation

Prioritize visual-first prior-based render over full diffusion sampling.
Use diffusion only as optional shallow local refiner.
EOF

echo "[9/9] Pack next-morning package..."

zip -j "$RUN_ROOT/next_morning_visual_first_package.zip" \
  "$RUN_ROOT/NEXT_MORNING_CONCLUSION.md" \
  "$RUN_ROOT/VISUAL_FIRST_POOL_SUMMARY.md" \
  "$RUN_ROOT/visual_first_pool_report.json" \
  "$RUN_ROOT/visual_first_scores.csv" \
  "$LOG_ROOT/master.log" \
  "$LOG_ROOT/build_visual_pool.log" \
  "$SCHED_DIR"/*.json \
  "$SCHED_DIR"/*.mp4 \
  "$VIS_DIR"/*.mp4 \
  2>/dev/null || true

echo "============================================================"
echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "EXP_NAME=$EXP_NAME"
echo "EXP_DIR=$EXP_DIR"
echo "NEXT_MORNING=$RUN_ROOT/NEXT_MORNING_CONCLUSION.md"
echo "PACKAGE=$RUN_ROOT/next_morning_visual_first_package.zip"
echo "============================================================"
