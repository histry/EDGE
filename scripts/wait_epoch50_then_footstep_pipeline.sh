#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-online}

mkdir -p logs output/v12_dhw4_base output/v12_footstep_stage1 output/v12_gait_phase_adapter

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/wait_epoch50_footstep_${TS}"
mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "Wait epoch50 then run v12 footstep pipeline"
echo "time=$TS"
echo "pwd=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "logs=$LOG_ROOT"
echo "============================================================"

# ===== strict split / XZ contract =====
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=0
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_REPORT_DIR=output/split_reports

# ===== wait for v12 epoch50 =====
export V12_EXP="runs/train_stage45/v12_no_leakage_xz_source_split"
export V12_CKPT="$V12_EXP/weights/train-50.pt"

echo "[Wait] Waiting for $V12_CKPT"
while [ ! -f "$V12_CKPT" ]; do
  echo "[$(date '+%F %T')] train-50.pt not ready yet..."
  sleep 300
done

echo "[$(date '+%F %T')] Found checkpoint: $V12_CKPT"

# ============================================================
# Stage 1A: rebuild v12 footstep-aware RAG DB
# ============================================================
export RAG_DB="data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz"
mkdir -p "$(dirname "$RAG_DB")"

echo "[Stage 1A] Build v12 footstep-aware RAG DB"
python -u build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out "$RAG_DB" \
  --checkpoint "$V12_CKPT" \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_model BAAI/bge-small-zh-v1.5 \
  --text_device cuda \
  2>&1 | tee "$LOG_ROOT/stage1a_build_v12_rag_db.log"

python - <<'PY' 2>&1 | tee "$LOG_ROOT/stage1a_check_rag_fields.log"
import numpy as np
p = "data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz"
d = np.load(p, allow_pickle=True)
keys = [
    "expressiveness_score",
    "locomotion_score",
    "footstep_score",
    "mobile_score",
    "root_speed_norm",
    "lower_activity_norm",
    "contact_switch",
    "alternating_foot_phase",
    "root_lower_sync",
]
print("RAG DB:", p)
for k in keys:
    print(k, k in d.files, d[k].shape if k in d.files else "")
PY

# ============================================================
# Stage 1B: regenerate v12 controlled base motion
# 使用你截图里的 test_keyframes / test_music_bank
# ============================================================
export OUT_DIR="output/v12_dhw4_base"
mkdir -p "$OUT_DIR"

export MUSIC="test_music_bank/dunhuangwu2_20s.wav"
export START_POSE="test_keyframes/dyl002_600_1800_start.npy"
export MID1_POSE="test_keyframes/dyl002_600_1800_mid1.npy"
export MID2_POSE="test_keyframes/dyl002_600_1800_mid2.npy"
export END_POSE="test_keyframes/dyl002_600_1800_end.npy"

# S 型 X/Z 地平面轨迹。需要更大位移可把 2.0 改成 3.0/4.0。
export TRAJ="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"

export BASE_OUT="$OUT_DIR/dhw4_v12.npy"

echo "[Stage 1B] Generate v12 base motion"
echo "  MUSIC=$MUSIC"
echo "  START=$START_POSE"
echo "  MID1=$MID1_POSE"
echo "  MID2=$MID2_POSE"
echo "  END=$END_POSE"
echo "  TRAJ=$TRAJ"
echo "  OUT=$BASE_OUT"

python -u generate_controlled.py \
  --checkpoint "$V12_CKPT" \
  --music "$MUSIC" \
  --feature_type hybrid \
  --start_pose "$START_POSE" \
  --end_pose "$END_POSE" \
  --mid_poses "$MID1_POSE,$MID2_POSE" \
  --mid_pose_frames "50,100" \
  --trajectory "$TRAJ" \
  --out "$BASE_OUT" \
  --pose_space normalized \
  --sampler ddpm \
  --save_motions \
  --no_render \
  2>&1 | tee "$LOG_ROOT/stage1b_generate_v12_base.log"

echo "[Stage 1B] Generated files:"
find "$OUT_DIR" -maxdepth 2 -type f \( -name "*.npy" -o -name "*.mp4" -o -name "*.json" \) | sort | tee "$LOG_ROOT/stage1b_generated_files.txt"

export BASE_MOTION="$BASE_OUT"

# 自动找 target trajectory
export TARGET_TRAJ="$(find "$OUT_DIR" -maxdepth 2 -type f -name "*target_traj*.npy" | head -n 1 || true)"

if [ ! -f "$BASE_MOTION" ]; then
  echo "ERROR: BASE_MOTION not found: $BASE_MOTION" >&2
  exit 1
fi

if [ -z "$TARGET_TRAJ" ] || [ ! -f "$TARGET_TRAJ" ]; then
  echo "ERROR: target trajectory .npy not found in $OUT_DIR" >&2
  echo "Please inspect $LOG_ROOT/stage1b_generated_files.txt" >&2
  exit 1
fi

echo "[Stage 1B] BASE_MOTION=$BASE_MOTION"
echo "[Stage 1B] TARGET_TRAJ=$TARGET_TRAJ"

# ============================================================
# Stage 1C: footstep-aware RAG + lower-body compositor
# ============================================================
export MID_FRAMES="25,50,75,100,125"
export EDGE_RAG_MOBILE_SPEED_THRESHOLD=0.010

run_compositor_variant () {
  local name="$1"
  local lower="$2"
  local torso="$3"
  local upper="$4"
  local window="$5"

  export OUT_PREFIX="output/v12_footstep_stage1/dhw4_v12_${name}"
  export EDGE_COMPOSITOR_LOWER_STRENGTH="$lower"
  export EDGE_COMPOSITOR_TORSO_STRENGTH="$torso"
  export EDGE_COMPOSITOR_UPPER_STRENGTH="$upper"
  export EDGE_COMPOSITOR_WINDOW="$window"

  echo "[Stage 1C][$name] lower=$lower torso=$torso upper=$upper window=$window"
  bash scripts/run_stage1_footstep_rag_compositor.sh \
    2>&1 | tee "$LOG_ROOT/stage1c_compositor_${name}.log"
}

run_compositor_variant "safe"   "0.60" "0.15" "0.00" "35"
run_compositor_variant "main"   "0.85" "0.25" "0.00" "45"
run_compositor_variant "strong" "1.00" "0.35" "0.10" "55"

# ============================================================
# Stage 2: gait phase adapter training
# ============================================================
echo "[Stage 2] Start gait phase adapter training"

export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_GAIT_CONTACT_LOSS=1
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=0.60

export EDGE_DIFF_CONTACT_LOSS=1
export EDGE_DCL_CONTACT_SOURCE=auto
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height

export CHECKPOINT="$V12_CKPT"
export PROJECT="runs/train_footstep_phase"
export EXP_NAME="v12_gait_phase_adapter_v1"
export BATCH_SIZE=4
export EPOCHS=20
export LR=1e-4

bash scripts/run_stage2_gait_phase_adapter_train.sh \
  2>&1 | tee "$LOG_ROOT/stage2_gait_phase_adapter.log"

echo "============================================================"
echo "Done."
echo "Logs: $LOG_ROOT"
echo "============================================================"
