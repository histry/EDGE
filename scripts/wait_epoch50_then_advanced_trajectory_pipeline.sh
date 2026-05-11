#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-online}

mkdir -p logs
mkdir -p output/v12_dhw4_base
mkdir -p output/v12_footstep_stage1
mkdir -p output/v12_decoupled
mkdir -p output/v12_stage4_bev

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/wait_epoch50_advanced_traj_${TS}"
mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "Wait Epoch 50 -> Advanced Trajectory ChoreoRAG Pipeline"
echo "time=$TS"
echo "pwd=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "logs=$LOG_ROOT"
echo "============================================================"

# ============================================================
# Global strict data / coordinate contract
# ============================================================
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=0
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_REPORT_DIR=output/split_reports

# ============================================================
# Wait for clean v12 baseline checkpoint
# ============================================================
export V12_EXP="runs/train_stage45/v12_no_leakage_xz_source_split"
export V12_CKPT="$V12_EXP/weights/train-50.pt"

echo "[WAIT] Waiting for checkpoint: $V12_CKPT"

while [ ! -f "$V12_CKPT" ]; do
  echo "[$(date '+%F %T')] train-50.pt not ready yet..."
  sleep 300
done

echo "[$(date '+%F %T')] Found checkpoint: $V12_CKPT"

# ============================================================
# Stage 1A: Build Footstep-aware Dynamic/Dual-score RAG DB
# ============================================================
export RAG_DB="data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz"
mkdir -p "$(dirname "$RAG_DB")"

echo "============================================================"
echo "[Stage 1A] Build v12 Footstep-aware RAG DB"
echo "RAG_DB=$RAG_DB"
echo "============================================================"

python -u build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out "$RAG_DB" \
  --checkpoint "$V12_CKPT" \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_model BAAI/bge-small-zh-v1.5 \
  --text_device cuda \
  2>&1 | tee "$LOG_ROOT/stage1a_build_v12_footstep_rag_db.log"

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
# Stage 1B: Regenerate v12 base motion
# ============================================================
export OUT_DIR="output/v12_dhw4_base"
mkdir -p "$OUT_DIR"

export MUSIC="test_music_bank/dunhuangwu2_20s.wav"
export START_POSE="test_keyframes/dyl002_600_1800_start.npy"
export MID1_POSE="test_keyframes/dyl002_600_1800_mid1.npy"
export MID2_POSE="test_keyframes/dyl002_600_1800_mid2.npy"
export END_POSE="test_keyframes/dyl002_600_1800_end.npy"

# X/Z ground-plane S trajectory
export TRAJ="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"
export BASE_OUT="$OUT_DIR/dhw4_v12.npy"

echo "============================================================"
echo "[Stage 1B] Generate v12 base motion"
echo "MUSIC=$MUSIC"
echo "START_POSE=$START_POSE"
echo "MID1_POSE=$MID1_POSE"
echo "MID2_POSE=$MID2_POSE"
echo "END_POSE=$END_POSE"
echo "TRAJ=$TRAJ"
echo "BASE_OUT=$BASE_OUT"
echo "============================================================"

# Try --out first. If current generate_controlled.py uses another arg, log will show it.
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
export TARGET_TRAJ="$(find "$OUT_DIR" -maxdepth 2 -type f -name "*target_traj*.npy" | head -n 1 || true)"

if [ ! -f "$BASE_MOTION" ]; then
  echo "ERROR: BASE_MOTION not found: $BASE_MOTION" >&2
  exit 1
fi

if [ -z "$TARGET_TRAJ" ] || [ ! -f "$TARGET_TRAJ" ]; then
  echo "ERROR: target trajectory .npy not found in $OUT_DIR" >&2
  echo "Please inspect: $LOG_ROOT/stage1b_generated_files.txt" >&2
  exit 1
fi

echo "[Stage 1B] BASE_MOTION=$BASE_MOTION"
echo "[Stage 1B] TARGET_TRAJ=$TARGET_TRAJ"

# ============================================================
# Stage 1C: Footstep-aware RAG + Segment Lower-body Compositor
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

  echo "============================================================"
  echo "[Stage 1C][$name] Lower-body Compositor"
  echo "OUT_PREFIX=$OUT_PREFIX"
  echo "lower=$lower torso=$torso upper=$upper window=$window"
  echo "============================================================"

  bash scripts/run_stage1_footstep_rag_compositor.sh \
    2>&1 | tee "$LOG_ROOT/stage1c_compositor_${name}.log"
}

run_compositor_variant "safe"   "0.60" "0.15" "0.00" "35"
run_compositor_variant "main"   "0.85" "0.25" "0.00" "45"
run_compositor_variant "strong" "1.00" "0.35" "0.10" "55"

# ============================================================
# Stage 2: Advanced trajectory + gait phase adapter training
# ============================================================
echo "============================================================"
echo "[Stage 2] Advanced Trajectory + Gait Phase Adapter Training"
echo "============================================================"

export CHECKPOINT="$V12_CKPT"
export PROJECT="runs/train_advanced_traj_phase"
export EXP_NAME="v12_gait_fourier_sparse_adapter_v1"
export BATCH_SIZE=4
export EPOCHS=20
export LR=1e-4

# Advanced trajectory condition
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_GAIT_CONTACT_LOSS=1
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=0.60

export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1
export EDGE_TRAJ_WAYPOINT_FRAMES=0,50,100,149

# Dynamic trajectory CFG for generation/eval hooks
export EDGE_DYNAMIC_TRAJ_CFG=1
export EDGE_TRAJ_CFG_BASE=2.0
export EDGE_TRAJ_CFG_SPEED_W=2.0
export EDGE_TRAJ_CFG_CURVATURE_W=1.0
export EDGE_TRAJ_CFG_MIN=1.0
export EDGE_TRAJ_CFG_MAX=5.0

# Contact losses
export EDGE_DIFF_CONTACT_LOSS=1
export EDGE_DCL_CONTACT_SOURCE=auto
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height

bash scripts/run_stage2_advanced_traj_adapter_train.sh \
  2>&1 | tee "$LOG_ROOT/stage2_advanced_traj_adapter.log"

# ============================================================
# Stage 3: Decoupled upper/lower deterministic merge
# ============================================================
echo "============================================================"
echo "[Stage 3] Decoupled upper/lower merge"
echo "============================================================"

export LOWER_MOTION="output/v12_footstep_stage1/dhw4_v12_main_composited.npy"
export UPPER_MOTION="$BASE_MOTION"
export OUT="output/v12_decoupled/dhw4_v12_decoupled_merge.npy"

if [ -f "$LOWER_MOTION" ] && [ -f "$UPPER_MOTION" ]; then
  bash scripts/run_stage3_decoupled_merge.sh \
    2>&1 | tee "$LOG_ROOT/stage3_decoupled_merge.log"
else
  echo "WARNING: skip Stage 3, missing LOWER_MOTION or UPPER_MOTION"
  echo "LOWER_MOTION=$LOWER_MOTION"
  echo "UPPER_MOTION=$UPPER_MOTION"
fi

# ============================================================
# Summary
# ============================================================
echo "============================================================"
echo "Advanced trajectory overnight pipeline finished."
echo "Logs: $LOG_ROOT"
echo "Key outputs:"
find output/v12_dhw4_base -type f | sort || true
find output/v12_footstep_stage1 -type f | sort || true
find output/v12_decoupled -type f | sort || true
find runs/train_advanced_traj_phase -type f -name "*.pt" | sort || true
echo "============================================================"
