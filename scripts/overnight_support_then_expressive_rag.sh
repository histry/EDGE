#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_MODE=${WANDB_MODE:-online}

mkdir -p logs
mkdir -p output/stage2_support_eval
mkdir -p output/stage2_support_eval_final
mkdir -p output/stage3_expressive_rag_eval
mkdir -p output/metric_reports

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="logs/overnight_support_expressive_${TS}"
mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "Overnight: Support-aware adapter -> Expressive Text/Pose RAG"
echo "time=$TS"
echo "pwd=$(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "logs=$LOG_ROOT"
echo "============================================================"

# ============================================================
# Global clean data / coordinate contract
# ============================================================
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=0
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_REPORT_DIR=output/split_reports

# ============================================================
# Paths
# ============================================================
export V12_CKPT="runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt"
export SUPPORT_EXP="runs/train_advanced_traj_phase/v12_gait_fourier_sparse_adapter_v1"
export SUPPORT_CKPT="$SUPPORT_EXP/weights/train-20.pt"

# 如果 train-20.pt 不存在，等待当前 Stage 2 训练完成。
echo "[WAIT] Waiting for support-aware trajectory/gait adapter checkpoint:"
echo "       $SUPPORT_CKPT"

while [ ! -f "$SUPPORT_CKPT" ]; do
  echo "[$(date '+%F %T')] support checkpoint not ready yet..."
  sleep 300
done

echo "[$(date '+%F %T')] Found support checkpoint: $SUPPORT_CKPT"
sleep 120

# ============================================================
# Common generation inputs
# ============================================================
export MUSIC="test_music_bank/dunhuangwu2_20s.wav"
export START_POSE="test_keyframes/dyl002_600_1800_start.npy"
export MID1_POSE="test_keyframes/dyl002_600_1800_mid1.npy"
export MID2_POSE="test_keyframes/dyl002_600_1800_mid2.npy"
export END_POSE="test_keyframes/dyl002_600_1800_end.npy"

# ============================================================
# Stage A: Evaluate support-aware trajectory/gait adapter
# ============================================================
echo "============================================================"
echo "[Stage A] Evaluate support-aware trajectory/gait adapter"
echo "============================================================"

# 开启 support / trajectory / gait 分支；关闭未训练的 Text/Pose RAG，先干净评估 support 能力。
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1
export EDGE_TRAJ_WAYPOINT_FRAMES=0,50,100,149
export EDGE_DYNAMIC_TRAJ_CFG=1
export EDGE_TRAJ_CFG_BASE=2.0
export EDGE_TRAJ_CFG_SPEED_W=2.0
export EDGE_TRAJ_CFG_CURVATURE_W=1.0
export EDGE_TRAJ_CFG_MIN=1.0
export EDGE_TRAJ_CFG_MAX=5.0

generate_eval () {
  local ckpt="$1"
  local out_prefix="$2"
  local traj="$3"
  local extra_name="$4"

  mkdir -p "$(dirname "$out_prefix")"

  echo "------------------------------------------------------------"
  echo "[GEN] ckpt=$ckpt"
  echo "[GEN] out=$out_prefix"
  echo "[GEN] name=$extra_name"
  echo "[GEN] traj=$traj"
  echo "------------------------------------------------------------"

  if [ "$traj" = "__NO_TRAJ__" ]; then
    python -u generate_controlled.py \
      --checkpoint "$ckpt" \
      --music "$MUSIC" \
      --feature_type hybrid \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --mid_poses "$MID1_POSE,$MID2_POSE" \
      --mid_pose_frames "50,100" \
      --out "$out_prefix.npy" \
      --pose_space normalized \
      --sampler ddpm
  else
    python -u generate_controlled.py \
      --checkpoint "$ckpt" \
      --music "$MUSIC" \
      --feature_type hybrid \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --mid_poses "$MID1_POSE,$MID2_POSE" \
      --mid_pose_frames "50,100" \
      --trajectory "$traj" \
      --out "$out_prefix.npy" \
      --pose_space normalized \
      --sampler ddpm
  fi
}

# Support adapter 评估四组：
generate_eval "$SUPPORT_CKPT" "output/stage2_support_eval/dhw4_support_no_traj" "__NO_TRAJ__" "no_traj" \
  2>&1 | tee "$LOG_ROOT/stageA_generate_no_traj.log"

generate_eval "$SUPPORT_CKPT" "output/stage2_support_eval/dhw4_support_static" "0,0;0,0" "static" \
  2>&1 | tee "$LOG_ROOT/stageA_generate_static.log"

generate_eval "$SUPPORT_CKPT" "output/stage2_support_eval/dhw4_support_small" "0,0;0.2,0.1;-0.2,0.15;0,0.05" "small" \
  2>&1 | tee "$LOG_ROOT/stageA_generate_small.log"

generate_eval "$SUPPORT_CKPT" "output/stage2_support_eval/dhw4_support_S" "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" "S" \
  2>&1 | tee "$LOG_ROOT/stageA_generate_S.log"

# 指标统计函数
cat > "$LOG_ROOT/metric_eval.py" <<'PY'
import json
import numpy as np
from pathlib import Path
import sys

ROT_START = 7
ROT_DIM = 6
LOWER = [1,2,4,5,7,8,10,11]
TORSO = [3,6,9]
UPPER = [12,13,14,15,16,17,18,19,20,21,22,23]

def idx(joints):
    out = []
    for j in joints:
        out += list(range(ROT_START + ROT_DIM*j, ROT_START + ROT_DIM*j + ROT_DIM))
    return out

LOWER_IDX = idx(LOWER)
TORSO_IDX = idx(TORSO)
UPPER_IDX = idx(UPPER)

def load_motion(p):
    x = np.load(p, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        d = x.item()
        x = d.get("motion", d.get("pose", x))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"{p}: expected [T,151], got {x.shape}")
    return x

def rms(x):
    return float(np.sqrt(np.mean((x[1:] - x[:-1]) ** 2))) if len(x) > 1 else 0.0

def one(path):
    m = load_motion(path)
    root_path = float(np.linalg.norm(m[1:, [4,6]] - m[:-1, [4,6]], axis=-1).sum())
    contact = (m[:, 0:4] > 0.5).astype(np.float32)
    contact_switch = float(np.abs(contact[1:] - contact[:-1]).mean()) if len(contact) > 1 else 0.0
    return {
        "frames": int(len(m)),
        "lower": rms(m[:, LOWER_IDX]),
        "torso": rms(m[:, TORSO_IDX]),
        "upper": rms(m[:, UPPER_IDX]),
        "root_path": root_path,
        "contact_mean": float(contact.mean()),
        "contact_switch": contact_switch,
    }

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = {}

for p in sorted(root.glob("*.npy")):
    if "target_traj" in p.name:
        continue
    if p.name.endswith("_raw.npy"):
        continue
    try:
        rows[p.name] = one(p)
    except Exception as e:
        rows[p.name] = {"error": str(e)}

print(json.dumps(rows, indent=2, ensure_ascii=False))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved:", out)
PY

python "$LOG_ROOT/metric_eval.py" \
  output/stage2_support_eval \
  output/metric_reports/stageA_support_eval_metrics.json \
  2>&1 | tee "$LOG_ROOT/stageA_metrics.log"

# ============================================================
# Stage B: Expressive whole-body / Text-Pose RAG adapter training
# ============================================================
echo "============================================================"
echo "[Stage B] Train Expressive whole-body / Text-Pose RAG adapter"
echo "============================================================"

# 基于 support checkpoint 继续训，不从 v12 干净 baseline 重新开始。
# 保留 support/gait 条件，同时开启 Text/Pose RAG 分支。
export CHECKPOINT="$SUPPORT_CKPT"
export PROJECT="runs/train_support_textpose_rag"
export EXP_NAME="v12_support_gait_textpose_rag_v1"
export BATCH_SIZE=4
export EPOCHS=12
export LR=8e-5

export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1
export EDGE_TRAJ_WAYPOINT_FRAMES=0,50,100,149
export EDGE_DYNAMIC_TRAJ_CFG=1

export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_DIM=512
export EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS=64
export EDGE_RAG_CONTEXT_MAX_LEN=45
export EDGE_TEXT_CONTEXT_DROP_PROB=0.10

# 表达能力训练：不要一开始大幅改 decoder，先 adapter 训练。
python -u train.py \
  --project "$PROJECT" \
  --exp_name "$EXP_NAME" \
  --data_path data/dunhuang_bvh/processed \
  --checkpoint "$CHECKPOINT" \
  --train_stage adapter \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --save_interval 4 \
  --val_batches 10 \
  --learning_rate "$LR" \
  --trajectory_loss_weight 0.8 \
  --trajectory_velocity_loss_weight 0.25 \
  --foot_loss_weight 0.3 \
  --contact_loss_weight 0.3 \
  --sync_loss_weight 1.0 \
  2>&1 | tee "$LOG_ROOT/stageB_textpose_rag_adapter_train.log"

export COMBINED_CKPT="$PROJECT/$EXP_NAME/weights/train-12.pt"
if [ ! -f "$COMBINED_CKPT" ]; then
  export COMBINED_CKPT="$(find "$PROJECT/$EXP_NAME/weights" -type f -name 'train-*.pt' | sort -V | tail -n 1)"
fi

echo "[Stage B] Combined checkpoint: $COMBINED_CKPT"

# ============================================================
# Stage C: Evaluate combined support + expressive RAG checkpoint
# ============================================================
echo "============================================================"
echo "[Stage C] Evaluate combined checkpoint"
echo "============================================================"

export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_DROP_PROB=0.0

generate_eval "$COMBINED_CKPT" "output/stage3_expressive_rag_eval/dhw4_combined_no_traj" "__NO_TRAJ__" "combined_no_traj" \
  2>&1 | tee "$LOG_ROOT/stageC_generate_no_traj.log"

generate_eval "$COMBINED_CKPT" "output/stage3_expressive_rag_eval/dhw4_combined_static" "0,0;0,0" "combined_static" \
  2>&1 | tee "$LOG_ROOT/stageC_generate_static.log"

generate_eval "$COMBINED_CKPT" "output/stage3_expressive_rag_eval/dhw4_combined_small" "0,0;0.2,0.1;-0.2,0.15;0,0.05" "combined_small" \
  2>&1 | tee "$LOG_ROOT/stageC_generate_small.log"

generate_eval "$COMBINED_CKPT" "output/stage3_expressive_rag_eval/dhw4_combined_S" "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" "combined_S" \
  2>&1 | tee "$LOG_ROOT/stageC_generate_S.log"

python "$LOG_ROOT/metric_eval.py" \
  output/stage3_expressive_rag_eval \
  output/metric_reports/stageC_combined_eval_metrics.json \
  2>&1 | tee "$LOG_ROOT/stageC_metrics.log"

echo "============================================================"
echo "Overnight support + expressive RAG pipeline finished."
echo "Logs: $LOG_ROOT"
echo "Key metric reports:"
echo "  output/metric_reports/stageA_support_eval_metrics.json"
echo "  output/metric_reports/stageC_combined_eval_metrics.json"
echo "Key outputs:"
find output/stage2_support_eval -type f | sort || true
find output/stage3_expressive_rag_eval -type f | sort || true
find runs/train_support_textpose_rag -type f -name "*.pt" | sort -V || true
echo "============================================================"
