#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p output/bounded_cfg_root_gate_grid logs/bounded_cfg_root_gate_grid output/metric_reports

export COMBINED_CKPT="runs/train_support_textpose_rag/v12_support_gait_textpose_rag_v1/weights/train-12.pt"

# ===== trajectory / gait structure =====
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1

# ===== bounded dynamic CFG =====
export EDGE_DYNAMIC_TRAJ_CFG=1
export EDGE_TRAJ_CFG_LEGACY=0
export EDGE_TRAJ_CFG_FEATURE_GATE=1
export EDGE_TRAJ_CFG_EFFECT_SCALE=0.25
export EDGE_TRAJ_CFG_MAX_GAIN=1.0
export EDGE_TRAJ_CFG_ROOT_STEP_CLAMP=0.45

# 仍然用旧高权重测试，因为新版 bounded CFG 应该能兜住
export EDGE_TRAJ_CFG_BASE=2.0
export EDGE_TRAJ_CFG_SPEED_W=2.0
export EDGE_TRAJ_CFG_CURVATURE_W=1.0
export EDGE_TRAJ_CFG_MIN=1.0
export EDGE_TRAJ_CFG_MAX=5.0

# 非 root 部位 CFG gate
export EDGE_TRAJ_CFG_GATE_CONTACTS=0.0
export EDGE_TRAJ_CFG_GATE_ROOT_Y=0.0
export EDGE_TRAJ_CFG_GATE_PELVIS_ROT=0.10
export EDGE_TRAJ_CFG_GATE_LOWER=0.35
export EDGE_TRAJ_CFG_GATE_TORSO=0.50
export EDGE_TRAJ_CFG_GATE_UPPER=0.50

# ===== Text/Pose Context RAG =====
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_DROP_PROB=0.0
export EDGE_RAG_CONTEXT_MODE=normal
export EDGE_TEXT_CONTEXT_REQUIRED=1

export EDGE_TEXT_CONTEXT_INFER_SCALE=0.25
export EDGE_TEXT_CONTEXT_BODY_GATE=1
export EDGE_TEXT_CONTEXT_BODY_DELTA_SCALE=1.0
export EDGE_TEXT_CONTEXT_PRESERVE_ROOT_XZ=1

export EDGE_TEXT_CONTEXT_GATE_CONTACTS=0.0
export EDGE_TEXT_CONTEXT_GATE_ROOT_XZ=0.0
export EDGE_TEXT_CONTEXT_GATE_ROOT_Y=0.0
export EDGE_TEXT_CONTEXT_GATE_PELVIS_ROT=0.15
export EDGE_TEXT_CONTEXT_GATE_LOWER=0.25
export EDGE_TEXT_CONTEXT_GATE_TORSO=0.75
export EDGE_TEXT_CONTEXT_GATE_UPPER=1.0

export EDGE_RAG_CONTEXT_UNIT_PATHS="output/v12_footstep_stage1/dhw4_v12_main_mid01_f25_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid02_f50_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid03_f75_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid04_f100_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid05_f125_unit.npy"

MID_POSES="output/v12_footstep_stage1/dhw4_v12_main_mid01_f25.npy,output/v12_footstep_stage1/dhw4_v12_main_mid02_f50.npy,output/v12_footstep_stage1/dhw4_v12_main_mid03_f75.npy,output/v12_footstep_stage1/dhw4_v12_main_mid04_f100.npy,output/v12_footstep_stage1/dhw4_v12_main_mid05_f125.npy"
MID_FRAMES="25,50,75,100,125"
TRAJ="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"

# 关键：逐步释放 root X/Z 的 CFG gate
# preserve_root_xz=0 后，root_path 才有机会从 8.5 回到 17；
# root_step_clamp=0.45 防止重新爆炸。
for ROOT_GATE in 0.05 0.10 0.20 0.35 0.50 0.75 1.00; do
  export EDGE_TRAJ_CFG_PRESERVE_ROOT_XZ=0
  export EDGE_TRAJ_CFG_GATE_ROOT_XZ="$ROOT_GATE"

  TAG="rootgate${ROOT_GATE}"
  export EDGE_RAG_CONTEXT_REPORT_JSON="output/bounded_cfg_root_gate_grid/dhw4_${TAG}_context_report.json"

  echo "============================================================"
  echo "[RUN] $TAG"
  echo "EDGE_TRAJ_CFG_GATE_ROOT_XZ=$EDGE_TRAJ_CFG_GATE_ROOT_XZ"
  echo "EDGE_TRAJ_CFG_ROOT_STEP_CLAMP=$EDGE_TRAJ_CFG_ROOT_STEP_CLAMP"
  echo "============================================================"

  python generate_controlled.py \
    --checkpoint "$COMBINED_CKPT" \
    --music test_music_bank/dunhuangwu2_20s.wav \
    --feature_type hybrid \
    --start_pose test_keyframes/dyl002_600_1800_start.npy \
    --end_pose test_keyframes/dyl002_600_1800_end.npy \
    --mid_poses "$MID_POSES" \
    --mid_pose_frames "$MID_FRAMES" \
    --trajectory "$TRAJ" \
    --out "output/bounded_cfg_root_gate_grid/dhw4_${TAG}.npy" \
    --pose_space normalized \
    --sampler ddpm \
    --no_tto \
    2>&1 | tee "logs/bounded_cfg_root_gate_grid/dhw4_${TAG}.log"
done

echo "✅ bounded CFG root-gate grid finished."
