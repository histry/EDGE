#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

mkdir -p output/root_explosion_diagnosis logs/root_explosion_diagnosis output/metric_reports

export COMBINED_CKPT="runs/train_support_textpose_rag/v12_support_gait_textpose_rag_v1/weights/train-12.pt"

# ===== common model structure: keep support/trajectory/gait patches aligned =====
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1

# ===== common Text/Pose RAG settings =====
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_DROP_PROB=0.0
export EDGE_TEXT_CONTEXT_REQUIRED=0

# ===== body-gated RAG protection =====
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
export EDGE_TEXT_CONTEXT_INFER_SCALE=0.25

# ===== retrieved context units =====
export EDGE_RAG_CONTEXT_UNIT_PATHS="output/v12_footstep_stage1/dhw4_v12_main_mid01_f25_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid02_f50_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid03_f75_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid04_f100_unit.npy,output/v12_footstep_stage1/dhw4_v12_main_mid05_f125_unit.npy"

MID_POSES="output/v12_footstep_stage1/dhw4_v12_main_mid01_f25.npy,output/v12_footstep_stage1/dhw4_v12_main_mid02_f50.npy,output/v12_footstep_stage1/dhw4_v12_main_mid03_f75.npy,output/v12_footstep_stage1/dhw4_v12_main_mid04_f100.npy,output/v12_footstep_stage1/dhw4_v12_main_mid05_f125.npy"
MID_FRAMES="25,50,75,100,125"
TRAJ="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"

run_case () {
  local name="$1"
  local rag_mode="$2"
  local dcfg="$3"
  local no_tto="$4"

  echo "============================================================"
  echo "[CASE] $name"
  echo "RAG_MODE=$rag_mode  DCFG=$dcfg  NO_TTO=$no_tto"
  echo "============================================================"

  export EDGE_RAG_CONTEXT_MODE="$rag_mode"
  export EDGE_DYNAMIC_TRAJ_CFG="$dcfg"
  export EDGE_TRAJ_CFG_BASE=2.0
  export EDGE_TRAJ_CFG_SPEED_W=2.0
  export EDGE_TRAJ_CFG_CURVATURE_W=1.0
  export EDGE_TRAJ_CFG_MIN=1.0
  export EDGE_TRAJ_CFG_MAX=5.0
  export EDGE_RAG_CONTEXT_REPORT_JSON="output/root_explosion_diagnosis/${name}_context_report.json"

  EXTRA_ARGS=()
  if [ "$no_tto" = "1" ]; then
    EXTRA_ARGS+=(--no_tto)
  fi

  python generate_controlled.py \
    --checkpoint "$COMBINED_CKPT" \
    --music test_music_bank/dunhuangwu2_20s.wav \
    --feature_type hybrid \
    --start_pose test_keyframes/dyl002_600_1800_start.npy \
    --end_pose test_keyframes/dyl002_600_1800_end.npy \
    --mid_poses "$MID_POSES" \
    --mid_pose_frames "$MID_FRAMES" \
    --trajectory "$TRAJ" \
    --out "output/root_explosion_diagnosis/${name}.npy" \
    --pose_space normalized \
    --sampler ddpm \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "logs/root_explosion_diagnosis/${name}.log"
}

# 关键定位组：
# 1. RAG normal + dynamic CFG on/off
# 2. RAG no_context + dynamic CFG on/off
# 3. 上面各自再关 TTO，判断 raw/final 或 TTO 是否导致 root 爆炸

run_case "normal_dcfg1_tto_on"  "normal"     "1" "0"
run_case "normal_dcfg1_no_tto"  "normal"     "1" "1"
run_case "normal_dcfg0_no_tto"  "normal"     "0" "1"

run_case "noctx_dcfg1_tto_on"   "no_context" "1" "0"
run_case "noctx_dcfg1_no_tto"   "no_context" "1" "1"
run_case "noctx_dcfg0_no_tto"   "no_context" "0" "1"

echo "============================================================"
echo "Diagnosis generation finished."
echo "Outputs: output/root_explosion_diagnosis"
echo "Logs: logs/root_explosion_diagnosis"
echo "============================================================"
