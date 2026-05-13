#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_TURN_EVENT_MODEL_ADAPTER=1
export EDGE_TURN_EVENT_TRAJ_TOKEN=${EDGE_TURN_EVENT_TRAJ_TOKEN:-1}
export EDGE_TURN_EVENT_OUTPUT_ADAPTER=${EDGE_TURN_EVENT_OUTPUT_ADAPTER:-1}
export EDGE_TURN_EVENT_ADAPTER_CKPT=${EDGE_TURN_EVENT_ADAPTER_CKPT:-runs/turn_event_internal_adapter/turn_event_internal_adapter.pt}
export EDGE_TURN_EVENT_PRESERVE_ROOT_XZ=1
export EDGE_TURN_SUPPORT_LAG=${EDGE_TURN_SUPPORT_LAG:-8}
export EDGE_TURN_EXPR_LAG=${EDGE_TURN_EXPR_LAG:-4}
export EDGE_TURN_MIN_GAP=${EDGE_TURN_MIN_GAP:-18}
export EDGE_TURN_GATE_SIGMA=${EDGE_TURN_GATE_SIGMA:-5.0}
# Safe body gates for output adapter.
export EDGE_TURN_EVENT_GATE_CONTACTS=${EDGE_TURN_EVENT_GATE_CONTACTS:-0.0}
export EDGE_TURN_EVENT_GATE_ROOT_XZ=0.0
export EDGE_TURN_EVENT_GATE_ROOT_Y=0.0
export EDGE_TURN_EVENT_GATE_LOWER=${EDGE_TURN_EVENT_GATE_LOWER:-0.45}
export EDGE_TURN_EVENT_GATE_TORSO=${EDGE_TURN_EVENT_GATE_TORSO:-0.75}
export EDGE_TURN_EVENT_GATE_UPPER=${EDGE_TURN_EVENT_GATE_UPPER:-0.75}

if [ -f runs/train_support_textpose_rag/v12_support_gait_textpose_rag_v1/weights/train-12.pt ]; then
  CKPT=runs/train_support_textpose_rag/v12_support_gait_textpose_rag_v1/weights/train-12.pt
elif [ -f runs/train_advanced_traj_phase/v12_gait_fourier_sparse_adapter_v1/weights/train-20.pt ]; then
  CKPT=runs/train_advanced_traj_phase/v12_gait_fourier_sparse_adapter_v1/weights/train-20.pt
else
  CKPT=runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt
fi

OUT=${1:-output/v13_turn_event_hybrid/dhw4_model_internal_event_generated.npy}
mkdir -p "$(dirname "$OUT")" logs/turn_event_internal_adapter
python generate_controlled.py \
  --checkpoint "$CKPT" \
  --music test_music_bank/dunhuangwu2_20s.wav \
  --feature_type hybrid \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" \
  --out "$OUT" \
  --pose_space normalized \
  --sampler ddpm \
  --no_tto \
  2>&1 | tee logs/turn_event_internal_adapter/generate_model_internal_event.log

echo "✅ generated with model-internal turn-event adapter: $OUT"
