#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_TURN_EVENT_REFINER_TRAIN=1

TRAJ=${EDGE_TRAJ:-"0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"}
BASE=${EDGE_BASE_MOTION:-output/v13_functional_base/dhw4_expr_mobile_base.npy}
TARGET=${EDGE_TARGET_MOTION:-output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy}
PAIR=${EDGE_TURN_PAIR_JSONL:-output/metric_reports/turn_event_refiner_pairs.jsonl}
CKPT=${EDGE_TURN_REFINER_CKPT:-runs/turn_event_refiner/turn_event_refiner.pt}

mkdir -p runs/turn_event_refiner output/metric_reports

python tools/make_turn_event_pairs.py \
  --base "$BASE" \
  --target "$TARGET" \
  --trajectory "$TRAJ" \
  --out "$PAIR"

python tools/train_turn_aware_event_refiner.py \
  --pairs "$PAIR" \
  --out "$CKPT" \
  --epochs ${EDGE_TURN_REFINER_EPOCHS:-300} \
  --batch_size ${EDGE_TURN_REFINER_BATCH:-4} \
  --lr ${EDGE_TURN_REFINER_LR:-0.001}

python tools/apply_turn_aware_event_refiner.py \
  --checkpoint "$CKPT" \
  --base "$BASE" \
  --trajectory "$TRAJ" \
  --out output/v13_turn_event_hybrid/dhw4_turn_event_refined.npy

python tools/evaluate_functional_choreo_coupling.py \
  --trajectory "$TRAJ" \
  --motions "$BASE,$TARGET,output/v13_turn_event_hybrid/dhw4_turn_event_refined.npy" \
  --out output/metric_reports/dhw4_turn_event_refiner_eval.json

echo "✅ trained refiner: $CKPT"
echo "✅ refined motion: output/v13_turn_event_hybrid/dhw4_turn_event_refined.npy"
