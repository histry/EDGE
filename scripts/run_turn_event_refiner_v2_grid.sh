#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

mkdir -p logs/stage_day/turn_event_refiner_v2_grid

for MAX_DELTA in 0.08 0.12 0.16; do
  for LR in 0.0005 0.0007; do
    TAG="md${MAX_DELTA}_lr${LR}"
    echo "===== run $TAG ====="
    export EDGE_TURN_REFINER_MAX_DELTA="$MAX_DELTA"
    export EDGE_TURN_REFINER_LR="$LR"
    export EDGE_TURN_REFINER_EPOCHS=${EDGE_TURN_REFINER_EPOCHS:-400}
    bash scripts/run_train_turn_event_refiner_v2.sh \
      2>&1 | tee "logs/stage_day/turn_event_refiner_v2_grid/${TAG}.log"
    cp runs/turn_event_refiner/turn_event_refiner_v2.pt "runs/turn_event_refiner/turn_event_refiner_v2_${TAG}.pt"
    cp output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2.npy "output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2_${TAG}.npy"
    cp output/metric_reports/dhw4_turn_event_refiner_v2_eval.json "output/metric_reports/dhw4_turn_event_refiner_v2_${TAG}_eval.json"
  done
done
