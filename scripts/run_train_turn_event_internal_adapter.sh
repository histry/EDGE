#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_DYNAMIC_TRAJ_CFG=${EDGE_DYNAMIC_TRAJ_CFG:-0}
export EDGE_TURN_INTERNAL_EPOCHS=${EDGE_TURN_INTERNAL_EPOCHS:-600}
export EDGE_TURN_INTERNAL_LR=${EDGE_TURN_INTERNAL_LR:-0.0007}
export EDGE_TURN_INTERNAL_MAX_DELTA=${EDGE_TURN_INTERNAL_MAX_DELTA:-0.14}
mkdir -p logs/turn_event_internal_adapter runs/turn_event_internal_adapter output/v13_turn_event_hybrid output/metric_reports

python tools/train_turn_event_internal_adapter.py \
  --base output/v13_functional_base/dhw4_expr_mobile_base.npy \
  --anchor output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy \
  --trajectory "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" \
  --out runs/turn_event_internal_adapter/turn_event_internal_adapter.pt \
  --epochs "$EDGE_TURN_INTERNAL_EPOCHS" \
  --lr "$EDGE_TURN_INTERNAL_LR" \
  --max_delta "$EDGE_TURN_INTERNAL_MAX_DELTA" \
  2>&1 | tee logs/turn_event_internal_adapter/train_internal_adapter.log

python tools/apply_turn_event_internal_adapter.py \
  --motion output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy \
  --ckpt runs/turn_event_internal_adapter/turn_event_internal_adapter.pt \
  --trajectory "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" \
  --out output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy \
  2>&1 | tee logs/turn_event_internal_adapter/apply_internal_adapter.log

python tools/evaluate_functional_choreo_coupling.py \
  --trajectory "0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2" \
  --motions "output/final_hybrid_candidates/dhw4_final_expr_plus_support_main.npy,output/v13_functional_base/dhw4_expr_mobile_base.npy,output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy,output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy,output/v13_frame_sweep_hybrid/dhw4_v13_f35_60_85_110_135_mild.npy,output/v13_frame_sweep_hybrid/dhw4_v13_f40_65_90_115_140_mild.npy" \
  --out output/metric_reports/dhw4_turn_event_internal_adapter_eval.json \
  2>&1 | tee logs/turn_event_internal_adapter/eval_internal_adapter.log

echo "✅ internal adapter training complete"
echo "ckpt: runs/turn_event_internal_adapter/turn_event_internal_adapter.pt"
echo "motion: output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy"
echo "metrics: output/metric_reports/dhw4_turn_event_internal_adapter_eval.json"
