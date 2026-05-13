#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_TURN_SUPPORT_LAG=${EDGE_TURN_SUPPORT_LAG:-8}
export EDGE_TURN_EXPR_LAG=${EDGE_TURN_EXPR_LAG:-4}
export EDGE_TURN_MIN_GAP=${EDGE_TURN_MIN_GAP:-18}
export EDGE_TURN_GATE_SIGMA=${EDGE_TURN_GATE_SIGMA:-5.0}

TAG=${1:-dhw4_turn_event_v13}
TRAJ=${EDGE_TRAJ:-"0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"}
BASE=${EDGE_BASE_MOTION:-output/v13_functional_base/dhw4_expr_mobile_base.npy}

mkdir -p output/v13_turn_event_context output/v13_turn_event_hybrid output/metric_reports logs/stage_day/v13_turn_event

python functional_dual_context_selector.py \
  --rag_db data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
  --trajectory "$TRAJ" \
  --seq_len 150 \
  --count 5 \
  --auto_event_frames \
  --support_lag "$EDGE_TURN_SUPPORT_LAG" \
  --expressive_lag "$EDGE_TURN_EXPR_LAG" \
  --support_k 5 \
  --expressive_k 5 \
  --out_dir "output/v13_turn_event_context/$TAG" \
  --prefix "$TAG" \
  --unit_space normalized \
  2>&1 | tee "logs/stage_day/v13_turn_event/${TAG}_select.log"

source "output/v13_turn_event_context/$TAG/${TAG}_functional_context.env"

python functional_dual_context_compositor.py \
  --base "$BASE" \
  --support_units "$EDGE_SUPPORT_CONTEXT_UNIT_PATHS" \
  --expressive_units "$EDGE_EXPRESSIVE_CONTEXT_UNIT_PATHS" \
  --support_frames "$EDGE_SUPPORT_CONTEXT_FRAMES" \
  --expressive_frames "$EDGE_EXPRESSIVE_CONTEXT_FRAMES" \
  --out "output/v13_turn_event_hybrid/${TAG}.npy" \
  --window 45 \
  --support_lower_strength 0.45 \
  --support_contact_strength 0.55 \
  --support_torso_strength 0.05 \
  --expressive_torso_strength 0.20 \
  --expressive_upper_strength 0.30 \
  --expressive_lower_strength 0.05 \
  2>&1 | tee "logs/stage_day/v13_turn_event/${TAG}_compositor.log"

python tools/evaluate_functional_choreo_coupling.py \
  --trajectory "$TRAJ" \
  --motions "output/final_hybrid_candidates/dhw4_final_expr_plus_support_main.npy,$BASE,output/v13_turn_event_hybrid/${TAG}.npy,output/v12_footstep_stage1/dhw4_v12_main_composited.npy" \
  --out "output/metric_reports/${TAG}_coupling_metrics.json" \
  2>&1 | tee "logs/stage_day/v13_turn_event/${TAG}_eval.log"

echo "✅ no-train turn-aware event result: output/v13_turn_event_hybrid/${TAG}.npy"
echo "✅ metrics: output/metric_reports/${TAG}_coupling_metrics.json"
