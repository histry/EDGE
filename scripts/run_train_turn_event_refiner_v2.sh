#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_TURN_EVENT_REFINER_TRAIN=${EDGE_TURN_EVENT_REFINER_TRAIN:-1}

TRAJ=${EDGE_TURN_TRAJECTORY:-"0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"}
BASE=${EDGE_TURN_REFINER_BASE:-"output/v13_functional_base/dhw4_expr_mobile_base.npy"}
ANCHOR=${EDGE_TURN_REFINER_ANCHOR:-"output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy"}

# Multi pseudo-targets. Missing files are skipped safely.
CANDIDATES=(
  "output/v13_frame_sweep_hybrid/dhw4_v13_f35_60_85_110_135_mild.npy"
  "output/v13_frame_sweep_hybrid/dhw4_v13_f40_65_90_115_140_mild.npy"
  "output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy"
  "output/v13_functional_hybrid_sweep/dhw4_v13_mild.npy"
)
WEIGHTS=(
  "1.30"
  "1.00"
  "1.20"
  "0.80"
)

TARGETS=()
TARGET_WEIGHTS=()
for i in "${!CANDIDATES[@]}"; do
  if [ -f "${CANDIDATES[$i]}" ]; then
    TARGETS+=("${CANDIDATES[$i]}")
    TARGET_WEIGHTS+=("${WEIGHTS[$i]}")
  else
    echo "skip missing target: ${CANDIDATES[$i]}"
  fi
done

if [ "${#TARGETS[@]}" -lt 2 ]; then
  echo "[ERROR] Need at least two pseudo targets. Existing targets:" >&2
  printf '  %s\n' "${TARGETS[@]}" >&2
  exit 1
fi

mkdir -p output/metric_reports runs/turn_event_refiner output/v13_turn_event_hybrid logs/stage_day/turn_event_refiner_v2

TARGET_CSV=$(IFS=,; echo "${TARGETS[*]}")
WEIGHT_CSV=$(IFS=,; echo "${TARGET_WEIGHTS[*]}")

python tools/make_turn_event_pairs.py \
  --base "$BASE" \
  --anchor "$ANCHOR" \
  --targets "$TARGET_CSV" \
  --weights "$WEIGHT_CSV" \
  --trajectory "$TRAJ" \
  --seq_len 150 \
  --count 5 \
  --out_jsonl output/metric_reports/turn_event_refiner_pairs_v2.jsonl \
  --out_event_npy output/metric_reports/turn_event_refiner_pairs_v2.event.npy \
  --out_event_report output/metric_reports/turn_event_refiner_pairs_v2.event.json \
  2>&1 | tee logs/stage_day/turn_event_refiner_v2/01_make_pairs.log

python tools/train_turn_aware_event_refiner.py \
  --pairs output/metric_reports/turn_event_refiner_pairs_v2.jsonl \
  --out_ckpt runs/turn_event_refiner/turn_event_refiner_v2.pt \
  --epochs "${EDGE_TURN_REFINER_EPOCHS:-500}" \
  --lr "${EDGE_TURN_REFINER_LR:-0.0007}" \
  --hidden "${EDGE_TURN_REFINER_HIDDEN:-256}" \
  --depth "${EDGE_TURN_REFINER_DEPTH:-3}" \
  --dropout "${EDGE_TURN_REFINER_DROPOUT:-0.05}" \
  --max_delta "${EDGE_TURN_REFINER_MAX_DELTA:-0.14}" \
  2>&1 | tee logs/stage_day/turn_event_refiner_v2/02_train.log

python tools/apply_turn_aware_event_refiner.py \
  --ckpt runs/turn_event_refiner/turn_event_refiner_v2.pt \
  --base "$BASE" \
  --anchor "$ANCHOR" \
  --trajectory "$TRAJ" \
  --out output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2.npy \
  2>&1 | tee logs/stage_day/turn_event_refiner_v2/03_apply.log

MOTIONS="output/final_hybrid_candidates/dhw4_final_expr_plus_support_main.npy,$BASE,$ANCHOR,output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2.npy"
for t in "${TARGETS[@]}"; do
  MOTIONS="$MOTIONS,$t"
done

python tools/evaluate_functional_choreo_coupling.py \
  --motions "$MOTIONS" \
  --trajectory "$TRAJ" \
  --out output/metric_reports/dhw4_turn_event_refiner_v2_eval.json \
  2>&1 | tee logs/stage_day/turn_event_refiner_v2/04_eval.log

echo "✅ v2 refiner complete"
echo "ckpt: runs/turn_event_refiner/turn_event_refiner_v2.pt"
echo "motion: output/v13_turn_event_hybrid/dhw4_turn_event_refined_v2.npy"
echo "metrics: output/metric_reports/dhw4_turn_event_refiner_v2_eval.json"
