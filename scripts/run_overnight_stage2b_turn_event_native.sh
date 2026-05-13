#!/usr/bin/env bash
set -u
set -o pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

RUN_TAG=$(date +%Y%m%d_%H%M%S)
LOG_ROOT="logs/stage2b_overnight_${RUN_TAG}"
mkdir -p "$LOG_ROOT" \
  output/v13_turn_event_hybrid \
  output/metric_reports \
  output/rendered_candidates_real

echo "========== Stage2B Turn-event Native Overnight ==========" | tee "$LOG_ROOT/overnight.log"
date | tee -a "$LOG_ROOT/overnight.log"

# Optional: uncomment if you do not want wandb sync overnight.
# export WANDB_MODE=offline

BASE_V12="/home/disk/lsm/storage/EDGE/runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt"
SUPPORT_TEXTPose="/home/disk/lsm/storage/EDGE/runs/train_support_textpose_rag/v12_support_gait_textpose_rag_v1/weights/train-12.pt"

TRAJ="0,0;1.2,0.8;-1.2,1.6;1.2,2.4;0,3.2"
MUSIC="test_music_bank/dunhuangwu2_20s.wav"

run_cmd () {
  local name="$1"
  shift
  echo "" | tee -a "$LOG_ROOT/overnight.log"
  echo "========== [$name] ==========" | tee -a "$LOG_ROOT/overnight.log"
  echo "CMD: $*" | tee -a "$LOG_ROOT/overnight.log"

  "$@" > "$LOG_ROOT/${name}.log" 2>&1
  local code=$?

  if [ "$code" -ne 0 ]; then
    echo "❌ [$name] failed code=$code" | tee -a "$LOG_ROOT/overnight.log"
    tail -n 80 "$LOG_ROOT/${name}.log" | tee -a "$LOG_ROOT/overnight.log"
  else
    echo "✅ [$name] done" | tee -a "$LOG_ROOT/overnight.log"
  fi

  return 0
}

latest_train_ckpt () {
  find runs/train_advanced_traj_phase -type f -name "train-*.pt" \
    -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

summarize_metrics () {
  local json="$1"
  python - "$json" <<'PY'
import json, sys, os
path = sys.argv[1]
if not os.path.exists(path):
    print("missing metrics:", path)
    raise SystemExit(0)

data = json.load(open(path))
keys = [
    "root_path",
    "lower_activity",
    "torso_activity",
    "upper_activity",
    "contact_switch",
    "support_expression_coupling",
    "turn_expression_response",
    "target_turn_expression_response",
    "speed_expression_sync",
    "lower_torso_sync",
    "lower_upper_sync",
]
for name, m in data.items():
    print("\n===", os.path.basename(name), "===")
    for k in keys:
        print(f"{k}: {m.get(k)}")
PY
}

generate_eval_render () {
  local case_tag="$1"
  local ckpt="$2"

  echo "Generating / evaluating / rendering case=$case_tag ckpt=$ckpt" | tee -a "$LOG_ROOT/overnight.log"

  export EDGE_TURN_EVENT_MODEL_ADAPTER=1
  export EDGE_TURN_EVENT_TRAJ_TOKEN=1
  export EDGE_TURN_EVENT_OUTPUT_ADAPTER=0
  unset EDGE_TURN_EVENT_ADAPTER_CKPT || true
  export EDGE_DYNAMIC_TRAJ_CFG=0
  export EDGE_TURN_EVENT_PRESERVE_ROOT_XZ=1

  local motion="output/v13_turn_event_hybrid/${case_tag}.npy"
  local metrics="output/metric_reports/${case_tag}_eval.json"

  run_cmd "${case_tag}_generate" \
    python generate_controlled.py \
      --checkpoint "$ckpt" \
      --music "$MUSIC" \
      --feature_type hybrid \
      --start_pose test_keyframes/dyl002_600_1800_start.npy \
      --end_pose test_keyframes/dyl002_600_1800_end.npy \
      --trajectory "$TRAJ" \
      --out "$motion" \
      --pose_space normalized \
      --sampler ddpm \
      --no_tto

  if [ ! -f "$motion" ]; then
    echo "❌ motion missing for $case_tag: $motion" | tee -a "$LOG_ROOT/overnight.log"
    return 0
  fi

  run_cmd "${case_tag}_eval" \
    python tools/evaluate_functional_choreo_coupling.py \
      --motions "output/final_hybrid_candidates/dhw4_final_expr_plus_support_main.npy,output/v13_functional_base/dhw4_expr_mobile_base.npy,$motion,output/v13_turn_event_hybrid/dhw4_turn_event_internal_adapter.npy,output/v13_turn_event_hybrid/dhw4_turn_event_v13.npy,output/v13_frame_sweep_hybrid/dhw4_v13_f35_60_85_110_135_mild.npy,output/v13_frame_sweep_hybrid/dhw4_v13_f40_65_90_115_140_mild.npy" \
      --trajectory "$TRAJ" \
      --out "$metrics"

  echo "---------- metrics summary: $case_tag ----------" | tee -a "$LOG_ROOT/overnight.log"
  summarize_metrics "$metrics" | tee -a "$LOG_ROOT/overnight.log"

  run_cmd "${case_tag}_render_fixed" \
    python render_choreorag_results.py \
      --motion "$motion" \
      --music "$MUSIC" \
      --out "output/rendered_candidates_real/${case_tag}_fixed_rows.mp4" \
      --camera_mode fixed \
      --sixd_layout rows \
      --smooth_window 7

  run_cmd "${case_tag}_render_bodycenter" \
    python render_choreorag_results.py \
      --motion "$motion" \
      --music "$MUSIC" \
      --out "output/rendered_candidates_real/${case_tag}_bodycenter_rows.mp4" \
      --camera_mode fixed \
      --sixd_layout rows \
      --body_centered \
      --smooth_window 7
}

train_case () {
  local case_tag="$1"
  local start_ckpt="$2"
  local lower_gate="$3"
  local torso_gate="$4"
  local upper_gate="$5"

  if [ ! -f "$start_ckpt" ]; then
    echo "❌ missing checkpoint for $case_tag: $start_ckpt" | tee -a "$LOG_ROOT/overnight.log"
    return 0
  fi

  echo "" | tee -a "$LOG_ROOT/overnight.log"
  echo "################################################################################" | tee -a "$LOG_ROOT/overnight.log"
  echo "START CASE: $case_tag" | tee -a "$LOG_ROOT/overnight.log"
  echo "CHECKPOINT: $start_ckpt" | tee -a "$LOG_ROOT/overnight.log"
  echo "GATES: lower=$lower_gate torso=$torso_gate upper=$upper_gate" | tee -a "$LOG_ROOT/overnight.log"
  echo "################################################################################" | tee -a "$LOG_ROOT/overnight.log"

  export CHECKPOINT="$start_ckpt"

  # Native adapter training: no output residual.
  export EDGE_TURN_EVENT_OUTPUT_ADAPTER=0
  unset EDGE_TURN_EVENT_ADAPTER_CKPT || true

  export EDGE_TURN_EVENT_MODEL_ADAPTER=1
  export EDGE_TURN_EVENT_TRAJ_TOKEN=1
  export EDGE_TURN_EVENT_FREEZE_BACKBONE=1

  export EDGE_TURN_SUPPORT_LAG=8
  export EDGE_TURN_EXPR_LAG=4
  export EDGE_TURN_MIN_GAP=18
  export EDGE_TURN_GATE_SIGMA=5.0

  export EDGE_DYNAMIC_TRAJ_CFG=0
  export EDGE_TURN_EVENT_PRESERVE_ROOT_XZ=1
  export EDGE_TURN_EVENT_GATE_ROOT_XZ=0.0
  export EDGE_TURN_EVENT_GATE_ROOT_Y=0.0
  export EDGE_TURN_EVENT_GATE_LOWER="$lower_gate"
  export EDGE_TURN_EVENT_GATE_TORSO="$torso_gate"
  export EDGE_TURN_EVENT_GATE_UPPER="$upper_gate"

  env | grep -E "CHECKPOINT|EDGE_TURN|EDGE_DYNAMIC_TRAJ_CFG" | sort \
    > "$LOG_ROOT/${case_tag}_env.log"

  run_cmd "${case_tag}_train" \
    bash scripts/run_stage2b_turn_event_native_adapter.sh

  local ckpt
  ckpt=$(latest_train_ckpt)

  if [ -z "$ckpt" ] || [ ! -f "$ckpt" ]; then
    echo "❌ no checkpoint found after $case_tag" | tee -a "$LOG_ROOT/overnight.log"
    return 0
  fi

  echo "LATEST_CKPT_${case_tag}=$ckpt" | tee "$LOG_ROOT/${case_tag}_latest_ckpt.txt" | tee -a "$LOG_ROOT/overnight.log"

  generate_eval_render "$case_tag" "$ckpt"

  echo "$ckpt"
}

echo "========== preflight ==========" | tee -a "$LOG_ROOT/overnight.log"
git log --oneline -5 > "$LOG_ROOT/git_log.txt" 2>&1 || true
git status --short > "$LOG_ROOT/git_status.txt" 2>&1 || true
nvidia-smi > "$LOG_ROOT/nvidia_smi_start.txt" 2>&1 || true

echo "BASE_V12=$BASE_V12" | tee -a "$LOG_ROOT/overnight.log"
echo "SUPPORT_TEXTPose=$SUPPORT_TEXTPose" | tee -a "$LOG_ROOT/overnight.log"

# ---------------------------------------------------------------------
# Case A: clean v12 baseline → balanced native event adapter
# ---------------------------------------------------------------------
A_CKPT=$(train_case \
  "dhw4_stage2b_native_v12_balanced" \
  "$BASE_V12" \
  "0.45" "0.75" "0.75" | tail -n 1)

# ---------------------------------------------------------------------
# Case B: continue from A → stronger turn-expression gates
# ---------------------------------------------------------------------
if [ -n "${A_CKPT:-}" ] && [ -f "$A_CKPT" ]; then
  B_START="$A_CKPT"
else
  B_START="$BASE_V12"
fi

B_CKPT=$(train_case \
  "dhw4_stage2b_native_v12_turnexpr" \
  "$B_START" \
  "0.30" "1.00" "1.00" | tail -n 1)

# ---------------------------------------------------------------------
# Case C: support/textpose checkpoint → turn-expression focused native adapter
# ---------------------------------------------------------------------
if [ -f "$SUPPORT_TEXTPose" ]; then
  train_case \
    "dhw4_stage2b_native_support_textpose_turnexpr" \
    "$SUPPORT_TEXTPose" \
    "0.35" "1.00" "1.00" >/dev/null
else
  echo "⚠️ support_textpose checkpoint missing, skip Case C: $SUPPORT_TEXTPose" | tee -a "$LOG_ROOT/overnight.log"
fi

echo "" | tee -a "$LOG_ROOT/overnight.log"
echo "========== OVERNIGHT FINISHED ==========" | tee -a "$LOG_ROOT/overnight.log"
date | tee -a "$LOG_ROOT/overnight.log"

echo "Metrics files:" | tee -a "$LOG_ROOT/overnight.log"
find output/metric_reports -type f -name "dhw4_stage2b_native*_eval.json" \
  -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort | tail -20 | tee -a "$LOG_ROOT/overnight.log"

echo "Rendered files:" | tee -a "$LOG_ROOT/overnight.log"
find output/rendered_candidates_real -type f -name "dhw4_stage2b_native*mp4" \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" | sort | tail -30 | tee -a "$LOG_ROOT/overnight.log"

