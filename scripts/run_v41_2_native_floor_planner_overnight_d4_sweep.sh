#!/usr/bin/env bash
set -u

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
source scripts/v41_2_native_floor_planner_env.sh

MASTER_ROOT="${MASTER_ROOT:-output/v41_2_fk_verified_native_floor_planner_overnight_d4_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$MASTER_ROOT"
echo "$MASTER_ROOT" > output/LATEST_V41_2_NATIVE_FLOOR_PLANNER_OVERNIGHT_D4.txt

echo "[V41.2 OVERNIGHT START] $(date)"
echo "[MASTER_ROOT] $MASTER_ROOT"

# safe dead alpha beta rescue
declare -a CONFIGS=(
  "0.012 0.060 7.0 2.2 20.0"
  "0.012 0.055 9.0 2.5 25.0"
  "0.012 0.052 11.0 2.5 30.0"
  "0.010 0.050 13.0 2.8 35.0"
  "0.010 0.046 15.0 3.0 40.0"
)

idx=0
for cfg in "${CONFIGS[@]}"; do
  idx=$((idx+1))
  read -r SAFE DEAD ALPHA BETA RESCUE <<< "$cfg"
  TAG="s${SAFE/./p}_d${DEAD/./p}_a${ALPHA/./p}_b${BETA/./p}_r${RESCUE/./p}"
  RUN_ROOT="$MASTER_ROOT/$idx-$TAG"
  mkdir -p "$RUN_ROOT"

  echo
  echo "============================================================"
  echo "[V41.2 RUN] $idx safe=$SAFE dead=$DEAD alpha=$ALPHA beta=$BETA rescue=$RESCUE"
  echo "[RUN_ROOT] $RUN_ROOT"
  echo "============================================================"

  export RUN_ROOT
  export V41_NATIVE_FLOOR_TAU_SAFE_M="$SAFE"
  export V41_NATIVE_FLOOR_TAU_DEAD_M="$DEAD"
  export V41_NATIVE_FLOOR_ALPHA="$ALPHA"
  export V41_NATIVE_FLOOR_BETA="$BETA"
  export V41_NATIVE_FLOOR_DEAD_RESCUE_PENALTY="$RESCUE"

  bash scripts/run_v41_2_native_floor_planner_d4.sh > "$RUN_ROOT/launcher.log" 2>&1
  CODE=$?
  echo "[V41.2 RUN EXIT] idx=$idx code=$CODE"

  python - <<PY
import json, glob, os
run="$RUN_ROOT"
audit=f"{run}/v41_2_augmented_index/v41_2_native_floor_barrier_audit.json"
if os.path.exists(audit):
    a=json.load(open(audit)); print("[AUDIT]", json.dumps(a.get("summary",{}), ensure_ascii=False))
files=glob.glob(f"{run}/v41_2_native_floor_planner/*motion_quality_postprocess*.json")
print("[RESULT JSON COUNT]", len(files))
for f in sorted(files):
    d=json.load(open(f)); post=d.get("post_audit",{}); pf=d.get("planner_feedback",{})
    print("[RESULT]", os.path.basename(f),
          "accepted=", pf.get("accepted"),
          "reject=", pf.get("reject_reasons"),
          "foot_pen=", post.get("foot_penetration_min_m"),
          "skate_p95=", post.get("foot_skate_p95_mpf"),
          "jerk_p95=", post.get("mean_joint_jerk_p95"),
          "local_ik_after=", d.get("local_floor_ik",{}).get("after",{}).get("max_penetration"),
          "root_y_delta_max=", d.get("root_y_delta_max"),
          "has_v41beat=", "beat_decoupled_support_stabilizer" in d)
PY
done

echo "[V41.2 OVERNIGHT DONE] $(date)"
echo "[MASTER_ROOT] $MASTER_ROOT"
