#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

source scripts/v40b_native_floor_env.sh

RUN_ID="${RUN_ID:-v40b_native_floor_reroute_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

echo "[V40B START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"

pick_existing() {
  for p in "$@"; do
    if [[ -n "${p:-}" && -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

SRC_JSON="${V40B_SOURCE_INDEX_JSON:-}"
SRC_NPZ="${V40B_SOURCE_INDEX_NPZ:-}"

if [[ -z "$SRC_JSON" ]]; then
  SRC_JSON="$(pick_existing \
    "${V34_INDEX_JSON:-}" \
    "${V26_INDEX_JSON:-}" \
    "data/v34_source_aware/v34_shared_event_index_source_aware.json" \
    "data/v34_source_aware/v34_shared_event_index.json" \
    "output/v38_source_aware_full_train_20260625_212818/v34_shared_event_index.json" \
    "output/v38_source_aware_full_train_20260625_212818/v34_source_aware/v34_shared_event_index_source_aware.json" \
  )" || { echo "[ERROR] Cannot find source JSON. Set V40B_SOURCE_INDEX_JSON." >&2; exit 2; }
fi

if [[ -z "$SRC_NPZ" ]]; then
  SRC_NPZ="$(pick_existing \
    "${V26_DURATION_INDEX_NPZ:-}" \
    "data/v34_source_aware/v34_shared_event_index_source_aware.npz" \
    "data/v34_source_aware/v34_shared_event_index.npz" \
    "data/v21_shared_event_index.npz" \
  )" || { echo "[ERROR] Cannot find source NPZ. Set V40B_SOURCE_INDEX_NPZ or V26_DURATION_INDEX_NPZ." >&2; exit 2; }
fi

echo "[SOURCE JSON] $SRC_JSON"
echo "[SOURCE NPZ ] $SRC_NPZ"

PRUNE_DIR="$RUN_ROOT/v40b_pruned_index"
mkdir -p "$PRUNE_DIR"
PRUNED_JSON="$PRUNE_DIR/v40b_native_floor_pruned_index.json"
PRUNED_NPZ="$PRUNE_DIR/v40b_native_floor_pruned_index.npz"
AUDIT_JSON="$PRUNE_DIR/v40b_native_floor_prune_audit.json"
REMOVED_TXT="$PRUNE_DIR/v40b_removed_event_ids.txt"

SEARCH_ROOT_ARGS=()
IFS=':' read -ra ROOTS <<< "${V40B_MOTION_SEARCH_ROOTS:-data:output:$(dirname "$SRC_JSON"):$(pwd)}"
for r in "${ROOTS[@]}"; do
  [[ -n "$r" ]] && SEARCH_ROOT_ARGS+=(--search_root "$r")
done

echo "[STAGE] native floor source pruning"
python tools/v40b_native_floor_prune.py \
  --index_json "$SRC_JSON" \
  --index_npz "$SRC_NPZ" \
  --out_json "$PRUNED_JSON" \
  --out_npz "$PRUNED_NPZ" \
  --audit_json "$AUDIT_JSON" \
  --removed_txt "$REMOVED_TXT" \
  --mode "${V40B_NATIVE_FLOOR_MODE:-remove}" \
  --missing_policy "${V40B_MISSING_POLICY:-keep}" \
  --quantile "${V40B_NATIVE_FLOOR_QUANTILE:-0.05}" \
  --margin "${V40B_NATIVE_FLOOR_MARGIN:-0.006}" \
  --tolerance "${V40B_NATIVE_FLOOR_TOLERANCE_M:-0.040}" \
  --soft_threshold "${V40B_NATIVE_FLOOR_SOFT_THRESHOLD:-0.055}" \
  --remove_threshold "${V40B_NATIVE_FLOOR_REMOVE_THRESHOLD:-0.080}" \
  --penalty_weight "${V40B_NATIVE_FLOOR_PENALTY_WEIGHT:-18.0}" \
  --max_remove_fraction "${V40B_MAX_REMOVE_FRACTION:-0.35}" \
  --min_remaining "${V40B_MIN_REMAINING_EVENTS:-128}" \
  "${SEARCH_ROOT_ARGS[@]}" \
  ${V40B_FORCE_PRUNE:+--force}

export V34_INDEX_JSON="$PRUNED_JSON"
export V26_INDEX_JSON="$PRUNED_JSON"
export V26_DURATION_INDEX_NPZ="$PRUNED_NPZ"
export V26_MUSIC="${V40B_MUSIC:-test_music_bank/dunhuangwu4.wav}"
export V32_KEYS="${V40B_KEYS:-dunhuangwu4}"
export V26_OUT_DIR="$RUN_ROOT/v40b_native_floor_reroute"
export V27_TRANSITION_DIFFUSION="${V27_TRANSITION_DIFFUSION:-1}"

: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

echo "[STAGE] V34 graph planner forced reroute"
bash scripts/run_v34_whole_song.sh

echo "[STAGE] V40 postprocess rerouted outputs"
IFS=';' read -ra KEYS <<< "$V32_KEYS"
for key in "${KEYS[@]}"; do
  raw="$V26_OUT_DIR/${key}_v26.npy"
  [[ -f "$raw" ]] || { echo "[ERROR] Missing rerouted raw motion: $raw" >&2; exit 3; }
  out="$V26_OUT_DIR/${key}_v40b_native_floor_reroute.npy"
  summary="$V26_OUT_DIR/${key}_v40b_native_floor_reroute.motion_quality_postprocess.v40b.json"
  python tools/v34_motion_quality_postprocess.py \
    --motion "$raw" \
    --out "$out" \
    --summary_json "$summary" \
    --contact_lock "${V34_CONTACT_LOCK_POSTPROCESS:-1}" \
    --root_y_physics "${V34_ROOT_Y_PHYSICS:-1}" \
    --collision_ik "${V34_COLLISION_IK:-0}" \
    --floor_clearance "${V34_FLOOR_CLEARANCE_POSTPROCESS:-1}" \
    --butterworth_filter "${V38_BUTTERWORTH_FILTER:-1}"
  echo "[V40B POST] $out"
  echo "[V40B SUMMARY] $summary"
done

python - <<'PY'
import json, os, glob
root=os.environ["V26_OUT_DIR"]
print("\n[V40B FINAL CHECK]")
for f in sorted(glob.glob(root+"/*v40b_native_floor_reroute.motion_quality_postprocess.v40b.json")):
    d=json.load(open(f))
    post=d.get("post_audit",{})
    pf=d.get("planner_feedback",{})
    print(os.path.basename(f))
    print("  version:", d.get("version"))
    print("  accepted:", pf.get("accepted"), "reject:", pf.get("reject_reasons"))
    print("  foot_pen:", post.get("foot_penetration_min_m"))
    print("  skate_p95:", post.get("foot_skate_p95_mpf"))
    print("  jerk_p95:", post.get("mean_joint_jerk_p95"))
    print("  has_v41:", "beat_decoupled_support_stabilizer" in d)
PY

echo "$RUN_ROOT" > output/LATEST_V40B_NATIVE_FLOOR_REROUTE.txt
echo "[V40B DONE] $(date)"
echo "[LATEST] output/LATEST_V40B_NATIVE_FLOOR_REROUTE.txt"
