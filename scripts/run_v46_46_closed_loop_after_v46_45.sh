#!/usr/bin/env bash
set -euo pipefail

# V46.46 closed-loop boundary-safe experiment launcher.
# Copy this script to <EDGE_ROOT>/scripts/ and run from anywhere.
# It assumes V46.44/V46.45 representation-contract fixes have already been applied.

EDGE_ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$EDGE_ROOT"
export PYTHONPATH="$EDGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# ===== Device / data =====
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_45_BVH_ROOT_ROT_MODE="${V46_45_BVH_ROOT_ROT_MODE:-yaw}"

AUDIO="${AUDIO:-dunhuangwu2.wav}"
SLOTS_JSON="${SLOTS_JSON:-}"
MUSIC_SEMANTIC_DIRS="${MUSIC_SEMANTIC_DIRS:-}"
MOTION_DIRS="${MOTION_DIRS:-change}"
RUN_ROOT="${RUN_ROOT:-output/v46_46_closed_loop_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

# ===== Closed-loop switches =====
export V46_46_CLOSED_LOOP_ENABLE="${V46_46_CLOSED_LOOP_ENABLE:-1}"
export V46_46_CANDIDATE_TOPK="${V46_46_CANDIDATE_TOPK:-64}"
export V46_46_RESELECT_TOPK="${V46_46_RESELECT_TOPK:-12}"
export V46_46_MAX_RESELECT_ROUNDS="${V46_46_MAX_RESELECT_ROUNDS:-2}"
export V46_46_RESELECT_ENABLE="${V46_46_RESELECT_ENABLE:-1}"
export V46_46_RISK_ADAPT_TRANSITION_ENABLE="${V46_46_RISK_ADAPT_TRANSITION_ENABLE:-1}"

# Safety thresholds.  Keep these moderately strict for research comparison.
export V46_46_MAX_BOUNDARY_JERK="${V46_46_MAX_BOUNDARY_JERK:-5000}"
export V46_46_MAX_EXIT_FK="${V46_46_MAX_EXIT_FK:-0.040}"
export V46_46_MAX_EXIT_ROT="${V46_46_MAX_EXIT_ROT:-0.12}"
export V46_46_MAX_FOOT_SLIP="${V46_46_MAX_FOOT_SLIP:-0.28}"
export V46_46_MAX_FOOT_PENETRATION="${V46_46_MAX_FOOT_PENETRATION:-0.0025}"

# Risk-adaptive transition budget weights.
export V46_46_TLEN_POSE_W="${V46_46_TLEN_POSE_W:-10.0}"
export V46_46_TLEN_VEL_W="${V46_46_TLEN_VEL_W:-4.0}"
export V46_46_TLEN_YAW_W="${V46_46_TLEN_YAW_W:-3.0}"
export V46_46_TLEN_CONTACT_W="${V46_46_TLEN_CONTACT_W:-8.0}"
export V46_46_TLEN_FK_W="${V46_46_TLEN_FK_W:-80.0}"
export V46_46_TLEN_EXTRA_MAX="${V46_46_TLEN_EXTRA_MAX:-14.0}"

# Default V46 generators.  Set to 0 for ablation.
export V46_46_USE_REFINER="${V46_46_USE_REFINER:-1}"
export V46_46_USE_DIFFUSION="${V46_46_USE_DIFFUSION:-1}"
export V46_46_USE_IK="${V46_46_USE_IK:-1}"

# ===== Optional rebuild/retrain =====
# Set REBUILD_DB=1 after changing change/*.bvh or V46.44/V46.45 loader contract.
REBUILD_DB="${REBUILD_DB:-0}"
TRAIN_V44="${TRAIN_V44:-0}"
TRAIN_V45="${TRAIN_V45:-0}"
TRAIN_V46="${TRAIN_V46:-0}"

DB="${DB:-$RUN_ROOT/v46_46_db}"
CONTRASTIVE="${CONTRASTIVE:-$RUN_ROOT/v44_contrastive.pt}"
REFINER="${REFINER:-$RUN_ROOT/v45_refiner.pt}"
DIFFUSION="${DIFFUSION:-$RUN_ROOT/v46_diffusion.pt}"

if [[ "$REBUILD_DB" == "1" ]]; then
  echo "[V46.46] Rebuilding DB from: $MOTION_DIRS"
  python tools/v46_motionrag_diff.py build-db \
    --motion_dirs $MOTION_DIRS \
    --out_db "$DB"
fi

if [[ "$TRAIN_V44" == "1" ]]; then
  echo "[V46.46] Training V44 contrastive"
  python tools/v46_motionrag_diff.py train-contrastive \
    --db "$DB" \
    --out "$CONTRASTIVE" \
    ${MUSIC_SEMANTIC_DIRS:+--music_semantic_dirs $MUSIC_SEMANTIC_DIRS}
fi

if [[ "$TRAIN_V45" == "1" ]]; then
  echo "[V46.46] Training V45 refiner"
  python tools/v46_motionrag_diff.py train-refiner \
    --db "$DB" \
    --out "$REFINER"
fi

if [[ "$TRAIN_V46" == "1" ]]; then
  echo "[V46.46] Training V46 diffusion"
  python tools/v46_motionrag_diff.py train-diffusion \
    --db "$DB" \
    --out "$DIFFUSION"
fi

if [[ ! -e "$DB/events.npz" && ! -e "$DB" ]]; then
  echo "[V46.46 ERROR] DB not found: $DB" >&2
  echo "Set DB=... or REBUILD_DB=1" >&2
  exit 2
fi

ARGS=(
  generate
  --config "${CONFIG:-configs/v46_motionrag_diff_config.json}"
  --audio "$AUDIO"
  --db "$DB"
  --out "$RUN_ROOT/dunhuangwu2_v46_46_closed_loop.npy"
  --json "$RUN_ROOT/dunhuangwu2_v46_46_closed_loop.report.json"
  --render_output "$RUN_ROOT/dunhuangwu2_v46_46_closed_loop.mp4"
)

[[ -n "$SLOTS_JSON" ]] && ARGS+=(--slots_json "$SLOTS_JSON")
[[ -n "$MUSIC_SEMANTIC_DIRS" ]] && ARGS+=(--music_semantic_dirs $MUSIC_SEMANTIC_DIRS)
[[ -e "$CONTRASTIVE" ]] && ARGS+=(--contrastive "$CONTRASTIVE")
[[ -e "$REFINER" ]] && ARGS+=(--refiner "$REFINER")
[[ -e "$DIFFUSION" ]] && ARGS+=(--diffusion "$DIFFUSION")

cp "$0" "$RUN_ROOT/launcher_used.sh" || true
printenv | grep -E '^(V46|V26|REBUILD|TRAIN|DB|AUDIO|SLOTS|MOTION|RUN_ROOT|CUDA)' | sort > "$RUN_ROOT/env_used.txt" || true

echo "[V46.46] Running closed-loop generation"
echo "[V46.46] RUN_ROOT=$RUN_ROOT"
python tools/v46_46_boundary_closed_loop.py "${ARGS[@]}" | tee "$RUN_ROOT/generate_stdout.jsonl"

echo "[V46.46] Done. Outputs:"
ls -lh "$RUN_ROOT" || true
