#!/usr/bin/env bash
# One-command V46.53.1 full rebuild, retraining and current-WAV whole-song generation.
set -Eeuo pipefail
ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"

export ROOT_DIR
export V46_51_PYTHON="${V46_51_PYTHON:-/home/disk/lsm/conda_envs/edge/bin/python}"
export CHANGE_BVH_DIR="${CHANGE_BVH_DIR:-$ROOT_DIR/change}"
export MUSIC_DIRS="${MUSIC_DIRS:-$ROOT_DIR/data/v21_router_music_999/splits/train}"
export AUDIO="${1:-${AUDIO:-$ROOT_DIR/test_music_bank/dunhuangwu2.wav}}"
export RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/output/v46_53_1_research_${RUN_TAG}}"

[[ -s "$AUDIO" ]] || { echo "[FATAL] Input audio missing: $AUDIO" >&2; exit 2; }
[[ -d "$MUSIC_DIRS" ]] || { echo "[FATAL] Training music directory missing: $MUSIC_DIRS" >&2; exit 2; }
[[ "$MUSIC_DIRS" != *test_music_bank* ]] || { echo "[FATAL] test_music_bank cannot enter training" >&2; exit 2; }

mkdir -p "$ROOT_DIR/logs"
LOG="$ROOT_DIR/logs/v46_53_1_full_${RUN_TAG}.log"
echo "[RUN] AUDIO=$AUDIO"
echo "[RUN] OUT_ROOT=$OUT_ROOT"
echo "[RUN] LOG=$LOG"

bash scripts/run_v46_50_full_rebuild_retrain.sh "$AUDIO" 2>&1 | tee "$LOG"
