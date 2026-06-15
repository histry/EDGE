#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# ============================================================
# Required V26/V33 assets
# ============================================================
export V26_INDEX_JSON="${V26_INDEX_JSON:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json}"
export V26_DURATION_INDEX_NPZ="${V26_DURATION_INDEX_NPZ:-data/v26_music_dominant_duration_index.npz}"

export V26_ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
export V26_V23_CKPT="${V26_V23_CKPT:-output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt}"

PLANNER_POINTER="output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt"

if [[ -z "${V26_PLANNER_CKPT:-}" ]]; then
  [[ -f "$PLANNER_POINTER" ]] || {
    echo "[ERROR] Missing planner pointer: $PLANNER_POINTER" >&2
    exit 2
  }

  export V26_PLANNER_CKPT
  V26_PLANNER_CKPT="$(cat "$PLANNER_POINTER")"
fi

export V26_START_POSE="${V26_START_POSE:-data/canonical_dunhuang_start_pose.npy}"

export V27_HYPERBOLIC_CKPT="${V27_HYPERBOLIC_CKPT:-output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt}"

export V26_HIERARCHY_INDEX_NPZ="${V26_HIERARCHY_INDEX_NPZ:-output/v28_diffusion_retrain_musicclap_20260610_024828/v28_hyperbolic_hierarchical_event_index.npz}"

export V33_EVENT_CONTACT_CACHE="${V33_EVENT_CONTACT_CACHE:-data/v33_event_contact_cache.npz}"

# ============================================================
# Reuse V34 event library and V33 synchronised dataset
# ============================================================
export V34_BUILD_EVENT_LIBRARY=0
export V34_LIBRARY_DIR="${V34_LIBRARY_DIR:-data/v34_event_library}"
export V34_INDEX_JSON="${V34_INDEX_JSON:-data/v34_shared_event_index.json}"

export V34_CALIBRATE_THRESHOLDS="${V34_CALIBRATE_THRESHOLDS:-1}"

export V34_DATASET="${V34_DATASET:-data/v33_transition_dataset.npz}"

# ============================================================
# Reuse trained V33 checkpoint; no retraining
# ============================================================
export V34_TRAIN=0

export V27_TRANSITION_DIFFUSION_CKPT="${V27_TRANSITION_DIFFUSION_CKPT:-output/v33_event_contact_20260611_204533/v32_contact_inr_training/checkpoints/best.pt}"

# Diagnostic compatibility run. The post-handshake evaluator still records
# unsafe boundaries, but does not terminate before all reports are written.
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"

# ============================================================
# Deterministic V34.2 timing configuration
#
# Do not inherit V26 timing variables from old tmux sessions.
# These values guarantee that the first slot is split before
# retrieval when it is too long for the event-duration support.
# ============================================================
unset V26_MAX_SINGLE_EVENT_SECONDS
unset V26_CALM_MAX_SINGLE_EVENT_SECONDS
unset V26_MIN_SUBPHRASE_SECONDS
unset V26_MAX_EVENTS_PER_PHRASE
unset V26_TRANSITION_MIN_FRAMES
unset V26_TRANSITION_MAX_FRAMES
unset V26_MIN_CONTENT_FRAMES
unset V26_MIN_PHRASE_SECONDS
unset V26_MAX_PHRASE_SECONDS
unset V26_MAX_PHRASES
unset V26_BEAM_SIZE
unset V26_CANDIDATE_TOP_K
unset V26_GRAPH_NODE_TOP_K

export V26_MIN_PHRASE_SECONDS=2.50
export V26_MAX_PHRASE_SECONDS=7.50

export V26_MAX_SINGLE_EVENT_SECONDS=2.80
export V26_CALM_MAX_SINGLE_EVENT_SECONDS=2.60
export V26_MIN_SUBPHRASE_SECONDS=1.45
export V26_MAX_EVENTS_PER_PHRASE=4
export V26_MAX_PHRASES=160

export V26_MIN_CONTENT_FRAMES=18
export V26_TRANSITION_MIN_FRAMES=16
export V26_TRANSITION_MAX_FRAMES=54

export V26_BEAM_SIZE=48
export V26_CANDIDATE_TOP_K=768
export V26_GRAPH_NODE_TOP_K=512

# Use the same hierarchy index as the V33 diagnostic experiment.
export V26_HIERARCHY_INDEX_NPZ=output/v28_diffusion_retrain_musicclap_20260610_024828/v28_hyperbolic_hierarchical_event_index.npz

echo "============================================================"
echo "[V34.2 EFFECTIVE TIMING]"
echo "MAX_SINGLE_EVENT_SECONDS=$V26_MAX_SINGLE_EVENT_SECONDS"
echo "CALM_MAX_SINGLE_EVENT_SECONDS=$V26_CALM_MAX_SINGLE_EVENT_SECONDS"
echo "MIN_SUBPHRASE_SECONDS=$V26_MIN_SUBPHRASE_SECONDS"
echo "MAX_EVENTS_PER_PHRASE=$V26_MAX_EVENTS_PER_PHRASE"
echo "MIN_CONTENT_FRAMES=$V26_MIN_CONTENT_FRAMES"
echo "TRANSITION_MIN_FRAMES=$V26_TRANSITION_MIN_FRAMES"
echo "TRANSITION_MAX_FRAMES=$V26_TRANSITION_MAX_FRAMES"
echo "HIERARCHY_INDEX=$V26_HIERARCHY_INDEX_NPZ"
echo "============================================================"

# ============================================================
# Strict Warp-aware Retrieval with transition-budget negotiation
# ============================================================
export V34_WARP_HARD_PRUNE=1
export V34_WARP_MIN="${V34_WARP_MIN:-0.82}"
export V34_WARP_MAX="${V34_WARP_MAX:-1.30}"
export V34_WARP_TOLERANCE="${V34_WARP_TOLERANCE:-0.0}"

export V34_WARP_PREFILTER_TOP_K="${V34_WARP_PREFILTER_TOP_K:-768}"

export V34_TRANSITION_BUDGET_PENALTY_WEIGHT="${V34_TRANSITION_BUDGET_PENALTY_WEIGHT:-0.035}"

export V26_MIN_TIME_WARP="${V26_MIN_TIME_WARP:-0.82}"
export V26_MAX_TIME_WARP="${V26_MAX_TIME_WARP:-1.30}"

export V32_STRICT_LOCKED_WARP=1
export V32_MAX_WARP_VIOLATIONS=0

# ============================================================
# C3 boundary and Exit Handshake
# ============================================================
export V34_EXIT_HANDSHAKE="${V34_EXIT_HANDSHAKE:-1}"
export V34_EXIT_HANDSHAKE_FRAMES="${V34_EXIT_HANDSHAKE_FRAMES:-10}"
export V34_EXIT_HANDSHAKE_CANDIDATES="${V34_EXIT_HANDSHAKE_CANDIDATES:-8,10,12,16,20}"
export V34_HANDSHAKE_MODE="${V34_HANDSHAKE_MODE:-replace}"

export V34_POST_HANDSHAKE_ABSOLUTE_VETO="${V34_POST_HANDSHAKE_ABSOLUTE_VETO:-1}"
export V34_JERK_MATCH_SHRINK="${V34_JERK_MATCH_SHRINK:-0.35}"

# ============================================================
# V33 checkpoint inference settings
# ============================================================
export V32_CANDIDATES="${V32_CANDIDATES:-8}"
export V32_GUIDANCE="${V32_GUIDANCE:-1.0}"
export V32_INR_TRUST="${V32_INR_TRUST:-0.25}"
export V32_INFERENCE_STEPS="${V32_INFERENCE_STEPS:-40}"

export V34_RUN_SEPTIC_BASELINE="${V34_RUN_SEPTIC_BASELINE:-1}"

export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
export V32_KEYS="${V32_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"

# ============================================================
# Run directory and launcher log
# ============================================================
export RUN_ID="${RUN_ID:-v34_1_inference_v33ckpt_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"

mkdir -p output "$RUN_ROOT"

echo "$RUN_ROOT" > output/LATEST_V34_1_INFERENCE_LAUNCH.txt

# Capture errors that occur before run_v34_full_research.sh creates run.log.
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

echo "============================================================"
echo "[V34.1 LAUNCH] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
echo "============================================================"

# ============================================================
# Defensive preflight
# ============================================================
required_files=(
  "$V26_INDEX_JSON"
  "$V26_DURATION_INDEX_NPZ"
  "$V26_ROUTER_CKPT"
  "$V26_V23_CKPT"
  "$V26_PLANNER_CKPT"
  "$V26_START_POSE"
  "$V27_HYPERBOLIC_CKPT"
  "$V26_HIERARCHY_INDEX_NPZ"
  "$V33_EVENT_CONTACT_CACHE"
  "$V34_INDEX_JSON"
  "$V34_DATASET"
  "$V27_TRANSITION_DIFFUSION_CKPT"
  "scripts/run_v34_full_research.sh"
  "scripts/run_v34_whole_song.sh"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[PREFLIGHT ERROR] Missing required file: $path" >&2
    exit 2
  fi

  echo "[PREFLIGHT OK] $path"
done

[[ -d "$V34_LIBRARY_DIR" ]] || {
  echo "[PREFLIGHT ERROR] Missing V34 library directory: $V34_LIBRARY_DIR" >&2
  exit 2
}

echo "[PREFLIGHT OK] $V34_LIBRARY_DIR"

python - <<'PY'
import torch

print("[PYTORCH]", torch.__version__)
print("[CUDA AVAILABLE]", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

print("[GPU]", torch.cuda.get_device_name(0))
PY

echo "[PREFLIGHT PASSED]"
echo "[STARTING] scripts/run_v34_full_research.sh"

bash scripts/run_v34_full_research.sh
