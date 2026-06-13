#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

# Reuse assets that already passed audit in the failed V34 run.
export V34_BUILD_EVENT_LIBRARY=0
export V34_LIBRARY_DIR="${V34_LIBRARY_DIR:-data/v34_event_library}"
export V34_INDEX_JSON="${V34_INDEX_JSON:-data/v34_shared_event_index.json}"
export V34_CALIBRATE_THRESHOLDS="${V34_CALIBRATE_THRESHOLDS:-1}"
export V34_TRAIN=0
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export V34_DATASET="${V34_DATASET:-data/v33_transition_dataset.npz}"
export V27_TRANSITION_DIFFUSION_CKPT="${V27_TRANSITION_DIFFUSION_CKPT:-output/v33_event_contact_20260611_204533/v32_contact_inr_training/checkpoints/best.pt}"

# Keep strict paper bounds. The patch negotiates the transition budget inside
# this interval; it does not permit a warp violation.
export V34_WARP_HARD_PRUNE=1
export V34_WARP_MIN="${V34_WARP_MIN:-0.82}"
export V34_WARP_MAX="${V34_WARP_MAX:-1.30}"
export V34_WARP_TOLERANCE="${V34_WARP_TOLERANCE:-0.0}"
export V34_WARP_PREFILTER_TOP_K="${V34_WARP_PREFILTER_TOP_K:-768}"
export V34_TRANSITION_BUDGET_PENALTY_WEIGHT="${V34_TRANSITION_BUDGET_PENALTY_WEIGHT:-0.035}"
export V32_MAX_WARP_VIOLATIONS=0

export RUN_ID="${RUN_ID:-v34_1_inference_v33ckpt_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_1_INFERENCE_LAUNCH.txt

bash scripts/run_v34_full_research.sh
