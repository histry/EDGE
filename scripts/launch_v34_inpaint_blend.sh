#!/usr/bin/env bash
set -euo pipefail

# Launch V34 with boundary-compatible Event-RAG retrieval, masked local
# boundary inpainting, and latent transition-manifold blending.
#
# Default mode reuses the existing V33/V34 checkpoint inference path.  Set
# V34_TRAIN=1 when you want to run the full overnight research pipeline after
# replacing the code files.

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

# Keep strict locked-slot timing.  This solves duration drift but does not by
# itself solve incompatible motion boundaries.
export V34_WARP_HARD_PRUNE="${V34_WARP_HARD_PRUNE:-1}"
export V34_WARP_MIN="${V34_WARP_MIN:-0.82}"
export V34_WARP_MAX="${V34_WARP_MAX:-1.30}"
export V34_WARP_TOLERANCE="${V34_WARP_TOLERANCE:-0.0}"
export V34_WARP_PREFILTER_TOP_K="${V34_WARP_PREFILTER_TOP_K:-1024}"

# New boundary-compatibility retrieval policy.  These are intentionally
# retrieval-time thresholds; the later Contact-INR and absolute boundary gate
# still run as independent checks.
export V34_BOUNDARY_COMPAT="${V34_BOUNDARY_COMPAT:-1}"
export V34_COMPAT_HARD_PRUNE="${V34_COMPAT_HARD_PRUNE:-1}"
export V34_BOUNDARY_COMPAT_WEIGHT="${V34_BOUNDARY_COMPAT_WEIGHT:-1.20}"
export V34_COMPAT_MAX_POSE_JUMP="${V34_COMPAT_MAX_POSE_JUMP:-0.42}"
export V34_COMPAT_MAX_VELOCITY_JUMP="${V34_COMPAT_MAX_VELOCITY_JUMP:-0.060}"
export V34_COMPAT_MAX_ACCELERATION_JUMP="${V34_COMPAT_MAX_ACCELERATION_JUMP:-0.120}"
export V34_COMPAT_MAX_CONTACT_JUMP="${V34_COMPAT_MAX_CONTACT_JUMP:-0.62}"
export V34_COMPAT_MAX_YAW_GAP_DEG="${V34_COMPAT_MAX_YAW_GAP_DEG:-62}"
export V34_COMPAT_MAX_TRANSITION_COST="${V34_COMPAT_MAX_TRANSITION_COST:-0.95}"
export V34_COMPAT_SEMANTIC_HARD_PRUNE="${V34_COMPAT_SEMANTIC_HARD_PRUNE:-0}"

# GPU cache accelerates candidate-pair boundary metrics.  Disable only for CPU
# debugging or if CUDA/PyTorch3D is unavailable.
export V34_USE_GPU_RETRIEVAL="${V34_USE_GPU_RETRIEVAL:-1}"
export V34_GPU_STRICT="${V34_GPU_STRICT:-0}"
export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_INPAINT_REQUIRE_DIFFUSION="${V34_INPAINT_REQUIRE_DIFFUSION:-1}"
export V34_INPAINT_TRIGGER_RATIO="${V34_INPAINT_TRIGGER_RATIO:-0.72}"
export V34_INPAINT_COMPAT_SCORE_TRIGGER="${V34_INPAINT_COMPAT_SCORE_TRIGGER:-0.10}"
export V34_INPAINT_TAIL_FRAMES="${V34_INPAINT_TAIL_FRAMES:-6}"
export V34_INPAINT_HEAD_FRAMES="${V34_INPAINT_HEAD_FRAMES:-6}"
export V34_INPAINT_CONTEXT_FRAMES="${V34_INPAINT_CONTEXT_FRAMES:-4}"
export V34_INPAINT_BLEND="${V34_INPAINT_BLEND:-${V32_INR_TRUST:-0.35}}"
export V34_INPAINT_STEPS="${V34_INPAINT_STEPS:-${V32_INFERENCE_STEPS:-40}}"
export V34_INPAINT_MAX_RISK_RATIO="${V34_INPAINT_MAX_RISK_RATIO:-1.03}"
export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"
export V34_LATENT_BLEND_TOP_K="${V34_LATENT_BLEND_TOP_K:-3}"
export V34_LATENT_BLEND_TEMPERATURE="${V34_LATENT_BLEND_TEMPERATURE:-0.08}"
export V34_LATENT_BLEND_KEEP_RATIO="${V34_LATENT_BLEND_KEEP_RATIO:-1.01}"

if [[ "${V34_TRAIN:-0}" == "1" ]]; then
  export V34_TRAIN=1
  export RUN_ID="${RUN_ID:-v34_inpaint_blend_full_$(date +%Y%m%d_%H%M%S)}"
  bash scripts/launch_v34_full_overnight.sh "$@"
else
  export V34_TRAIN=0
  export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
  export RUN_ID="${RUN_ID:-v34_inpaint_blend_infer_$(date +%Y%m%d_%H%M%S)}"
  bash scripts/resume_v34_inference_v33ckpt.sh "$@"
fi
