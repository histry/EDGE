#!/usr/bin/env bash
set -euo pipefail

# Robust V34 launcher for the boundary-inpaint + graceful-degradation pipeline.
# It is intentionally explicit so inference cannot silently fall back to an old
# checkpoint or run only the baseline stage without visible stage markers.

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

export V34_TRAIN="${V34_TRAIN:-0}"
export V34_BUILD_EVENT_LIBRARY="${V34_BUILD_EVENT_LIBRARY:-0}"
export V34_OVERWRITE_EVENT_LIBRARY="${V34_OVERWRITE_EVENT_LIBRARY:-0}"

export V34_BOUNDARY_COMPAT="${V34_BOUNDARY_COMPAT:-1}"
export V34_COMPAT_HARD_PRUNE="${V34_COMPAT_HARD_PRUNE:-1}"
export V34_BOUNDARY_COMPAT_WEIGHT="${V34_BOUNDARY_COMPAT_WEIGHT:-1.20}"
export V34_USE_GPU_RETRIEVAL="${V34_USE_GPU_RETRIEVAL:-1}"

export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_INPAINT_REQUIRE_DIFFUSION="${V34_INPAINT_REQUIRE_DIFFUSION:-1}"
export V34_INPAINT_ACCEPT_IF_ABSOLUTE_IMPROVES="${V34_INPAINT_ACCEPT_IF_ABSOLUTE_IMPROVES:-1}"
export V34_INPAINT_MAX_RISK_RATIO="${V34_INPAINT_MAX_RISK_RATIO:-1.03}"

export V34_POST_HANDSHAKE_ABSOLUTE_VETO="${V34_POST_HANDSHAKE_ABSOLUTE_VETO:-1}"
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export V34_DEFER_UNSAFE_BOUNDARY="${V34_DEFER_UNSAFE_BOUNDARY:-1}"
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE="${V34_HANDSHAKE_FALLBACK_ON_UNSAFE:-1}"
export V34_POST_HANDSHAKE_REPAIR="${V34_POST_HANDSHAKE_REPAIR:-0}"

export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"
export V34_LATENT_BLEND_TOP_K="${V34_LATENT_BLEND_TOP_K:-3}"

if [[ -z "${V27_TRANSITION_DIFFUSION_CKPT:-}" ]]; then
  latest_ckpt="$(
    { find output \
      -path '*/v34_contact_inr_training/checkpoints/best.pt' \
      -type f -printf '%T@ %p\n' 2>/dev/null || true; } \
    | sort -nr | awk 'NR==1 {print $2}'
  )"
  if [[ -n "$latest_ckpt" ]]; then
    export V27_TRANSITION_DIFFUSION_CKPT="$latest_ckpt"
  fi
fi

export RUN_ID="${RUN_ID:-v34_graceful_pipeline_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_GRACEFUL_PIPELINE.txt

echo "[GRACEFUL PIPELINE] RUN_ROOT=$RUN_ROOT"
echo "[GRACEFUL PIPELINE] V34_TRAIN=$V34_TRAIN"
echo "[GRACEFUL PIPELINE] V27_TRANSITION_DIFFUSION_CKPT=${V27_TRANSITION_DIFFUSION_CKPT:-unset}"

if [[ "$V34_TRAIN" == "1" ]]; then
  bash scripts/launch_v34_inpaint_blend.sh "$@"
else
  bash scripts/resume_v34_inference_v33ckpt.sh "$@"
fi
