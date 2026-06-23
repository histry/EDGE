#!/usr/bin/env bash
set -euo pipefail

# Pure-inference V34 rhythm repair launcher.
#
# Use this after Dense Boundary has already removed hard physical discontinuity
# but the generated dance collapses into "quick pose change + long hold" rhythm
# degeneration.  It reuses the trained transition checkpoint and only changes
# retrieval-time search costs.

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

export V34_TRAIN=0
export V34_BUILD_EVENT_LIBRARY="${V34_BUILD_EVENT_LIBRARY:-0}"
export V34_OVERWRITE_EVENT_LIBRARY="${V34_OVERWRITE_EVENT_LIBRARY:-0}"

# Keep Dense Boundary active, but reduce its dominance so the solver does not
# prefer low-energy safe snippets over musically alive snippets.
export V34_BOUNDARY_COMPAT="${V34_BOUNDARY_COMPAT:-1}"
export V34_COMPAT_HARD_PRUNE="${V34_COMPAT_HARD_PRUNE:-1}"
export V34_BOUNDARY_COMPAT_WEIGHT="${V34_BOUNDARY_COMPAT_WEIGHT:-1.15}"
export V34_COMPAT_DENSE_SCORE="${V34_COMPAT_DENSE_SCORE:-1}"
export V34_COMPAT_DENSE_POWER="${V34_COMPAT_DENSE_POWER:-2.0}"
export V34_COMPAT_DENSE_CAP="${V34_COMPAT_DENSE_CAP:-4.0}"

# Dynamic relaxation remains enabled; otherwise the low-resource graph can
# deadlock when rhythm repair rejects too many static paths.
export V34_RELAX_CONSTRAINTS_ON_EMPTY="${V34_RELAX_CONSTRAINTS_ON_EMPTY:-1}"
export V34_RELAX_COMPAT_ON_EMPTY="${V34_RELAX_COMPAT_ON_EMPTY:-1}"
export V34_RELAX_SEMANTIC_ON_EMPTY="${V34_RELAX_SEMANTIC_ON_EMPTY:-1}"
export V34_RELAX_RESCUE_TOP_K="${V34_RELAX_RESCUE_TOP_K:-768}"
export V34_RELAX_COMPAT_PENALTY_WEIGHT="${V34_RELAX_COMPAT_PENALTY_WEIGHT:-5.00}"
export V34_RELAX_SEMANTIC_PENALTY_WEIGHT="${V34_RELAX_SEMANTIC_PENALTY_WEIGHT:-4.50}"
export V34_RELAX_CONTACT_PENALTY_WEIGHT="${V34_RELAX_CONTACT_PENALTY_WEIGHT:-2.00}"

# Anti-collapse rhythm field.  These parameters target the diagnosed failure:
# high first-second energy ratio, low tail energy, and repeated static tags.
export V34_RHYTHM_DEGRADATION_PENALTY="${V34_RHYTHM_DEGRADATION_PENALTY:-1}"
export V34_RHYTHM_WEIGHT="${V34_RHYTHM_WEIGHT:-1.0}"
export V34_HOLD_PENALTY_WEIGHT="${V34_HOLD_PENALTY_WEIGHT:-3.50}"
export V34_STREAK_PENALTY_WEIGHT="${V34_STREAK_PENALTY_WEIGHT:-2.00}"
export V34_DENSITY_PENALTY_WEIGHT="${V34_DENSITY_PENALTY_WEIGHT:-4.00}"
export V34_HOLD_FIRST1S_RATIO_LIMIT="${V34_HOLD_FIRST1S_RATIO_LIMIT:-0.65}"
export V34_HOLD_TAIL_ENERGY_LIMIT="${V34_HOLD_TAIL_ENERGY_LIMIT:-0.020}"
export V34_MIN_SLOT_MEAN_ENERGY="${V34_MIN_SLOT_MEAN_ENERGY:-0.015}"
export V34_DENSITY_SLOT_DURATION_MIN_SEC="${V34_DENSITY_SLOT_DURATION_MIN_SEC:-1.50}"
export V34_STATIC_STREAK_ALLOW="${V34_STATIC_STREAK_ALLOW:-2}"
export V34_STATIC_EVENT_TAGS="${V34_STATIC_EVENT_TAGS:-pose_hold,calm_flow,neutral_flow}"

export V34_WARP_HARD_PRUNE="${V34_WARP_HARD_PRUNE:-1}"
export V34_WARP_MIN="${V34_WARP_MIN:-0.82}"
export V34_WARP_MAX="${V34_WARP_MAX:-1.30}"
export V34_WARP_RELAX_ON_EMPTY="${V34_WARP_RELAX_ON_EMPTY:-1}"
export V34_WARP_RELAX_MIN="${V34_WARP_RELAX_MIN:-0.82}"
export V34_WARP_RELAX_MAX="${V34_WARP_RELAX_MAX:-1.30}"
export V34_WARP_PREFILTER_TOP_K="${V34_WARP_PREFILTER_TOP_K:-1024}"

export V34_BOUNDARY_INPAINT="${V34_BOUNDARY_INPAINT:-1}"
export V34_INPAINT_ON_RELAXED_CONSTRAINT="${V34_INPAINT_ON_RELAXED_CONSTRAINT:-1}"
export V34_LATENT_SNIPPET_BLEND="${V34_LATENT_SNIPPET_BLEND:-1}"
export V34_USE_GPU_RETRIEVAL="${V34_USE_GPU_RETRIEVAL:-1}"
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export V34_DEFER_UNSAFE_BOUNDARY="${V34_DEFER_UNSAFE_BOUNDARY:-1}"
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE="${V34_HANDSHAKE_FALLBACK_ON_UNSAFE:-1}"

if [[ -z "${RUN_ID:-}" ]]; then
  export RUN_ID="v34_rhythm_repair_infer_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_RHYTHM_REPAIR.txt

echo "[V34 RHYTHM REPAIR] RUN_ROOT=$RUN_ROOT"
echo "[V34 RHYTHM REPAIR] V34_TRAIN=$V34_TRAIN"
echo "[V34 RHYTHM REPAIR] boundary_weight=$V34_BOUNDARY_COMPAT_WEIGHT"
echo "[V34 RHYTHM REPAIR] rhythm_weight=$V34_RHYTHM_WEIGHT"

bash scripts/resume_v34_inference_v33ckpt.sh "$@"
