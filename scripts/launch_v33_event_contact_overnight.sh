#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"

export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

export V26_INDEX_JSON="${V26_INDEX_JSON:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json}"
export V26_DURATION_INDEX_NPZ="${V26_DURATION_INDEX_NPZ:-data/v26_music_dominant_duration_index.npz}"
export V26_ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
export V26_V23_CKPT="${V26_V23_CKPT:-output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt}"
export V26_PLANNER_CKPT="${V26_PLANNER_CKPT:-$(cat output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt)}"
export V26_START_POSE="${V26_START_POSE:-data/canonical_dunhuang_start_pose.npy}"
export V27_HYPERBOLIC_CKPT="${V27_HYPERBOLIC_CKPT:-output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt}"
export V26_HIERARCHY_INDEX_NPZ="${V26_HIERARCHY_INDEX_NPZ:-output/v28_diffusion_retrain_musicclap_20260610_024828/v28_hyperbolic_hierarchical_event_index.npz}"

export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
export V32_KEYS="${V32_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
export V32_SUPERVISION_MODE="${V32_SUPERVISION_MODE:-weak}"

# Event-level contact reconstruction.  Window-level relabeling is forbidden.
export V33_BUILD_EVENT_CONTACT_CACHE="${V33_BUILD_EVENT_CONTACT_CACHE:-1}"
export V33_CONTACT_TARGET_RATES="${V33_CONTACT_TARGET_RATES:-0.42,0.42,0.38,0.38}"
export V33_CONTACT_TRANSITION_PENALTY="${V33_CONTACT_TRANSITION_PENALTY:-1.40}"
export V33_CONTACT_MIN_RUN="${V33_CONTACT_MIN_RUN:-2}"
export V33_CONTACT_MAX_GAP="${V33_CONTACT_MAX_GAP:-2}"
export V33_MIN_CONTACT_RATE="${V33_MIN_CONTACT_RATE:-0.05}"
export V33_MAX_CONTACT_RATE="${V33_MAX_CONTACT_RATE:-0.75}"
export V33_MAX_ALL_FOUR_RATE="${V33_MAX_ALL_FOUR_RATE:-0.55}"
export V33_REQUIRE_OVERLAP_COMPARISONS="${V33_REQUIRE_OVERLAP_COMPARISONS:-1000}"

# Safe 4090 defaults.
export V32_AE_BATCH_SIZE="${V32_AE_BATCH_SIZE:-40}"
export V32_CONTACT_BATCH_SIZE="${V32_CONTACT_BATCH_SIZE:-32}"
export V32_LATENT_BATCH_SIZE="${V32_LATENT_BATCH_SIZE:-192}"
export V32_DECODED_BATCH_LIMIT="${V32_DECODED_BATCH_LIMIT:-8}"
export V32_AMP="${V32_AMP:-1}"

export V32_FOURIER_BANDS="${V32_FOURIER_BANDS:-5}"
export V32_ROTATION_RESIDUAL_SCALE="${V32_ROTATION_RESIDUAL_SCALE:-0.16}"
export V32_ROOT_Y_RESIDUAL_SCALE="${V32_ROOT_Y_RESIDUAL_SCALE:-0.045}"
export V32_CONTACT_LOGIT_SCALE="${V32_CONTACT_LOGIT_SCALE:-3.0}"

export V32_CANDIDATES="${V32_CANDIDATES:-8}"
export V32_GUIDANCE="${V32_GUIDANCE:-1.0}"
export V32_INR_TRUST="${V32_INR_TRUST:-0.35}"
export V32_ENABLE_EDGE_DAMPING="${V32_ENABLE_EDGE_DAMPING:-0}"
export V32_STRICT_LOCKED_WARP="${V32_STRICT_LOCKED_WARP:-1}"

export RUN_ID="${RUN_ID:-v33_event_contact_inr_overnight_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V33_OVERNIGHT_LAUNCH.txt
echo "$RUN_ROOT" > output/LATEST_V32_OVERNIGHT_LAUNCH.txt

required=(
  "$V26_INDEX_JSON"
  "$V26_DURATION_INDEX_NPZ"
  "$V26_ROUTER_CKPT"
  "$V26_V23_CKPT"
  "$V26_PLANNER_CKPT"
  "$V26_START_POSE"
  "$V27_HYPERBOLIC_CKPT"
  "$V26_HIERARCHY_INDEX_NPZ"
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || {
    echo "[PRECHECK ERROR] Missing $path" >&2
    exit 2
  }
done

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")
print("gpu", torch.cuda.get_device_name(0))
PY

nvidia-smi
bash scripts/run_v33_event_contact_inr_full.sh
