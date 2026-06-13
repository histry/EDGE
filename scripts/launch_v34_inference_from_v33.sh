#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
EDGE_ENV="${EDGE_ENV:-/home/disk/lsm/conda_envs/edge}"
cd "$ROOT"
export PATH="$EDGE_ENV/bin:$PATH"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export V26_INDEX_JSON="${V26_INDEX_JSON:-data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json}"
export V26_DURATION_INDEX_NPZ="${V26_DURATION_INDEX_NPZ:-data/v26_music_dominant_duration_index.npz}"
export V26_ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
export V26_V23_CKPT="${V26_V23_CKPT:-output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt}"
export V26_PLANNER_CKPT="${V26_PLANNER_CKPT:-$(cat output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt)}"
export V26_START_POSE="${V26_START_POSE:-data/canonical_dunhuang_start_pose.npy}"
export V27_HYPERBOLIC_CKPT="${V27_HYPERBOLIC_CKPT:-output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt}"
export V26_HIERARCHY_INDEX_NPZ="${V26_HIERARCHY_INDEX_NPZ:-output/v28_diffusion_retrain_musicclap_20260610_024828/v28_hyperbolic_hierarchical_event_index.npz}"
export V33_EVENT_CONTACT_CACHE="${V33_EVENT_CONTACT_CACHE:-data/v33_event_contact_cache.npz}"
export V34_DATASET="${V34_DATASET:-data/v33_transition_dataset.npz}"
export V34_LIBRARY_DIR="${V34_LIBRARY_DIR:-data/v34_event_library}"
export V34_INDEX_JSON="${V34_INDEX_JSON:-data/v34_shared_event_index.json}"
export V27_TRANSITION_DIFFUSION_CKPT="${V27_TRANSITION_DIFFUSION_CKPT:-output/v33_event_contact_20260611_204533/v32_contact_inr_training/checkpoints/best.pt}"

export V34_TRAIN=0
export V34_BUILD_EVENT_LIBRARY="${V34_BUILD_EVENT_LIBRARY:-1}"
export V34_FAIL_ON_UNSAFE_BOUNDARY="${V34_FAIL_ON_UNSAFE_BOUNDARY:-0}"
export RUN_ID="${RUN_ID:-v34_inference_v33ckpt_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
echo "$RUN_ROOT" > output/LATEST_V34_INFERENCE_LAUNCH.txt

bash scripts/run_v34_full_research.sh
