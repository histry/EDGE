#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

RUN_ID="${RUN_ID:-v29_temporal_so3_transition_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

echo "[START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"

export V29_REBUILD_TRANSITION_DATASET="${V29_REBUILD_TRANSITION_DATASET:-1}"
export V29_RETRAIN_TRANSITION="${V29_RETRAIN_TRANSITION:-1}"
export V29_GENERATE="${V29_GENERATE:-1}"
export V29_EVALUATE="${V29_EVALUATE:-1}"
export V29_RENDER="${V29_RENDER:-1}"

DATASET="${V29_TRANSITION_DATASET:-$RUN_ROOT/v29_transition_dataset.npz}"
TRAIN_DIR="${V29_TRANSITION_TRAIN_DIR:-$RUN_ROOT/v29_transition_diffusion}"

python -m py_compile \
  tools/v29_motion_geometry.py \
  tools/v27_transition_diffusion.py \
  tools/build_v27_transition_diffusion_dataset.py \
  train_v27_transition_diffusion.py \
  tools/schedule_v29_whole_song.py \
  tools/evaluate_v26_long_dance.py \
  tools/evaluate_v27_public_metrics.py \
  tools/diagnose_v29_jitter.py \
  render_from_npy.py

if [[ "$V29_REBUILD_TRANSITION_DATASET" == "1" ]]; then
  python tools/build_v27_transition_diffusion_dataset.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_npz "$DATASET" \
    --max_len "${V29_DATA_MAX_LEN:-96}" \
    --min_len "${V29_DATA_MIN_LEN:-8}" \
    --samples_per_event "${V29_SAMPLES_PER_EVENT:-6}" \
    --source_pairs_per_event "${V29_SOURCE_PAIRS_PER_EVENT:-1.0}" \
    --pseudo_pairs_per_event "${V29_PSEUDO_PAIRS_PER_EVENT:-0.45}" \
    --condition_dropout "${V29_CONDITION_DROPOUT:-0.08}" \
    --max_source_gap "${V29_MAX_SOURCE_GAP:-96}" \
    --pseudo_max_pose_deg "${V29_PSEUDO_MAX_POSE_DEG:-42}" \
    --pseudo_max_velocity_deg_s "${V29_PSEUDO_MAX_VELOCITY_DEG_S:-260}" \
    --pseudo_max_root_y "${V29_PSEUDO_MAX_ROOT_Y:-0.18}" \
    --pseudo_max_contact_jump "${V29_PSEUDO_MAX_CONTACT_JUMP:-0.75}" \
    --seed "${V29_SEED:-20260610}"
fi

if [[ "$V29_RETRAIN_TRANSITION" == "1" ]]; then
  python train_v27_transition_diffusion.py \
    --data "$DATASET" \
    --out_dir "$TRAIN_DIR" \
    --epochs "${V29_EPOCHS:-420}" \
    --batch_size "${V29_BATCH_SIZE:-64}" \
    --hidden_dim "${V29_HIDDEN_DIM:-384}" \
    --num_blocks "${V29_NUM_BLOCKS:-10}" \
    --num_heads "${V29_NUM_HEADS:-8}" \
    --dropout "${V29_DROPOUT:-0.08}" \
    --diffusion_steps "${V29_DIFFUSION_TRAIN_STEPS:-100}" \
    --lr "${V29_LR:-1.5e-4}" \
    --weight_decay "${V29_WEIGHT_DECAY:-1e-4}" \
    --val_ratio "${V29_VAL_RATIO:-0.12}" \
    --patience "${V29_PATIENCE:-70}" \
    --ema_decay "${V29_EMA_DECAY:-0.999}" \
    --num_workers "${V29_NUM_WORKERS:-2}" \
    --amp "${V29_AMP:-1}" \
    --seed "${V29_SEED:-20260610}"
  export V27_TRANSITION_DIFFUSION_CKPT
  V27_TRANSITION_DIFFUSION_CKPT="$(cat "$TRAIN_DIR/BEST_V29_TRANSITION_DIFFUSION_CKPT.txt")"
else
  : "${V27_TRANSITION_DIFFUSION_CKPT:?Set checkpoint when V29_RETRAIN_TRANSITION=0}"
fi

if [[ "$V29_GENERATE" == "1" ]]; then
  export V26_OUT_DIR="${V26_OUT_DIR:-$RUN_ROOT/three_music_v29}"
  export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
  unset V26_MUSIC_GLOB
  export V27_TRANSITION_DIFFUSION="${V29_ENABLE_TRANSITION_DIFFUSION:-1}"
  export V27_TRANSITION_DIFFUSION_BLEND="${V29_TRANSITION_BLEND:-0.18}"
  export V27_TRANSITION_DIFFUSION_STEPS="${V29_TRANSITION_INFER_STEPS:-32}"
  export V29_TRANSITION_NOISE_STRENGTH="${V29_TRANSITION_NOISE_STRENGTH:-0.55}"
  export V29_TRANSITION_BLEND_POWER="${V29_TRANSITION_BLEND_POWER:-2.0}"
  export V29_TRANSITION_FILTER_WINDOW="${V29_TRANSITION_FILTER_WINDOW:-5}"
  export V29_TRANSITION_FILTER_STRENGTH="${V29_TRANSITION_FILTER_STRENGTH:-0.20}"
  export V26_EDGE_DAMPING_FRAMES="${V29_EDGE_DAMPING_FRAMES:-4}"
  export V26_EDGE_DAMPING_STRENGTH="${V29_EDGE_DAMPING_STRENGTH:-0.25}"

  bash scripts/run_v29_whole_song.sh
fi

if [[ "$V29_EVALUATE" == "1" ]]; then
  IFS=';' read -ra KEYS <<< "${V29_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
  for key in "${KEYS[@]}"; do
    MOTION="$V26_OUT_DIR/${key}_v26.npy"
    REPORT="$V26_OUT_DIR/${key}_v26.schedule_report.json"
    AUDIO="test_music_bank/${key}.wav"

    python tools/evaluate_v26_long_dance.py \
      --motion "$MOTION" \
      --schedule_report "$REPORT" \
      --out_json "$V26_OUT_DIR/${key}_v29.evaluation.json"

    python tools/evaluate_v27_public_metrics.py \
      --motion "$MOTION" \
      --audio "$AUDIO" \
      --index_json "$V26_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$V26_OUT_DIR/${key}_v29.public_metrics.json"

    python tools/diagnose_v29_jitter.py \
      --motion "$MOTION" \
      --out_json "$V26_OUT_DIR/${key}_v29.jitter.json"

    if [[ "$V29_RENDER" == "1" ]]; then
      # Scientific copy: no render-only smoothing.
      python render_from_npy.py \
        --motion "$MOTION" \
        --audio "$AUDIO" \
        --output "$V26_OUT_DIR/${key}_v29_scientific_fixed.mp4" \
        --camera_mode fixed \
        --render_smooth_window 1

      # Optional display copy. This must never replace scientific metrics.
      python render_from_npy.py \
        --motion "$MOTION" \
        --audio "$AUDIO" \
        --output "$V26_OUT_DIR/${key}_v29_display_fixed.mp4" \
        --camera_mode fixed \
        --render_smooth_window "${V29_DISPLAY_SMOOTH_WINDOW:-3}"
    fi
  done
fi

echo "$RUN_ROOT" > output/LATEST_V29_RESEARCH_RUN.txt
echo "[DONE] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
