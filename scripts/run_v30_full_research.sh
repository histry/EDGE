#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

RUN_ID="${RUN_ID:-v30_top_research_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

echo "[START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"

# ------------------------------------------------------------------
# Publication strictness. Set V30_ALLOW_WEAK_SUPERVISION=1 only for
# engineering smoke tests, never for the final claimed model.
# ------------------------------------------------------------------
export V30_ALLOW_WEAK_SUPERVISION="${V30_ALLOW_WEAK_SUPERVISION:-0}"
SOURCE_MANIFEST="${V30_SOURCE_MANIFEST:-}"
PAIR_MANIFEST="${V30_PAIR_MANIFEST:-}"
ROUTER_DATA="${V30_ROUTER_DATA:-}"

if [[ "$V30_ALLOW_WEAK_SUPERVISION" != "1" ]]; then
  : "${SOURCE_MANIFEST:?Set V30_SOURCE_MANIFEST for real boundary supervision}"
  : "${PAIR_MANIFEST:?Set V30_PAIR_MANIFEST for explicit music-motion pairs}"
fi

ALIGN_DATA="$RUN_ROOT/v30_alignment_dataset.npz"
ALIGN_DIR="$RUN_ROOT/v30_geometric_alignment"
TRANSITION_RAW="$RUN_ROOT/v30_transition_dataset_raw.npz"
TRANSITION_DATA="$RUN_ROOT/v30_transition_dataset.npz"
TRANSITION_DIR="$RUN_ROOT/v30_continuous_inr_diffusion"
EVENT_INDEX_NPZ="$RUN_ROOT/v30_geometric_event_index.npz"
EVENT_INDEX_JSON="$RUN_ROOT/v30_geometric_event_index.json"

python -m py_compile \
  tools/v29_motion_geometry.py \
  tools/v30_continuous_inr.py \
  tools/v30_geometric_alignment.py \
  tools/build_v30_alignment_dataset.py \
  train_v30_geometric_alignment.py \
  tools/build_v30_geometric_event_index.py \
  tools/v30_deep_music_features.py \
  tools/build_v27_transition_diffusion_dataset.py \
  tools/enrich_v30_transition_music.py \
  train_v27_transition_diffusion.py \
  tools/v27_transition_diffusion.py \
  tools/schedule_v30_whole_song.py \
  tools/evaluate_v30_frequency_metrics.py

if [[ "${V30_BUILD_ALIGNMENT_DATA:-1}" == "1" ]]; then
  EXTRA=()
  [[ -n "$PAIR_MANIFEST" ]] && EXTRA+=(--pair_manifest "$PAIR_MANIFEST")
  [[ -n "$ROUTER_DATA" ]] && EXTRA+=(--router_data "$ROUTER_DATA")
  REQUIRED_EXPLICIT=0
  REQUIRED_CLAP=0
  if [[ "$V30_ALLOW_WEAK_SUPERVISION" != "1" ]]; then
    REQUIRED_EXPLICIT="${V30_REQUIRE_EXPLICIT_PAIRS:-1000}"
    REQUIRED_CLAP="${V30_REQUIRE_CLAP_PAIRS:-900}"
  fi
  python tools/build_v30_alignment_dataset.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_npz "$ALIGN_DATA" \
    "${EXTRA[@]}" \
    --require_explicit_pairs "$REQUIRED_EXPLICIT" \
    --require_clap_valid_pairs "$REQUIRED_CLAP" \
    --router_weight "${V30_ROUTER_WEAK_WEIGHT:-0.25}" \
    --seed "${V30_SEED:-20260610}"
fi

if [[ "${V30_TRAIN_ALIGNMENT:-1}" == "1" ]]; then
  python train_v30_geometric_alignment.py \
    --data "$ALIGN_DATA" \
    --out_dir "$ALIGN_DIR" \
    --epochs "${V30_ALIGNMENT_EPOCHS:-320}" \
    --batch_size "${V30_ALIGNMENT_BATCH_SIZE:-192}" \
    --hidden_dim "${V30_ALIGNMENT_HIDDEN:-256}" \
    --embed_dim "${V30_ALIGNMENT_EMBED_DIM:-32}" \
    --lr "${V30_ALIGNMENT_LR:-2e-4}" \
    --val_ratio "${V30_ALIGNMENT_VAL_RATIO:-0.12}" \
    --patience "${V30_ALIGNMENT_PATIENCE:-60}" \
    --seed "${V30_SEED:-20260610}"
  export V30_ALIGNMENT_CKPT
  V30_ALIGNMENT_CKPT="$(cat "$ALIGN_DIR/BEST_V30_ALIGNMENT_CKPT.txt")"
else
  : "${V30_ALIGNMENT_CKPT:?Set V30_ALIGNMENT_CKPT}"
fi

python tools/build_v30_geometric_event_index.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --alignment_ckpt "$V30_ALIGNMENT_CKPT" \
  --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-}" \
  --out_npz "$EVENT_INDEX_NPZ" \
  --out_json "$EVENT_INDEX_JSON"

export V26_HIERARCHY_INDEX_NPZ="$EVENT_INDEX_NPZ"

if [[ "${V30_BUILD_TRANSITION_DATA:-1}" == "1" ]]; then
  EXTRA=()
  [[ -n "$SOURCE_MANIFEST" ]] && EXTRA+=(--source_manifest "$SOURCE_MANIFEST")
  [[ -n "${V30_FULL_MOTION_ROOT:-}" ]] && EXTRA+=(--full_motion_root "$V30_FULL_MOTION_ROOT")
  [[ -n "${V30_EXTERNAL_PRIOR_NPZ:-}" ]] && EXTRA+=(--external_prior_npz "$V30_EXTERNAL_PRIOR_NPZ")
  REAL_COUNT=0
  REAL_RATIO=0
  if [[ "$V30_ALLOW_WEAK_SUPERVISION" != "1" ]]; then
    REAL_COUNT="${V30_REQUIRE_REAL_BOUNDARIES:-1000}"
    REAL_RATIO="${V30_REQUIRE_REAL_BOUNDARY_RATIO:-0.10}"
  fi
  python tools/build_v27_transition_diffusion_dataset.py \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_npz "$TRANSITION_RAW" \
    "${EXTRA[@]}" \
    --max_len "${V30_TRANSITION_MAX_LEN:-120}" \
    --min_len "${V30_TRANSITION_MIN_LEN:-8}" \
    --samples_per_event "${V30_INTRA_EVENT_SAMPLES:-5}" \
    --real_masks_per_boundary "${V30_REAL_MASKS_PER_BOUNDARY:-4}" \
    --source_pairs_per_event "${V30_SOURCE_PAIRS_PER_EVENT:-1.0}" \
    --pseudo_pairs_per_event "${V30_PSEUDO_PAIRS_PER_EVENT:-0.15}" \
    --allow_synthetic_adjacent "${V30_ALLOW_SYNTHETIC_ADJACENT:-1}" \
    --require_real_boundary_count "$REAL_COUNT" \
    --require_real_boundary_ratio "$REAL_RATIO" \
    --seed "${V30_SEED:-20260610}"

  if [[ -n "$SOURCE_MANIFEST" ]]; then
    REQUIRED_MUSIC_COUNT=0
    REQUIRED_MUSIC_RATIO=0
    if [[ "$V30_ALLOW_WEAK_SUPERVISION" != "1" ]]; then
      REQUIRED_MUSIC_COUNT="${V30_REQUIRE_REAL_MUSIC_SAMPLES:-1000}"
      REQUIRED_MUSIC_RATIO="${V30_REQUIRE_REAL_MUSIC_RATIO:-0.10}"
    fi
    python tools/enrich_v30_transition_music.py \
      --input_npz "$TRANSITION_RAW" \
      --source_manifest "$SOURCE_MANIFEST" \
      --alignment_ckpt "$V30_ALIGNMENT_CKPT" \
      --out_npz "$TRANSITION_DATA" \
      --audio_root "${V30_AUDIO_ROOT:-}" \
      --require_real_music_count "$REQUIRED_MUSIC_COUNT" \
      --require_real_music_ratio "$REQUIRED_MUSIC_RATIO"
  else
    cp "$TRANSITION_RAW" "$TRANSITION_DATA"
  fi
fi

if [[ "${V30_TRAIN_TRANSITION:-1}" == "1" ]]; then
  python train_v27_transition_diffusion.py \
    --data "$TRANSITION_DATA" \
    --out_dir "$TRANSITION_DIR" \
    --stage all \
    --ae_epochs "${V30_AE_EPOCHS:-220}" \
    --diffusion_epochs "${V30_DIFFUSION_EPOCHS:-320}" \
    --batch_size "${V30_AE_BATCH_SIZE:-48}" \
    --latent_batch_size "${V30_LATENT_BATCH_SIZE:-256}" \
    --latent_dim "${V30_LATENT_DIM:-128}" \
    --condition_dim "${V30_CONDITION_DIM:-256}" \
    --encoder_hidden "${V30_ENCODER_HIDDEN:-320}" \
    --inr_hidden "${V30_INR_HIDDEN:-320}" \
    --inr_layers "${V30_INR_LAYERS:-5}" \
    --fourier_bands "${V30_FOURIER_BANDS:-10}" \
    --diffusion_hidden "${V30_DIFFUSION_HIDDEN:-512}" \
    --diffusion_blocks "${V30_DIFFUSION_BLOCKS:-6}" \
    --diffusion_steps "${V30_DIFFUSION_STEPS:-100}" \
    --condition_dropout "${V30_CONDITION_DROPOUT:-0.10}" \
    --amp "${V30_AMP:-1}" \
    --seed "${V30_SEED:-20260610}"
  export V27_TRANSITION_DIFFUSION_CKPT
  V27_TRANSITION_DIFFUSION_CKPT="$(cat "$TRANSITION_DIR/BEST_V30_CONTINUOUS_INR_DIFFUSION_CKPT.txt")"
else
  : "${V27_TRANSITION_DIFFUSION_CKPT:?Set V27_TRANSITION_DIFFUSION_CKPT}"
fi

if [[ "${V30_GENERATE:-1}" == "1" ]]; then
  export V26_OUT_DIR="${V26_OUT_DIR:-$RUN_ROOT/whole_song_v30}"
  export V27_TRANSITION_DIFFUSION=1
  export V27_TRANSITION_DIFFUSION_BLEND="${V30_INR_BLEND:-0.85}"
  export V27_TRANSITION_DIFFUSION_STEPS="${V30_INFERENCE_STEPS:-40}"
  export V30_ALIGNMENT_DEVICE="${V30_ALIGNMENT_DEVICE:-cuda:0}"
  export V30_CROSSMODAL_RETRIEVAL_WEIGHT="${V30_CROSSMODAL_RETRIEVAL_WEIGHT:-0.35}"
  export V30_LATENT_GUIDANCE="${V30_LATENT_GUIDANCE:-1.20}"
  export V30_INR_BLEND="${V30_INR_BLEND:-0.85}"
  bash scripts/run_v30_whole_song.sh
fi

if [[ "${V30_EVALUATE:-1}" == "1" ]]; then
  IFS=';' read -ra KEYS <<< "${V30_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
  for key in "${KEYS[@]}"; do
    MOTION="$V26_OUT_DIR/${key}_v26.npy"
    REPORT="$V26_OUT_DIR/${key}_v26.schedule_report.json"
    AUDIO="test_music_bank/${key}.wav"

    python tools/evaluate_v26_long_dance.py \
      --motion "$MOTION" \
      --schedule_report "$REPORT" \
      --out_json "$V26_OUT_DIR/${key}_v30.long_dance.json"

    python tools/evaluate_v27_public_metrics.py \
      --motion "$MOTION" \
      --audio "$AUDIO" \
      --index_json "$V26_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$V26_OUT_DIR/${key}_v30.public_metrics.json"

    python tools/evaluate_v30_frequency_metrics.py \
      --motion "$MOTION" \
      --schedule_report "$REPORT" \
      --index_json "$V26_INDEX_JSON" \
      --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
      --out_json "$V26_OUT_DIR/${key}_v30.frequency_foot.json" \
      --out_png "$V26_OUT_DIR/${key}_v30.dct_spectrum.png"

    python render_from_npy.py \
      --motion "$MOTION" \
      --audio "$AUDIO" \
      --output "$V26_OUT_DIR/${key}_v30_scientific_fixed.mp4" \
      --camera_mode fixed \
      --render_smooth_window 1
  done
fi

echo "$RUN_ROOT" > output/LATEST_V30_TOP_RESEARCH_RUN.txt
echo "[DONE] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
