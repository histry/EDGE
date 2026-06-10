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
: "${V31_SOURCE_MANIFEST:?Set V31_SOURCE_MANIFEST}"
: "${V31_PAIR_MANIFEST:?Set V31_PAIR_MANIFEST}"

RUN_ID="${RUN_ID:-v31_safe_research_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-output/$RUN_ID}"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/run.log") 2>&1

ALIGN_DATA="$RUN_ROOT/alignment_dataset.npz"
ALIGN_DIR="$RUN_ROOT/geometric_alignment"
EVENT_INDEX_NPZ="$RUN_ROOT/geometric_event_index.npz"
EVENT_INDEX_JSON="$RUN_ROOT/geometric_event_index.json"
TRANSITION_RAW="$RUN_ROOT/transition_dataset_raw.npz"
TRANSITION_DATA="$RUN_ROOT/transition_dataset_real_music.npz"
TRANSITION_DIR="$RUN_ROOT/v31_coefficient_diffusion"

echo "[START] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"

python -m py_compile \
  tools/v29_motion_geometry.py \
  tools/v31_bandlimited_transition.py \
  tools/v31_transition_quality.py \
  tools/v27_transition_diffusion.py \
  tools/build_v27_transition_diffusion_dataset.py \
  train_v27_transition_diffusion.py \
  tools/v30_geometric_alignment.py \
  tools/v30_deep_music_features.py \
  tools/build_v30_alignment_dataset.py \
  train_v30_geometric_alignment.py \
  tools/build_v30_geometric_event_index.py \
  tools/enrich_v30_transition_music.py \
  tools/schedule_v31_whole_song.py \
  tools/evaluate_v30_frequency_metrics.py \
  tools/evaluate_v31_retrieval_geometry.py \
  tools/summarize_v31_transition_gate.py

# 1. Explicit CLAP-motion alignment. The old fixed-random CLAP projection is
# never used as the learned retrieval representation.
python tools/build_v30_alignment_dataset.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --out_npz "$ALIGN_DATA" \
  --pair_manifest "$V31_PAIR_MANIFEST" \
  --require_explicit_pairs "${V31_REQUIRE_EXPLICIT_PAIRS:-1000}" \
  --require_clap_valid_pairs "${V31_REQUIRE_CLAP_PAIRS:-900}" \
  --seed "${V31_SEED:-20260610}"

python train_v30_geometric_alignment.py \
  --data "$ALIGN_DATA" \
  --out_dir "$ALIGN_DIR" \
  --epochs "${V31_ALIGNMENT_EPOCHS:-280}" \
  --batch_size "${V31_ALIGNMENT_BATCH_SIZE:-192}" \
  --hidden_dim "${V31_ALIGNMENT_HIDDEN:-256}" \
  --embed_dim "${V31_ALIGNMENT_EMBED_DIM:-32}" \
  --lr "${V31_ALIGNMENT_LR:-2e-4}" \
  --val_ratio "${V31_ALIGNMENT_VAL_RATIO:-0.12}" \
  --patience "${V31_ALIGNMENT_PATIENCE:-50}" \
  --seed "${V31_SEED:-20260610}"

export V30_ALIGNMENT_CKPT
V30_ALIGNMENT_CKPT="$(cat "$ALIGN_DIR/BEST_V30_ALIGNMENT_CKPT.txt")"

python tools/build_v30_geometric_event_index.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --alignment_ckpt "$V30_ALIGNMENT_CKPT" \
  --hyperbolic_ckpt "${V27_HYPERBOLIC_CKPT:-}" \
  --out_npz "$EVENT_INDEX_NPZ" \
  --out_json "$EVENT_INDEX_JSON"

export V26_HIERARCHY_INDEX_NPZ="$EVENT_INDEX_NPZ"

# 2. Audit geometry. It remains disabled unless the held-out result is
# explicitly convincing.
python tools/evaluate_v31_retrieval_geometry.py \
  --pair_manifest "$V31_PAIR_MANIFEST" \
  --event_index_npz "$EVENT_INDEX_NPZ" \
  --alignment_ckpt "$V30_ALIGNMENT_CKPT" \
  --out_json "$RUN_ROOT/retrieval_geometry_audit.json" \
  --seed "${V31_SEED:-20260610}"

export V31_ENABLE_GEOMETRIC_RETRIEVAL="${V31_ENABLE_GEOMETRIC_RETRIEVAL:-0}"
export V31_GEOMETRIC_RETRIEVAL_WEIGHT="${V31_GEOMETRIC_RETRIEVAL_WEIGHT:-0.0}"

# 3. Real full-sequence boundary supervision. Synthetic bridges and random
# pseudo pairs are disabled for the main model.
python tools/build_v27_transition_diffusion_dataset.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --source_manifest "$V31_SOURCE_MANIFEST" \
  --full_motion_root "${V31_FULL_MOTION_ROOT:-}" \
  --out_npz "$TRANSITION_RAW" \
  --max_len "${V31_MAX_LEN:-96}" \
  --min_len "${V31_MIN_LEN:-8}" \
  --samples_per_event "${V31_INTRA_EVENT_SAMPLES:-4}" \
  --real_masks_per_boundary "${V31_MASKS_PER_BOUNDARY:-4}" \
  --source_pairs_per_event "${V31_SOURCE_PAIRS_PER_EVENT:-1.0}" \
  --pseudo_pairs_per_event 0.0 \
  --allow_synthetic_adjacent 0 \
  --require_real_boundary_count "${V31_REQUIRE_REAL_BOUNDARY_SAMPLES:-1000}" \
  --require_real_boundary_ratio "${V31_REQUIRE_REAL_BOUNDARY_RATIO:-0.10}" \
  --require_unique_real_boundary_count "${V31_REQUIRE_UNIQUE_BOUNDARIES:-250}" \
  --seed "${V31_SEED:-20260610}"

python tools/enrich_v30_transition_music.py \
  --input_npz "$TRANSITION_RAW" \
  --source_manifest "$V31_SOURCE_MANIFEST" \
  --alignment_ckpt "$V30_ALIGNMENT_CKPT" \
  --out_npz "$TRANSITION_DATA" \
  --audio_root "${V31_AUDIO_ROOT:-}" \
  --require_real_music_count "${V31_REQUIRE_REAL_MUSIC_SAMPLES:-1000}" \
  --require_real_music_ratio "${V31_REQUIRE_REAL_MUSIC_RATIO:-0.10}"

# 4. Band-limited coefficient diffusion. No VAE/SIREN stage.
python train_v27_transition_diffusion.py \
  --data "$TRANSITION_DATA" \
  --out_dir "$TRANSITION_DIR" \
  --epochs "${V31_EPOCHS:-320}" \
  --batch_size "${V31_BATCH_SIZE:-128}" \
  --fit_batch_size "${V31_FIT_BATCH_SIZE:-64}" \
  --basis_count "${V31_BASIS_COUNT:-6}" \
  --pca_dim "${V31_PCA_DIM:-96}" \
  --condition_dim "${V31_CONDITION_DIM:-256}" \
  --hidden_dim "${V31_HIDDEN_DIM:-384}" \
  --diffusion_blocks "${V31_DIFFUSION_BLOCKS:-6}" \
  --diffusion_steps "${V31_DIFFUSION_STEPS:-100}" \
  --rotation_residual_cap "${V31_ROTATION_RESIDUAL_CAP:-0.16}" \
  --root_y_residual_cap "${V31_ROOT_Y_RESIDUAL_CAP:-0.045}" \
  --ridge "${V31_RIDGE:-0.002}" \
  --max_fit_rmse "${V31_MAX_FIT_RMSE:-0.14}" \
  --include_synthetic 0 \
  --weighted_sampling 1 \
  --lr "${V31_LR:-1.8e-4}" \
  --val_ratio "${V31_VAL_RATIO:-0.12}" \
  --condition_dropout "${V31_CONDITION_DROPOUT:-0.10}" \
  --decoded_weight "${V31_DECODED_WEIGHT:-0.10}" \
  --decoded_batch_limit "${V31_DECODED_BATCH_LIMIT:-12}" \
  --patience "${V31_PATIENCE:-60}" \
  --amp "${V31_AMP:-1}" \
  --seed "${V31_SEED:-20260610}"

export V27_TRANSITION_DIFFUSION_CKPT
V27_TRANSITION_DIFFUSION_CKPT="$(
  cat "$TRANSITION_DIR/BEST_V31_TRANSITION_CKPT.txt"
)"

export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
unset V26_MUSIC_GLOB || true
export V31_KEYS="${V31_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"

# 5. Deterministic C2 baseline.
if [[ "${V31_RUN_C2_BASELINE:-1}" == "1" ]]; then
  export V26_OUT_DIR="$RUN_ROOT/c2_baseline"
  export V27_TRANSITION_DIFFUSION=0
  bash scripts/run_v31_whole_song.sh
fi

# 6. Risk-gated learned residual.
export V26_OUT_DIR="$RUN_ROOT/v31_safe"
export V27_TRANSITION_DIFFUSION=1
export V31_CANDIDATES="${V31_CANDIDATES:-6}"
export V31_GUIDANCE="${V31_GUIDANCE:-1.0}"
export V31_RESIDUAL_TRUST="${V31_RESIDUAL_TRUST:-0.20}"
export V31_INFERENCE_STEPS="${V31_INFERENCE_STEPS:-36}"
bash scripts/run_v31_whole_song.sh

# 7. Evaluation.
IFS=';' read -ra KEYS <<< "$V31_KEYS"
for key in "${KEYS[@]}"; do
  MOTION="$V26_OUT_DIR/${key}_v26.npy"
  REPORT="$V26_OUT_DIR/${key}_v26.schedule_report.json"
  AUDIO="test_music_bank/${key}.wav"

  python tools/evaluate_v26_long_dance.py \
    --motion "$MOTION" \
    --schedule_report "$REPORT" \
    --out_json "$V26_OUT_DIR/${key}_v31.long_dance.json"

  python tools/evaluate_v27_public_metrics.py \
    --motion "$MOTION" \
    --audio "$AUDIO" \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_json "$V26_OUT_DIR/${key}_v31.public_metrics.json"

  python tools/evaluate_v30_frequency_metrics.py \
    --motion "$MOTION" \
    --schedule_report "$REPORT" \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_json "$V26_OUT_DIR/${key}_v31.frequency_foot.json" \
    --out_png "$V26_OUT_DIR/${key}_v31.dct_spectrum.png"

  python tools/diagnose_v29_jitter.py \
    --motion "$MOTION" \
    --out_json "$V26_OUT_DIR/${key}_v31.jitter.json"

  python render_from_npy.py \
    --motion "$MOTION" \
    --audio "$AUDIO" \
    --output "$V26_OUT_DIR/${key}_v31_scientific_fixed.mp4" \
    --camera_mode fixed \
    --render_smooth_window 1
done

python tools/summarize_v31_transition_gate.py \
  --report_glob "$V26_OUT_DIR/*_v26.schedule_report.json" \
  --out_json "$RUN_ROOT/v31_transition_gate_summary.json"

echo "$RUN_ROOT" > output/LATEST_V31_SAFE_RESEARCH_RUN.txt
echo "[DONE] $(date)"
echo "[RUN_ROOT] $RUN_ROOT"
