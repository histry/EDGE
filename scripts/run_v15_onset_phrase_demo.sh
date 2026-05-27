#!/usr/bin/env bash
set -euo pipefail

# Run from EDGE repo root:
#   cd /home/disk/lsm/storage/EDGE
#   bash scripts/run_v15_onset_phrase_demo.sh

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# Physics-first retrieval. Increase beta or lower root/contact thresholds if
# boundary jumps remain visible.
export EDGE_ONSET_ALPHA_MUSIC=${EDGE_ONSET_ALPHA_MUSIC:-0.20}
export EDGE_ONSET_BETA_PHYSICS=${EDGE_ONSET_BETA_PHYSICS:-1.20}
export EDGE_ONSET_NOVELTY=${EDGE_ONSET_NOVELTY:-0.35}
export EDGE_ONSET_MAX_ROOT_Y_DELTA=${EDGE_ONSET_MAX_ROOT_Y_DELTA:-0.30}
export EDGE_ONSET_MAX_ROOT_SPEED_DELTA=${EDGE_ONSET_MAX_ROOT_SPEED_DELTA:-0.35}
export EDGE_ONSET_MAX_CONTACT_L1=${EDGE_ONSET_MAX_CONTACT_L1:-1.60}
export EDGE_ONSET_ROOT_DRIFT_KEEP=${EDGE_ONSET_ROOT_DRIFT_KEEP:-0.02}

DB=${DB:-data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz}
POOL=${POOL:-data/dunhuang_choreo_unit_rag/v15_physics_prior_pool.npz}
AUDIO_WAV=${AUDIO_WAV:-test_music_bank/dunhuangwu2.wav}
OUT_DIR=${OUT_DIR:-output/v15_onset_phrase}
OUT_NPY=${OUT_NPY:-$OUT_DIR/dw2_v15_onset_phrase_prior.npy}
PKL_DIR=${PKL_DIR:-$OUT_DIR/train_pkl}

mkdir -p "$OUT_DIR"

echo "[1/2] Build physics-aware prior pool"
python tools/build_physics_aware_prior_pool.py \
  --input "$DB" \
  --out "$POOL" \
  --min_activity 0.0002 \
  --max_root_radius 0.50

echo "[2/2] Compose onset-driven long phrase prior"
python generate_onset_phrase_prior.py \
  --pool "$POOL" \
  --audio_wav "$AUDIO_WAV" \
  --out "$OUT_NPY" \
  --length 150 \
  --fps 30 \
  --max_phrases 4 \
  --transition_tolerance 14 \
  --min_blend 6 \
  --max_blend 16 \
  --inplace \
  --export_pkl_dir "$PKL_DIR"

cat <<EOF

Done.
Prior NPY: $OUT_NPY
Report:    ${OUT_NPY%.npy}.report.json
PKL dir:   $PKL_DIR

Optional render if your repo has render_from_npy.py:
  python render_from_npy.py --input "$OUT_NPY" --output "${OUT_NPY%.npy}.mp4" --camera_mode fixed

Optional conservative fine-tuning on exported long-prior pkl:
  export EDGE_TRAIN_PROFILE=v3_unit_recon
  export EDGE_V3_UNIT_RECON=1
  export EDGE_X0_RECON_LOSS=1
  export EDGE_X0_RECON_LOSS_WEIGHT=0.45
  export EDGE_V3_DCT_KEEP=8
  export EDGE_V3_TEMPORAL_FEATURES=upper_torso
  python train.py --data_path "$PKL_DIR" --seq_len 150 --feature_type hybrid --audio_pairing_mode none --disable_traj_cond --exp_name v15_onset_phrase_recon
EOF
