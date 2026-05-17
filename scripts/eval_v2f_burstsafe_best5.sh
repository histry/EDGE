#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate edge

export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_TRAJ_EVENT_COND=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0

EXP_NAME=${EXP_NAME:-stationary_v2f_best5_burstsafe_x0w030}
WEIGHTS=${WEIGHTS:-runs/train_nextgen/${EXP_NAME}/weights}
ASSET_BASE=${ASSET_BASE:-output/stationary_v2d_nostatic_best5_endpoint_eval_x0w025_freezeaware/e200}
OUT_ROOT=${OUT_ROOT:-output/stationary_v2f_best5_burstsafe_eval}
MUSIC=${MUSIC:-test_music_bank/dunhuangwu2.wav}
UNITS=${UNITS:-"55370 61437 69373 70219 49122"}
STEPS=${STEPS:-"50 100 150 200 250 300"}
mkdir -p "$OUT_ROOT"

for step in $STEPS; do
  CKPT="$WEIGHTS/train-${step}.pt"
  [[ -f "$CKPT" ]] || continue
  OUT="$OUT_ROOT/e${step}"
  mkdir -p "$OUT"

  for unit in $UNITS; do
    ASSET="$ASSET_BASE/unit_${unit}_assets"
    GT="$ASSET/unit_${unit}_gt.npy"
    [[ -f "$GT" ]] || { echo "missing GT: $GT"; continue; }
    python - <<PY
import numpy as np
from pathlib import Path
asset = Path("$ASSET")
gt = np.load(asset / "unit_${unit}_gt.npy")
for f in [11, 22, 34]:
    path = asset / f"unit_${unit}_mid_{f:03d}.npy"
    np.save(path, gt[f])
    print("saved", path)
PY

    python generate_controlled.py \
      --checkpoint "$CKPT" \
      --music "$MUSIC" \
      --start_pose "$ASSET/unit_${unit}_start.npy" \
      --end_pose "$ASSET/unit_${unit}_end.npy" \
      --mid_poses "$ASSET/unit_${unit}_mid_011.npy,$ASSET/unit_${unit}_mid_022.npy,$ASSET/unit_${unit}_mid_034.npy" \
      --mid_pose_frames 11,22,34 \
      --out "$OUT/unit_${unit}.npy" \
      --feature_type hybrid \
      --audio_dim 803 \
      --seq_len 45 \
      --num_frames 45 \
      --pose_space physical \
      --sampler ddim \
      --guidance_weight 1.0 \
      --endpoint_keyframe_strength 0.65 \
      --mid_keyframe_strength 0.06 \
      --infer_keyframe_width 3 \
      --disable_traj_cond \
      --no_tto \
      --trajectory "0,0;0,0" \
      --mixed_precision bf16 \
      2>&1 | tee "logs/eval_v2f_${EXP_NAME}_e${step}_unit_${unit}.log"

    mkdir -p "$OUT/render_fixed"
    python render_from_npy.py \
      --motion "$OUT/unit_${unit}.npy" \
      --audio "$MUSIC" \
      --output "$OUT/render_fixed/unit_${unit}_fixed.mp4" \
      --camera_mode fixed \
      2>&1 | tee "logs/render_v2f_${EXP_NAME}_e${step}_unit_${unit}.log"
  done

  python tools/diagnose_endpoint_collapse_units.py \
    --pred_dir "$OUT" \
    --out_csv "$OUT_ROOT/endpoint_collapse_diag_e${step}.csv" \
    2>&1 | tee "logs/endpoint_collapse_diag_v2f_${EXP_NAME}_e${step}.log"

  python tools/diagnose_bursty_jitter_units.py \
    --pred_dir "$OUT" \
    --out_csv "$OUT_ROOT/burst_jitter_diag_e${step}.csv" \
    2>&1 | tee "logs/burst_jitter_diag_v2f_${EXP_NAME}_e${step}.log"
done
