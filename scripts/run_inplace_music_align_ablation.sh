#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

mkdir -p logs
mkdir -p output/inplace_music_align

# 必须已经在 edge 环境中运行
if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ CONDA_PREFIX is empty. Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"

echo "Using python: $PY"
"$PY" - <<'PY'
import sys
print("python =", sys.executable)
import torch, numpy
print("torch =", torch.__version__)
print("numpy =", numpy.__version__)
PY

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT="${CKPT:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
RAG_DB="${RAG_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"

GEN="generate_v10_choreo.py"
if [ -f generate_v11_choreo.py ]; then
  GEN="generate_v11_choreo.py"
fi

if [ ! -f "$GEN" ]; then
  echo "❌ Cannot find generate_v10_choreo.py or generate_v11_choreo.py"
  echo "Available generate files:"
  ls generate*.py || true
  exit 1
fi

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE="${END_POSE:-test_keyframes/demo_dyl002_end.npy}"

if [ ! -f "$CKPT" ]; then
  echo "❌ checkpoint not found: $CKPT"
  exit 1
fi

if [ ! -f "$START_POSE" ]; then
  echo "❌ start pose not found: $START_POSE"
  exit 1
fi

if [ ! -f "$END_POSE" ]; then
  echo "❌ end pose not found: $END_POSE"
  exit 1
fi

TRAJ_STATIC='0,0;0,0;0,0;0,0'
TRAJ_SMALL='0,0;0.05,0.04;-0.04,0.08;0,0.12'

export EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
export EDGE_AUDIO_DEVICE=cpu
export EDGE_RUN_MODE=formal
export EDGE_EXPERIMENT_PROFILE=v10

export EDGE_V10_RAG_DB="$RAG_DB"
export EDGE_V10_SEARCH_METHOD=beam
export EDGE_V10_BEAM_WIDTH=8
export EDGE_V10_JERK_PENALTY=1
export EDGE_V10_ADAPTIVE_PLANNER=1
export EDGE_V10_ADAPTIVE_NEAR_POSE_SCALE=3.0
export EDGE_V10_ADAPTIVE_NEAR_JERK_SCALE=0.1

export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_REQUIRED=0
export EDGE_UNIT_PRIOR_TEMPORAL=1
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_DCT_DECAY=soft_exp
export EDGE_UNIT_PRIOR_LOW_FREQ_K=4
export EDGE_UNIT_PRIOR_FEATURES=upper+torso
export EDGE_UNIT_PRIOR_STRENGTH=0.006

# 原地阶段先不要让 dynamic trajectory CFG 干扰 root
export EDGE_DYNAMIC_TRAJ_CFG=0

export EDGE_FORMAL_AUTO_EVAL=1
export EDGE_FORMAL_EVAL_REQUIRED=0

# 默认先跑三首音乐；调试时可以用：
# MUSIC_LIST="dunhuangwu2" BEAT_WEIGHTS="0 0.03" bash scripts/run_inplace_music_align_ablation.sh
MUSIC_LIST="${MUSIC_LIST:-dunhuangwu2 dunhuangwu3 dunhuangwu4}"
BEAT_WEIGHTS="${BEAT_WEIGHTS:-0 0.01 0.03 0.05}"
TRAJ_LIST="${TRAJ_LIST:-static small}"

for MUSIC_NAME in $MUSIC_LIST; do
  MUSIC="test_music_bank/${MUSIC_NAME}.wav"

  if [ ! -f "$MUSIC" ]; then
    echo "⚠️ music not found, skip: $MUSIC"
    continue
  fi

  for TRAJ_NAME in $TRAJ_LIST; do
    if [ "$TRAJ_NAME" = "static" ]; then
      TRAJ="$TRAJ_STATIC"
    elif [ "$TRAJ_NAME" = "small" ]; then
      TRAJ="$TRAJ_SMALL"
    else
      echo "⚠️ unknown TRAJ_NAME=$TRAJ_NAME, skip"
      continue
    fi

    for W in $BEAT_WEIGHTS; do
      export EDGE_BEAT_GUIDANCE=1
      export EDGE_BEAT_GUIDANCE_WEIGHT="$W"
      export EDGE_BEAT_GUIDANCE_TARGET=1.35
      export EDGE_BEAT_GUIDANCE_FEATURES=all

      OUT="output/inplace_music_align/${MUSIC_NAME}_${TRAJ_NAME}_beat${W}"

      echo ""
      echo "============================================================"
      echo "Running:"
      echo "  music = $MUSIC"
      echo "  traj  = $TRAJ_NAME"
      echo "  beat  = $W"
      echo "  ckpt  = $CKPT"
      echo "  out   = ${OUT}.npy"
      echo "============================================================"

      "$PY" "$GEN" \
        --checkpoint "$CKPT" \
        --music "$MUSIC" \
        --start_pose "$START_POSE" \
        --end_pose "$END_POSE" \
        --out "${OUT}.npy" \
        --feature_type hybrid \
        --trajectory "$TRAJ" \
        --sampler ddim \
        2>&1 | tee "logs/${MUSIC_NAME}_${TRAJ_NAME}_beat${W}.log"

      if [ -f scripts/validate_formal_run.py ]; then
        "$PY" scripts/validate_formal_run.py --prefix "$OUT" || true
      fi
    done
  done
done

echo ""
echo "✅ Finished inplace music alignment ablation."
