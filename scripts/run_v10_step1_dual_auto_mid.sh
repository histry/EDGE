#!/usr/bin/env bash
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

: "${CHECKPOINT:=/home/disk/lsm/storage/EDGE/runs/train_stage45/v9_rag_summary_token_e4/weights/train-4.pt}"
if [ ! -f "$CHECKPOINT" ]; then
  CHECKPOINT="/home/disk/lsm/storage/EDGE/runs/train_stage45/v8b_content_stage2_smooth/weights/train-4.pt"
fi

: "${EDGE_V10_RAG_DB:=/home/disk/lsm/storage/EDGE/data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"
: "${MUSIC:=/home/disk/lsm/storage/EDGE/test_music_bank/dunhuangwu2.wav}"
: "${START_POSE:=/home/disk/lsm/storage/EDGE/test_keyframes/demo_dyl002_start.npy}"
: "${END_POSE:=/home/disk/lsm/storage/EDGE/test_keyframes/demo_dyl002_end.npy}"
: "${TRAJ:=0,0;0.5,0.7;-0.3,1.2;0,1.6}"

export EDGE_V10_RAG_DB
export EDGE_UNIT_SOFT_PRIOR="${EDGE_UNIT_SOFT_PRIOR:-1}"
export EDGE_UNIT_PRIOR_DCT="${EDGE_UNIT_PRIOR_DCT:-1}"
export EDGE_UNIT_PRIOR_LOW_FREQ_K="${EDGE_UNIT_PRIOR_LOW_FREQ_K:-4}"
export EDGE_UNIT_PRIOR_FEATURES="${EDGE_UNIT_PRIOR_FEATURES:-upper}"
export EDGE_UNIT_PRIOR_STRENGTH="${EDGE_UNIT_PRIOR_STRENGTH:-0.012}"
export EDGE_UNIT_PRIOR_MAX_LEN="${EDGE_UNIT_PRIOR_MAX_LEN:-45}"
export EDGE_ENABLE_RAG_SUMMARY_TOKEN="${EDGE_ENABLE_RAG_SUMMARY_TOKEN:-1}"
export EDGE_RAG_SUMMARY_DIM="${EDGE_RAG_SUMMARY_DIM:-7}"
export EDGE_RAG_SUMMARY_BLEND_RADIUS="${EDGE_RAG_SUMMARY_BLEND_RADIUS:-18}"
export EDGE_RAG_SUMMARY_MODE="${EDGE_RAG_SUMMARY_MODE:-mean}"
mkdir -p output/v10_eval

export EDGE_V10_MODE=dual_auto_mid
export EDGE_V10_MID_FRAMES="${EDGE_V10_MID_FRAMES:-50,100}"
export EDGE_V10_SEARCH_METHOD="${EDGE_V10_SEARCH_METHOD:-greedy}"
export EDGE_V10_TOP_K="${EDGE_V10_TOP_K:-64}"
export EDGE_V10_BEAM_WIDTH="${EDGE_V10_BEAM_WIDTH:-8}"
export EDGE_V10_MID_STRENGTH="${EDGE_V10_MID_STRENGTH:-0.035}"
export EDGE_V10_KEYFRAME_WIDTH="${EDGE_V10_KEYFRAME_WIDTH:-0}"
export EDGE_V10_OUT_PREFIX="output/v10_eval/v10_step1_dual_wu2"
export OUT_PATH="output/v10_eval/v10_step1_dual_wu2.npy"

python generate_v10_choreo.py \
  --checkpoint "$CHECKPOINT" \
  --music "$MUSIC" \
  --feature_type hybrid \
  --audio_dim 803 \
  --start_pose "$START_POSE" \
  --end_pose "$END_POSE" \
  --trajectory "$TRAJ" \
  --auto_mid_pose_space normalized \
  --mid_keyframe_strength "$EDGE_V10_MID_STRENGTH" \
  --infer_keyframe_width "$EDGE_V10_KEYFRAME_WIDTH" \
  --no_tto \
  --out "$OUT_PATH"
