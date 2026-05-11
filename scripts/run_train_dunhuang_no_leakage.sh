#!/usr/bin/env bash
set -euo pipefail

# Strict source-level Train/Val isolation.  Keep these ON for all formal runs.
export EDGE_DUNHUANG_SPLIT_MODE="source_file"
export EDGE_DUNHUANG_STRICT_SPLIT="1"
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT="0"
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH="1"
export EDGE_TRAJECTORY_PLANE="xz"
export EDGE_DUNHUANG_SPLIT_REPORT_DIR="output/split_reports"

# Optional: use a hand-curated split instead of seeded source split.
# JSON example: {"train": ["video_a", "video_b"], "val": ["video_c"]}
# export EDGE_DUNHUANG_SPLIT_MANIFEST="data/dunhuang_bvh/split_manifest.json"

python train.py \
  --data_path data/dunhuang_bvh/processed \
  --feature_type hybrid \
  --audio_dim 803 \
  --audio_pairing_mode none \
  --seq_len 150 \
  --dunhuang_split_ratio 0.9 \
  --dunhuang_split_seed 42 \
  --trajectory_loss_weight 1.0 \
  --trajectory_velocity_loss_weight 0.25 \
  --train_stage full \
  "$@"
