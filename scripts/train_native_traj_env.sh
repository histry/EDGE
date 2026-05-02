#!/usr/bin/env bash
set -euo pipefail

# Environment settings for native trajectory training.
# Use these with your normal train.py / EDGE training command.

export EDGE_TRAJ_LOSS_EARLY_BOOST=3.0
export EDGE_TRAJ_LOSS_POWER=1.0
export EDGE_TRAJ_ENDPOINT_WEIGHT=2.0
export EDGE_TRAJ_ACC_WEIGHT=0.10

# Example only. Replace with your actual train.py arguments.
python train.py \
  --data_path data/dunhuang \
  --feature_type hybrid \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --exp_name native_traj_loss_v1 \
  --trajectory_loss_weight 3.0 \
  --trajectory_velocity_loss_weight 0.8 \
  --epochs 10
