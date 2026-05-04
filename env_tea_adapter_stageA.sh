#!/usr/bin/env bash
# TEA-MotionAdapter Stage A: native trajectory adapter warmup.
# Run from EDGE root after applying install_tea_motion_adapter_patch.py.

set -euo pipefail

export EDGE_ENERGY_DROP_PROB=0.15

python train.py \
  --project runs/train \
  --exp_name tea_adapter_stageA \
  --data_path data/dunhuang_bvh/processed \
  --checkpoint weights/your_pace_or_stage45_checkpoint.pt \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --batch_size 16 \
  --epochs 30 \
  --learning_rate 1e-4 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --audio_pairing_mode none \
  --train_stage adapter \
  --cond_drop_prob 0.25 \
  --energy_condition_prob 0.7 \
  --energy_condition_drop_prob 0.15 \
  --energy_loss_weight 0.25 \
  --trajectory_loss_weight 1.0 \
  --trajectory_velocity_loss_weight 0.5 \
  --root_lower_coupling_loss_weight 0.5 \
  --root_lower_speed_threshold 0.012 \
  --root_lower_min_motion 0.010 \
  --contact_loss_weight 0.8 \
  --foot_loss_weight 2.5 \
  --sync_loss_weight 1.2 \
  --traj_aug_prob 0.8 \
  --traj_aug_scale_min 0.25 \
  --traj_aug_scale_max 0.75 \
  --traj_aug_rot_deg 30 \
  --save_interval 5 \
  --val_batches 10
