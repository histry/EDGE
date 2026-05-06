#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

export V8B_E4=/home/disk/lsm/storage/EDGE/runs/train_stage45/v8b_content_stage2_smooth/weights/train-4.pt

python train.py \
  --checkpoint "$V8B_E4" \
  --project runs/train_stage45 \
  --exp_name v9_rag_summary_token_e4 \
  --data_path data/dunhuang_bvh/processed \
  --feature_type hybrid \
  --audio_dim 803 \
  --train_stage stage2 \
  --epochs 4 \
  --learning_rate 2e-6 \
  --batch_size 16 \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --keyframe_condition_prob 0.60 \
  --keyframe_condition_width 2 \
  --keyframe_loss_weight 0.8 \
  --mid_keyframe_condition_prob 0.45 \
  --mid_keyframe_count 2 \
  --mid_keyframe_condition_width 1 \
  --mid_keyframe_selection mixed \
  --trajectory_loss_weight 0.5 \
  --trajectory_velocity_loss_weight 0.2 \
  --contact_loss_weight 0.8 \
  --foot_loss_weight 3.5 \
  --sync_loss_weight 0.8 \
  --root_lower_coupling_loss_weight 0.25 \
  --root_lower_speed_threshold 0.012 \
  --root_lower_min_motion 0.006 \
  --energy_condition_prob 0.7 \
  --energy_condition_drop_prob 0.15 \
  --energy_loss_weight 0.20 \
  --save_interval 2 \
  --val_batches 10 \
  --enable_rag_summary_token \
  --rag_summary_dim 7 \
  --rag_summary_drop_prob 0.15
