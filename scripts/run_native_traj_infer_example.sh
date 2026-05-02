#!/usr/bin/env bash
set -euo pipefail

# Example: stronger native trajectory following without TTO or hard post anchor.
# Run from the EDGE repository root after copying sitecustomize.py and
# trajectory_native_control.py there.

export EDGE_TRAJ_GUIDANCE_WEIGHT=1.8
export EDGE_TRAJ_GUIDANCE_START_FRAC=0.20
export EDGE_TRAJ_GUIDANCE_END_FRAC=1.00
export EDGE_TRAJ_GUIDANCE_POWER=1.0

python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.9,0.7;-0.8,1.2;0,1.8" \
  --out output/native_traj/dyl002_native_cfg_no_tto.npy \
  --pose_space normalized \
  --num_frames 150 \
  --infer_keyframe_width 1 \
  --no_tto \
  --save_eval_assets
