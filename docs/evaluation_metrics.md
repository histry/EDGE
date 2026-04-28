# Quantitative Evaluation Metrics

This project now provides `eval_quantitative.py` for generated 151-D `.npy` motions.

## Metrics

### Keyframe Error

Inputs: generated motion and target keyframe `.npy` files.

The script loads normalized keyframes through the checkpoint normalizer by default, then compares the generated frame with the target frame.

Reported values:

- `keyframe_mpjpe_m_mean`: FK joint MPJPE in meters, averaged across keyframes.
- `keyframe_mpjpe_m_max`: worst keyframe MPJPE.
- `keyframe_rot_err_deg_mean`: mean SO(3) rotation angle error in degrees.
- `keyframe_feature_rmse_mean`: RMSE over root height and local joint rotations.

By default, root X/Z is ignored for keyframe pose error because root X/Z is owned by trajectory control.

### Trajectory Error

Inputs: generated root X/Z and target trajectory.

Reported values:

- `trajectory_ade_m`: mean per-frame root X/Z distance.
- `trajectory_rmse_m`: root X/Z RMSE.
- `trajectory_max_error_m`: worst-frame root X/Z error.
- `trajectory_final_error_m`: final-frame root X/Z error.
- `trajectory_dtw_m_per_frame`: DTW distance normalized by sequence length.

The target trajectory can be provided either as control points through `--trajectory` or as an explicit `.npy` through `--target_traj`.

### Foot Sliding Rate

Inputs: generated motion contacts and FK foot joints.

The script computes foot horizontal velocity during contact frames.

Reported values:

- `foot_slide_rate`: ratio of contact foot-frame pairs with horizontal speed above `--slide_speed_threshold`.
- `foot_contact_speed_mean_mps`: mean horizontal foot speed during contact.
- `foot_contact_speed_p95_mps`: 95th percentile horizontal foot speed during contact.

When contact channels are saturated or unusable, `--contact_source auto` falls back to height-based foot contact.

### BeatAlign

Inputs: generated motion and audio.

Motion beats are local minima of smoothed global joint velocity. Audio beats are extracted with `librosa.beat.beat_track`.

Reported values:

- `beatalign_motion_to_audio`: each motion beat matched to nearest audio beat.
- `beatalign_audio_to_motion`: each audio beat matched to nearest motion beat.
- `beatalign_symmetric`: mean of the two directions.

The Gaussian tolerance is controlled by `--beatalign_sigma_frames`, default 3 frames.

## Example

```bash
/home/disk/lsm/conda_envs/edge/bin/python eval_quantitative.py \
  --motion output/stage2B_train5_4key_midmove_filter_20s_resmoothed_v6.npy \
  --audio test_music_bank/dunhuangwu2_20s.wav \
  --checkpoint runs/train/exp_dunhuang_keyframe_stage2B_stability_from102/weights/train-5.pt \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --mid_poses test_keyframes/dyl002_600_1800_mid1.npy,test_keyframes/dyl002_600_1800_mid2.npy \
  --mid_pose_frames 180,360 \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --out_json output/stage2B_train5_4key_midmove_filter_20s_resmoothed_v6_metrics.json \
  --out_csv output/stage2B_train5_4key_midmove_filter_20s_resmoothed_v6_metrics.csv
```

Add `--trajectory "0,0;1,2;-1,4;0,5"` or `--target_traj path/to/trajectory.npy` when trajectory error is needed.
