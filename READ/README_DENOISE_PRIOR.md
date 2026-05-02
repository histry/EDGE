# EDGE Retrieved Clip Prior in DDPM Denoising Loop

This patch upgrades RAG-Diffusion from postprocess clip blending to denoising-loop soft prior.

## Files

- `generate_controlled.py`: full replacement. Adds `--retrieved_clip_prior_denoise` and builds a soft prior from `auto_mid_plan.json` before DDPM sampling.
- `retrieved_clip_prior.py`: new file. Loads retrieved source clips, builds normalized prior tensors, and monkey-patches `GaussianDiffusion.p_mean_variance` so prior is applied to predicted `x_start` at every denoising step.

## Replace

```bash
cd /home/disk/lsm/storage/EDGE
cp generate_controlled.py generate_controlled.py.bak_before_denoise_prior
cp /path/to/generate_controlled.py .
cp /path/to/retrieved_clip_prior.py .
```

## Run

```bash
python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --out output/check_mmr_rag/dyl002_mmr_auto3_denoiseprior_anchor08.npy \
  --pose_space normalized \
  --num_frames 150 \
  --auto_mid_keyframes \
  --rag_db data/dunhuang_rag_db/mmr_rag_index.npz \
  --mmr_checkpoint runs/mmr/mmr_aist_dunhuang_motion_adapt.pt \
  --auto_mid_count 3 \
  --auto_mid_min_gap 18 \
  --auto_mid_mmr_weight 0.5 \
  --auto_mid_pose_weight 1.0 \
  --auto_mid_energy_weight 0.4 \
  --auto_mid_contact_weight 0.3 \
  --auto_mid_end_weight 0.3 \
  --infer_keyframe_width 1 \
  --retrieved_clip_prior_denoise \
  --retrieved_prior_window 24 \
  --retrieved_prior_strength 0.16 \
  --retrieved_prior_anneal_power 1.5 \
  --retrieved_prior_body_part upper \
  --retrieved_prior_protect_width 2 \
  --retrieved_prior_source_pose_space auto \
  --retrieved_prior_debug_assets \
  --post_anchor_trajectory \
  --trajectory_anchor_strength 0.8 \
  --save_auto_keyframes \
  --save_eval_assets
```

Then run Leg IK as before:

```bash
python postprocess_leg_ik.py \
  --motion output/check_mmr_rag/dyl002_mmr_auto3_denoiseprior_anchor08.npy \
  --out output/check_mmr_rag/dyl002_mmr_auto3_denoiseprior_anchor08_legik.npy \
  --device cpu \
  --height_threshold 0.035 \
  --min_contact_len 3 \
  --steps 220 \
  --lr 0.01 \
  --foot_weight 0.8 \
  --reg_weight 0.08 \
  --smooth_weight 0.12 \
  --y_weight 0.10
```
