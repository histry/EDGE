# EDGE MMR-RAG Diversity + body_no_root Denoising Prior Patch

This patch addresses two issues observed in the MMR-RAG auto-mid pipeline:

1. Auto mid-keyframes may repeat the same retrieved source/source_frame, causing repeated one-leg hopping.
2. Retrieved clip prior only guided upper body rotations, so it had limited visible impact on repeated lower-body/root-orientation patterns.

Files:

- `auto_keyframe_planner.py`
  - Adds source-region hard diversity.
  - Adds contact-pattern diversity scoring.
  - Raises recommended default motion-energy/contact weights.

- `retrieved_clip_prior.py`
  - Adds `body_no_root` mode: all 24 joint rotations are softly guided, while root translation/contact channels stay untouched.

- `generate_controlled.py`
  - Adds CLI params:
    - `--auto_mid_source_gap`
    - `--auto_mid_disallow_same_source`
    - `--auto_mid_contact_diversity_weight`
    - `--retrieved_prior_body_part body_no_root`

Recommended command:

```bash
python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --out output/check_mmr_rag/dyl002_mmr_auto3_diverse_bodyprior_anchor08.npy \
  --pose_space normalized \
  --num_frames 150 \
  --auto_mid_keyframes \
  --rag_db data/dunhuang_rag_db/mmr_rag_index.npz \
  --mmr_checkpoint runs/mmr/mmr_aist_dunhuang_motion_adapt.pt \
  --auto_mid_count 3 \
  --auto_mid_min_gap 18 \
  --auto_mid_source_gap 150 \
  --auto_mid_disallow_same_source \
  --auto_mid_mmr_weight 0.45 \
  --auto_mid_pose_weight 1.0 \
  --auto_mid_diversity_weight 0.30 \
  --auto_mid_energy_weight 0.70 \
  --auto_mid_contact_weight 0.60 \
  --auto_mid_contact_diversity_weight 0.40 \
  --auto_mid_end_weight 0.40 \
  --infer_keyframe_width 1 \
  --retrieved_clip_prior_denoise \
  --retrieved_prior_window 28 \
  --retrieved_prior_strength 0.18 \
  --retrieved_prior_anneal_power 1.5 \
  --retrieved_prior_body_part body_no_root \
  --retrieved_prior_protect_width 2 \
  --retrieved_prior_source_pose_space auto \
  --retrieved_prior_debug_assets \
  --post_anchor_trajectory \
  --trajectory_anchor_strength 0.8 \
  --save_auto_keyframes \
  --save_eval_assets
```
