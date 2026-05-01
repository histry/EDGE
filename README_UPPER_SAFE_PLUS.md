# EDGE upper_safe_plus prior + energy-reranked auto-mid planner

This patch is the safe follow-up after `body_no_root` and `torso_arms` proved too aggressive.

## What changed

1. `retrieved_clip_prior.py`
   - Adds `upper_safe_plus`.
   - Guides only spine, neck/head, shoulders and arms.
   - Does not guide joint 0 pelvis/root orientation.
   - Does not guide hips, knees, ankles or feet.
   - Keeps `upper` as a backward-compatible alias to `upper_safe_plus`.

2. `auto_keyframe_planner.py`
   - Keeps hard source diversity:
     - `source_gap`
     - `disallow_same_source`
   - Adds robust motion-energy normalization.
   - Adds energy reranking inside the top-K compatible candidates:
     - `energy_rerank_top_k`
     - `energy_rerank_weight`
   - This encourages more dynamic retrieved clips without letting energy dominate pose/trajectory compatibility.

3. `generate_controlled.py`
   - Exposes:
     - `--retrieved_prior_body_part upper_safe_plus`
     - `--auto_mid_energy_rerank_top_k`
     - `--auto_mid_energy_rerank_weight`
   - Makes `upper_safe_plus` the default retrieved prior body part.

## Recommended command

```bash
python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --out output/check_mmr_rag/dyl002_mmr_auto3_upper_safe_plus_anchor08.npy \
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
  --auto_mid_energy_rerank_top_k 80 \
  --auto_mid_energy_rerank_weight 0.25 \
  --auto_mid_contact_weight 0.60 \
  --auto_mid_contact_diversity_weight 0.40 \
  --auto_mid_end_weight 0.40 \
  --infer_keyframe_width 1 \
  --retrieved_clip_prior_denoise \
  --retrieved_prior_window 28 \
  --retrieved_prior_strength 0.18 \
  --retrieved_prior_anneal_power 1.5 \
  --retrieved_prior_body_part upper_safe_plus \
  --retrieved_prior_protect_width 2 \
  --retrieved_prior_source_pose_space auto \
  --retrieved_prior_debug_assets \
  --post_anchor_trajectory \
  --trajectory_anchor_strength 0.8 \
  --save_auto_keyframes \
  --save_eval_assets
```

Then run Leg IK:

```bash
python postprocess_leg_ik.py \
  --motion output/check_mmr_rag/dyl002_mmr_auto3_upper_safe_plus_anchor08.npy \
  --out output/check_mmr_rag/dyl002_mmr_auto3_upper_safe_plus_anchor08_legik.npy \
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

## Expected behavior

Compared with `torso_arms` / `body_no_root`, this version should keep foot-contact metrics healthier because it does not touch pelvis/root orientation or the leg chain.
Compared with the old `upper` prior, this patch mainly improves retrieval diversity and motion-energy selection.
