# EDGE planner-quality + upper prior patch

This patch keeps the currently most stable denoising prior route:

- `retrieved_prior_body_part=upper`
- `retrieved_prior_strength=0.16`
- `retrieved_prior_window=24`

and moves the next improvement to auto-mid retrieval quality:

1. hard source/source-frame diversity constraints;
2. moderate motion-energy target instead of high-energy chasing;
3. stronger contact stability and contact-pattern diversity;
4. metadata/debug fields for score parts and rerank decisions.

## Files

- `auto_keyframe_planner.py`
- `retrieved_clip_prior.py`
- `generate_controlled.py`

## Main new arguments

- `--auto_mid_energy_target 0.55`
- `--auto_mid_energy_band 0.25`
- `--auto_mid_energy_rerank_weight 0.08`
- `--auto_mid_contact_weight 0.85`
- `--auto_mid_contact_diversity_weight 0.60`
- `--retrieved_prior_body_part upper`

## Recommended command

```bash
python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --out output/check_mmr_rag/dyl002_mmr_auto3_planner_quality_upper_anchor08.npy \
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
  --auto_mid_energy_weight 0.45 \
  --auto_mid_energy_target 0.55 \
  --auto_mid_energy_band 0.25 \
  --auto_mid_energy_rerank_top_k 80 \
  --auto_mid_energy_rerank_weight 0.08 \
  --auto_mid_contact_weight 0.85 \
  --auto_mid_contact_diversity_weight 0.60 \
  --auto_mid_end_weight 0.40 \
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


## v2 fix

This package fixes a previous mismatch where generate_controlled.py passed
`energy_target` and `energy_band` but auto_keyframe_planner.py did not accept them.

The planner now uses medium-energy target scoring:
`motion_energy_cost = abs(motion_energy_norm - energy_target) / energy_band`,
and the top-K energy rerank also prefers clips close to `energy_target` instead of
blindly selecting the highest-energy clip.
