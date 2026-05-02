# Retrieved Clip Soft Prior v1 for EDGE

## Why
The old MMR-RAG pipeline retrieves a motion clip but only uses the center pose as an auto mid-keyframe. This loses temporal dynamics and can produce repeated pose pulses / single-leg jumps. This patch adds a soft temporal prior by blending retrieved clip upper-body rotations around each auto-mid frame.

## Files
- `postprocess_retrieved_clip_prior.py`: new postprocess script.

## Recommended pipeline

### 1. Generate MMR-RAG auto-mid as before
```bash
python generate_controlled.py \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --music /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --out output/check_mmr_rag/dyl002_mmr_auto3_anchor08.npy \
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
  --post_anchor_trajectory \
  --trajectory_anchor_strength 0.8 \
  --save_auto_keyframes \
  --save_eval_assets
```

### 2. Apply retrieved clip soft prior
```bash
python postprocess_retrieved_clip_prior.py \
  --motion output/check_mmr_rag/dyl002_mmr_auto3_anchor08.npy \
  --auto_plan output/check_mmr_rag/dyl002_mmr_auto3_anchor08_auto_mid_plan.json \
  --out output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior.npy \
  --edge_checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --source_pose_space normalized \
  --window 24 \
  --blend_strength 0.35 \
  --body_part upper \
  --protect_frames 0,149 \
  --protect_width 3 \
  --temporal_smooth 5 \
  --save_prior_assets
```

If upper body moves too much, use:
```bash
--body_part arms --blend_strength 0.25
```

If it still looks like isolated pulses, use:
```bash
--window 32 --blend_strength 0.45
```

### 3. Run Leg IK after clip prior
```bash
python postprocess_leg_ik.py \
  --motion output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior.npy \
  --out output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior_legik.npy \
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

### 4. Evaluate
Use the auto mid paths printed by `generate_controlled.py`:
```bash
python eval_quantitative.py \
  --motion output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior_legik.npy \
  --raw_motion output/check_mmr_rag/dyl002_mmr_auto3_anchor08_raw.npy \
  --post_motion output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior_legik.npy \
  --checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --audio /home/lsm/disk_space/storage/EDGE/test_music_bank/dunhuangwu2.wav \
  --target_traj output/check_mmr_rag/dyl002_mmr_auto3_anchor08_target_traj.npy \
  --start_pose test_keyframes/dyl002_600_1800_start.npy \
  --mid_poses output/check_mmr_rag/dyl002_mmr_auto3_anchor08_auto_mid1_f045.npy,output/check_mmr_rag/dyl002_mmr_auto3_anchor08_auto_mid2_f091.npy,output/check_mmr_rag/dyl002_mmr_auto3_anchor08_auto_mid3_f121.npy \
  --mid_pose_frames 45,91,121 \
  --end_pose test_keyframes/dyl002_600_1800_end.npy \
  --out_json output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior_legik_metrics.json \
  --out_csv output/check_mmr_rag/dyl002_mmr_auto3_anchor08_clipprior_legik_metrics.csv \
  --keyframe_space normalized
```

## What to compare
Compare:
- `mmr_auto3_legik`
- `mmr_auto3_clipprior_legik`
- `kw3_legik_smooth`

Metrics:
- `beatalign_symmetric`: should stay high or improve.
- `keyframe_mpjpe_m_mean`: should stay < 0.02m.
- `foot_contact_speed_p95_mps`: may need IK and should not explode.
- Visual: fewer repeated arm pulses / fewer single-leg jumps.

## Important limitation
This v1 is a soft prior postprocess, not a full denoising-loop conditioning. It is designed to validate whether retrieved temporal clips improve motion continuity before modifying `model/diffusion.py` deeply.
