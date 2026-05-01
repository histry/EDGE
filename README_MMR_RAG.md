# EDGE MMR-RAG Replacement Pack

This pack implements the route:

AIST++ paired music-motion data -> true MMR dual-tower training
Dunhuang BVH -> motion-domain adaptation + MMR-RAG clip database
Chinese folk/Dunhuang music -> inference-time retrieval and optional weak pairing

## Files

- `model/mmr_encoder.py`: AudioEncoder + MotionEncoder dual tower.
- `losses/mmr_loss.py`: symmetric InfoNCE loss and retrieval metrics.
- `prepare_mmr_pairs.py`: create AIST++ paired index from processed .npy folders.
- `train_mmr.py`: train true MMR on paired AIST++ clips.
- `adapt_mmr_motion_dunhuang.py`: adapt MotionEncoder on unpaired Dunhuang motion clips.
- `build_mmr_rag_db.py`: build clip-level Dunhuang MMR-RAG index with motion embeddings.
- `build_weak_pairs.py`: create low-confidence weak Chinese music-Dunhuang motion pairs by rhythm similarity.
- `build_dunhuang_rag_db.py`: legacy pose-level RAG DB builder retained for fallback.
- `auto_keyframe_planner.py`: auto-mid planner using proxy score or true MMR-RAG when checkpoint/index are provided.
- `generate_controlled.py`: full replacement with MMR-RAG auto mid-keyframe options.

## Recommended order

1. Prepare AIST++ paired clips:

```bash
python prepare_mmr_pairs.py \
  --audio_feature_dir data/aist_mmr/audio \
  --motion_dir data/aist_mmr/motion \
  --out_dir data/mmr_aist
```

2. Train true MMR:

```bash
python train_mmr.py \
  --index_json data/mmr_aist/index_train.json \
  --val_index_json data/mmr_aist/index_val.json \
  --out runs/mmr/mmr_aist_pretrain.pt \
  --seq_len 150 \
  --audio_dim 803 \
  --motion_dim 151 \
  --embed_dim 256 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4
```

3. Adapt MotionEncoder to Dunhuang BVH:

```bash
python adapt_mmr_motion_dunhuang.py \
  --mmr_checkpoint runs/mmr/mmr_aist_pretrain_best.pt \
  --dunhuang_motion_dir data/dunhuang_bvh/processed \
  --edge_checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --pose_space normalized \
  --out runs/mmr/mmr_aist_dunhuang_motion_adapt.pt \
  --seq_len 150 \
  --stride 30 \
  --lr 1e-5 \
  --epochs 20
```

4. Build clip-level MMR-RAG DB:

```bash
python build_mmr_rag_db.py \
  --mmr_checkpoint runs/mmr/mmr_aist_dunhuang_motion_adapt.pt \
  --motion_dir data/dunhuang_bvh/processed \
  --edge_checkpoint runs/best/dunhuang_stage1_best_exp17_train30.pt \
  --pose_space normalized \
  --out data/dunhuang_rag_db/mmr_rag_index.npz \
  --seq_len 150 \
  --window 150 \
  --stride 30
```

5. Generate with true MMR-RAG auto mid-keyframes:

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

Then run `postprocess_leg_ik.py` as before.
