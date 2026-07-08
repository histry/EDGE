# V46.41 Stage-Anchored Guided Temporal Generative Transactions

This package is designed for the current EDGE/V46 whole-song Dunhuang dance project.
It combines:

1. **Chang-E BVH canonicalization**: converts 19-node 114-channel 6DoF BVH to rot-only meter-scale BVH.
2. **V46.38 MSSD/AESD routing compatibility**: run `apply_v46_38_complete_routing_patch.py` first.
3. **Macroscopic Stage Anchoring (MSA)**: low-frequency global root-XZ prior to control long-horizon drift.
4. **TGT + KBO**: local transaction windows, atomic commit/rollback, kinematic barrier oracle.
5. **Diffusion early-abort**: intermediate KBO probe during denoising.
6. **HN-DPO-style hard-negative mining**: saves rejected candidates and KBO-safe preferences.

## Install

```bash
cd /home/disk/lsm/storage/EDGE
unzip -o /mnt/data/EDGE_V46_41_STAGE_ANCHOR_GUIDED_TGT_solution.zip -d /mnt/data/
cp /mnt/data/v46_41_solution/tools/*.py tools/
cp /mnt/data/v46_41_solution/scripts/*.sh scripts/
chmod +x tools/v46_41_*.py tools/canonicalize_chang_e_bvh_rot_only_meter.py tools/apply_v46_41_stage_anchor_guided_tgt_patch.py scripts/run_v46_41_stage_anchor_guided_full_retrain.sh
```

## Run

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V26_V23_CKPT="./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt"
nohup bash scripts/run_v46_41_stage_anchor_guided_full_retrain.sh > output/v46_41_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Main switches

```bash
export V46_41_MSA_ENABLE=1
export V46_41_STAGE_RADIUS_M=1.80
export V46_41_MSA_REFERENCE_STRENGTH=0.10
export V46_41_MSA_TRANSACTION_STRENGTH=0.08
export V46_41_TGT_ENABLE=1
export V46_41_IK_TGT_ENABLE=1
export V46_41_DIFFUSION_EARLY_ABORT_ENABLE=1
export V46_41_HN_DPO_SAVE_PAIRS=1
export V46_41_HN_DPO_DIR="$RUN_ROOT/v46_41_hn_pairs"
```
