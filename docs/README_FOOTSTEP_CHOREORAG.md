# Footstep-Aware Decoupled ChoreoRAG Patch

This package implements the staged plan:

1. **Footstep-aware dual-score/dynamic RAG + Segment-level Lower-body Compositor**  
   No retraining. Rebuild RAG DB, select mobile units by target trajectory speed, blend lower-body units into the current generated motion.

2. **Gait / Footstep Phase Adapter Training**  
   Lightweight adapter training. Adds `cond["gait_phase"]` and a trainable gait branch inside the trajectory projection. Existing checkpoints remain usable.

3. **Decoupled Upper/Lower Inpainting Finalization**  
   First practical implementation is a deterministic merge utility: lower/root/contact from the locomotion pass, upper/torso style from Text/Pose Context RAG. Full diffusion inpainting can be built on top of the same masks/conditioning.

## Files

Replace/add these files at the EDGE repository root:

```text
train.py
build_choreo_unit_rag_db.py
footstep_phase_utils.py
gait_phase_dataset_patch.py
gait_phase_adapter_patch.py
footstep_aware_unit_selector.py
segment_lower_body_compositor.py
decoupled_upper_lower_merge.py
scripts/run_stage1_footstep_rag_compositor.sh
scripts/run_stage2_gait_phase_adapter_train.sh
scripts/run_stage3_decoupled_merge.sh
```

## Stage 1: no retraining validation

```bash
export RAG_DB=data/dunhuang_choreo_unit_rag/index_footstep_u45_s15.npz
export CKPT=runs/train_stage45/stage45_riskfix_v1_e504/weights/train-10.pt
export INPUT_DIR=data/dunhuang_bvh/processed
export BASE_MOTION=output/text_context_rag_ckpt_eval/dhw4_textctx_e4_hd6000.npy
export TARGET_TRAJ=output/text_context_rag_ckpt_eval/dhw4_textctx_e4_hd6000_target_traj.npy
export OUT_PREFIX=output/footstep_stage1/dhw4_footstep
export MID_FRAMES=25,50,75,100,125

bash scripts/run_stage1_footstep_rag_compositor.sh
```

Output:

```text
output/footstep_stage1/dhw4_footstep_composited.npy
output/footstep_stage1/dhw4_footstep_footstep_plan.json
```

Recommended ablation strengths:

```bash
# safe
export EDGE_COMPOSITOR_LOWER_STRENGTH=0.60
export EDGE_COMPOSITOR_TORSO_STRENGTH=0.15
export EDGE_COMPOSITOR_UPPER_STRENGTH=0.00
export EDGE_COMPOSITOR_WINDOW=35

# main
export EDGE_COMPOSITOR_LOWER_STRENGTH=0.85
export EDGE_COMPOSITOR_TORSO_STRENGTH=0.25
export EDGE_COMPOSITOR_UPPER_STRENGTH=0.00
export EDGE_COMPOSITOR_WINDOW=45

# strong
export EDGE_COMPOSITOR_LOWER_STRENGTH=1.00
export EDGE_COMPOSITOR_TORSO_STRENGTH=0.35
export EDGE_COMPOSITOR_UPPER_STRENGTH=0.10
export EDGE_COMPOSITOR_WINDOW=55
```

## Stage 2: gait phase adapter training

```bash
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_GAIT_CONTACT_LOSS=1
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=0.60
export EDGE_DIFF_CONTACT_LOSS=1
export EDGE_DCL_CONTACT_SOURCE=auto
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height

export CHECKPOINT=runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt
export PROJECT=runs/train_footstep_phase
export EXP_NAME=gait_phase_adapter_v1
export BATCH_SIZE=4
export EPOCHS=20
export LR=1e-4

bash scripts/run_stage2_gait_phase_adapter_train.sh
```

First run should stay adapter-only. If the video still ignores footstep phase, run a second short decoder-unfrozen pass:

```bash
export EXP_NAME=gait_phase_adapter_v1_decoder_lite
export LR=3e-5
export EPOCHS=10
bash scripts/run_stage2_gait_phase_adapter_train.sh --adapter_train_decoder
```

## Stage 3: decoupled upper/lower merge validation

```bash
export LOWER_MOTION=output/footstep_stage1/dhw4_footstep_composited.npy
export UPPER_MOTION=output/text_context_rag_ckpt_eval/dhw4_textctx_e4_hd6000.npy
export OUT=output/decoupled/dhw4_decoupled_merge.npy
bash scripts/run_stage3_decoupled_merge.sh
```

This deterministic merge is not the final diffusion inpainting pass, but it verifies the final target decomposition:

```text
root/lower/contact = locomotion source
upper/arms/style   = Text/Pose Context RAG source
```

## Environment switches

```bash
# Stage 1 selector/compositor
export EDGE_RAG_MOBILE_SPEED_THRESHOLD=0.010
export EDGE_COMPOSITOR_WINDOW=45
export EDGE_COMPOSITOR_LOWER_STRENGTH=0.85
export EDGE_COMPOSITOR_TORSO_STRENGTH=0.25
export EDGE_COMPOSITOR_UPPER_STRENGTH=0.00
export EDGE_COMPOSITOR_CONTACT_STRENGTH=0.75

# Stage 2 model/dataset patch
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_PHASE_DIM=6
export EDGE_GAIT_PHASE_DROP_PROB=0.10
export EDGE_GAIT_PHASE_SPEED_THRESHOLD=0.01
export EDGE_GAIT_PHASE_STRIDE_LENGTH=0.35
export EDGE_GAIT_CONTACT_LOSS=1
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=0.60

# Contact-loss safety
export EDGE_DIFF_CONTACT_LOSS=1
export EDGE_DCL_CONTACT_SOURCE=auto
export EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85
export EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height
```

## Notes

- All trajectory/root calculations use X/Z ground plane: feature dims `[4,6]`.
- `build_choreo_unit_rag_db.py` now saves `expressiveness_score`, `locomotion_score`, `footstep_score`, and `mobile_score`.
- `gait_phase_adapter_patch.py` wraps `trajectory_projection`; it does not change the base `DanceDecoder` constructor signature.
- Existing checkpoints remain loadable; new gait parameters are initialized by the current model and trained in adapter stage.
