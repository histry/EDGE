# EDGE Advanced Trajectory / Footstep-aware Decoupled ChoreoRAG Patch

This patch implements a staged trajectory-control roadmap for EDGE without forcing all experiments on at once.
All new features are environment-gated.

## Files

Replacement files:

- `train.py` — installs all runtime patches before constructing EDGE.
- `build_choreo_unit_rag_db.py` — builds Footstep-aware dual-score RAG DB.
- `sitecustomize.py` — installs inference/training runtime patches for scripts such as `generate_controlled.py`.

New files:

- `footstep_phase_utils.py`
- `trajectory_representation_utils.py`
- `trajectory_enhancement_patch.py`
- `gait_phase_dataset_patch.py`
- `gait_phase_adapter_patch.py`
- `footstep_aware_unit_selector.py`
- `segment_lower_body_compositor.py`
- `decoupled_upper_lower_merge.py`
- scripts under `scripts/`

## Stage 1: no-retrain validation

Goal: test whether the current failure is caused by RAG selecting/executing the wrong lower-body units.

```bash
export V12_CKPT=runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt
export RAG_DB=data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz

python build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out "$RAG_DB" \
  --checkpoint "$V12_CKPT" \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_model BAAI/bge-small-zh-v1.5 \
  --text_device cuda
```

Run compositor:

```bash
export RAG_DB=data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz
export BASE_MOTION=output/v12_dhw4_base/dhw4_v12.npy
export TARGET_TRAJ=output/v12_dhw4_base/dhw4_v12_target_traj.npy
export OUT_PREFIX=output/v12_footstep_stage1/dhw4_v12_main
export MID_FRAMES=25,50,75,100,125
export EDGE_COMPOSITOR_LOWER_STRENGTH=0.85
export EDGE_COMPOSITOR_TORSO_STRENGTH=0.25
export EDGE_COMPOSITOR_UPPER_STRENGTH=0.00
export EDGE_COMPOSITOR_WINDOW=45
bash scripts/run_stage1_footstep_rag_compositor.sh
```

## Stage 2: lightweight adapter retraining

This adds:

- gait/contact phase prior
- Fourier trajectory features
- speed / heading / curvature features
- sparse waypoint mask
- dynamic trajectory CFG during inference

```bash
export CHECKPOINT=runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt
export PROJECT=runs/train_advanced_traj_phase
export EXP_NAME=v12_gait_fourier_sparse_adapter_v1
export BATCH_SIZE=4
export EPOCHS=20
export LR=1e-4
bash scripts/run_stage2_advanced_traj_adapter_train.sh
```

Important flags:

```bash
export EDGE_GAIT_PHASE_COND=1
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1
export EDGE_TRAJ_WAYPOINT_FRAMES=0,50,100,149
export EDGE_DYNAMIC_TRAJ_CFG=1
```

## Stage 3: decoupled upper/lower validation

Deterministic merge first:

```bash
export LOWER_MOTION=output/v12_footstep_stage1/dhw4_v12_main_composited.npy
export UPPER_MOTION=output/v12_dhw4_base/dhw4_v12.npy
export OUT=output/decoupled/dhw4_v12_decoupled.npy
bash scripts/run_stage3_decoupled_inpainting_plan.sh
```

Full diffusion inpainting is a later architecture pass; this deterministic merge verifies whether the target decomposition works visually.

## Stage 4: BEV / stage-map experimental branch

After Stage 2 is validated:

```bash
export CHECKPOINT=runs/train_advanced_traj_phase/v12_gait_fourier_sparse_adapter_v1/weights/train-20.pt
export EDGE_TRAJ_BEV_COND=1
bash scripts/run_stage4_bev_experimental_train.sh
```

This adds a BEV heatmap residual to trajectory tokens. It is optional and should not replace gait phase / sparse waypoint training as the first fix.

## Core environment switches

```bash
# Footstep phase
export EDGE_GAIT_PHASE_COND=1
export EDGE_GAIT_CONTACT_LOSS=1
export EDGE_GAIT_CONTACT_LOSS_WEIGHT=0.60

# Advanced trajectory features
export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=1
export EDGE_TRAJ_FOURIER_BANDS=6
export EDGE_TRAJ_SPARSE_WAYPOINT=1

# Dynamic inference CFG
export EDGE_DYNAMIC_TRAJ_CFG=1
export EDGE_TRAJ_CFG_BASE=2.0
export EDGE_TRAJ_CFG_SPEED_W=2.0
export EDGE_TRAJ_CFG_CURVATURE_W=1.0

# Optional BEV
export EDGE_TRAJ_BEV_COND=1
export EDGE_TRAJ_BEV_SIZE=32
```

## Why this patch is checkpoint-friendly

The original `cond['trajectory']` stays as normalized X/Z. The original model still internally creates X/Z + ΔX/ΔZ. Advanced features are injected as residual trajectory tokens through runtime wrappers, so old checkpoints can still load.

## Expected experimental interpretation

- If Stage 1 compositor improves leg motion, the main bottleneck is soft RAG not executing lower-body units.
- If Stage 1 does not improve, the RAG DB lacks useful locomotion/footstep units or the selected units are unsuitable.
- If Stage 2 improves generation without compositor, the model has learned trajectory speed → gait phase → lower-body/contact mapping.
- Stage 3/4 are for the final paper architecture.
