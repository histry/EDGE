# PACE-ChoreoRAG Patch

This patch implements the project-specific solution for root-lower coupling collapse:

1. Beat-aware trajectory pacing.
2. Root-speed scale cap.
3. Elastic sparse trajectory anchors.
4. Retrieved 45-frame unit soft prior with `upper`, `loco_safe`, and `loco_upper` modes.
5. Optional planner-side root-speed / spatial-range hard lock.

All new behavior is disabled unless environment variables are enabled.

## Install

Copy files to the EDGE repo root:

```bash
cp pace_choreorag_trajectory.py /home/disk/lsm/storage/EDGE/
cp choreorag_unit_prior.py /home/disk/lsm/storage/EDGE/
cp install_pace_choreorag_patch.py /home/disk/lsm/storage/EDGE/
cd /home/disk/lsm/storage/EDGE
python install_pace_choreorag_patch.py
python -m py_compile generate_controlled.py choreorag_unit_prior.py pace_choreorag_trajectory.py auto_keyframe_planner.py
```

## Recommended main config

```bash
source scripts/env_reward_collapse.sh
export RAG_EXPR=data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz

export EDGE_UNIT_MIN_EXPRESSIVENESS=0.45
export EDGE_UNIT_EXPRESSIVENESS_BONUS=0.25

export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_FEATURES=loco_upper
export EDGE_UNIT_PRIOR_STRENGTH=0.025
export EDGE_UNIT_PRIOR_MAX_LEN=45
export EDGE_UNIT_PRIOR_DECAY_GAMMA=1.0

export EDGE_TRAJ_BEAT_PACING=1
export EDGE_TRAJ_AUTO_SCALE=1
export EDGE_TRAJ_TARGET_ROOT_SPEED=0.012
export EDGE_TRAJ_MAX_ROOT_SPEED=0.016
export EDGE_TRAJ_MIN_SCALE=0.25
export EDGE_TRAJ_MAX_SCALE=0.45
export EDGE_TRAJ_ELASTIC_ANCHOR=1
export EDGE_TRAJ_ANCHOR_STRIDE=15
export EDGE_TRAJ_ANCHOR_BLEND=1.0
```

## Run

Use the original full S trajectory; PACE will scale and pace it:

```bash
python generate_controlled.py \
  --checkpoint $CKPT_MAIN \
  --music $MUSIC3 \
  --feature_type hybrid \
  --audio_dim 803 \
  --out $OUT_ROOT/pace_choreorag/dunhuangwu3_pace.npy \
  --start_pose $START_POSE \
  --end_pose $END_POSE \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --auto_mid_keyframes \
  --auto_mid_count 1 \
  --rag_db $RAG_EXPR \
  --no_tto \
  --infer_keyframe_width 0 \
  --mid_keyframe_strength 0.10
```

Expected log:

```text
✅ PACE trajectory: scale=..., root_speed=...->..., range=...->..., anchors=...
✅ ChoreoRAG unit soft prior applied: count=1, strength=0.025, features=loco_upper
```

## Optional planner hard lock

The installer tries to add these optional gates if compatible:

```bash
export EDGE_UNIT_MIN_ROOT_SPEED_NORM=0.30
export EDGE_UNIT_MIN_SPATIAL_RANGE_NORM=0.30
```

Use them carefully; only enable when the trajectory span is large enough and after trajectory scaling has been validated.

## Scale sweep diagnostics

```bash
python pace_scale_sweep_report.py \
  --diagnostics_dir output/reward_collapse/diagnostics \
  --glob 'dunhuangwu3_*traj*_diag.json' \
  --out output/reward_collapse/pace_scale_sweep_summary.csv
```
