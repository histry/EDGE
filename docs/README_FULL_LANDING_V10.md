# EDGE V10 Full-Landing Patch

This replacement package completes the practical landing of:

1. ZeroInitTrajectoryAdapter
2. Energy-Conditioned CFG
3. Context Token RAG v1.5

It is designed to remain checkpoint-compatible with existing V9/V10 weights.

## Files

```text
edge_full_landing_patch.py
sitecustomize.py
scripts/run_v10_full_landing_step4.sh
scripts/run_v10_energy_cfg_ablation.sh
scripts/run_v10_trajectory_rep_ablation.sh
scripts/run_train_adapter_zero_traj_energy.sh
README_FULL_LANDING_V10.md
```

## What is changed

### 1. ZeroInitTrajectoryAdapter optimization

The model already has ZeroInitTrajectoryAdapter.  This patch adds runtime trajectory representation control:

```bash
export EDGE_TRAJECTORY_REP=absolute
export EDGE_TRAJECTORY_REP=relative_abs_vel
export EDGE_TRAJECTORY_REP=centered_abs_vel
```

Recommended default:

```bash
export EDGE_TRAJECTORY_REP=relative_abs_vel
```

This removes global X/Z translation offset while preserving the path shape and frame-to-frame velocity.

### 2. Energy-Conditioned CFG

The model already supports `cond["energy"]` and `EDGE_ENERGY_CFG_SCALE`.  This patch fills missing inference energy automatically:

```bash
export EDGE_DEFAULT_ENERGY=0.65
# or
export EDGE_AUDIO_ENERGY_AS_COND=1
export EDGE_MUSIC_TENSION_AS_ENERGY=1
export EDGE_ENERGY_CFG_SCALE=0.65
```

### 3. Context Token RAG v1.5

The full PoseEncoder + Cross-Attention Context RAG is still a future architecture change.  
This patch improves the existing V9 7D RAG Summary Token safely:

```bash
export EDGE_CONTEXT_RAG_ENHANCE=1
export EDGE_CONTEXT_RAG_SMOOTH=5
export EDGE_CONTEXT_RAG_SCALE=0.50
```

No checkpoint shape changes.

## Quick test

```bash
python -m py_compile edge_full_landing_patch.py sitecustomize.py
bash scripts/run_v10_full_landing_step4.sh
```

Check logs:

```bash
grep -E "full-landing patch|checkpoint_has_rag|V9 RAG summary attached|Traceback|ERROR" -n logs/v10_full_landing_step4.log
```

Expected:

```text
Installed EDGE full-landing patch
checkpoint_has_rag=True
V9 RAG summary attached
no Traceback
```

## Recommended experiment order

1. Standard full-landing Step4:

```bash
bash scripts/run_v10_full_landing_step4.sh
```

2. Energy CFG ablation:

```bash
bash scripts/run_v10_energy_cfg_ablation.sh
```

3. Trajectory representation ablation:

```bash
bash scripts/run_v10_trajectory_rep_ablation.sh
```

4. Adapter-stage training:

```bash
bash scripts/run_train_adapter_zero_traj_energy.sh
```

## Important wording for paper/report

Accurate wording:

```text
We have implemented a checkpoint-compatible Context Token RAG v1.5 based on
V9 RAG Summary Tokens.  The full PoseEncoder + Cross-Attention Context RAG is
reserved for the next architecture stage.
```

Do not claim the full Cross-Attention Context RAG is already trained unless that architecture is added and evaluated later.
