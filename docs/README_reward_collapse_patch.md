# EDGE / ChoreoRAG Reward-Collapse Patch

This bundle implements the four-stage plan with environment-variable isolation.

## Files

- `build_choreo_unit_rag_db.py`: direct replacement. Adds `expressiveness_score` and normalized stats.
- `music_choreo_planner.py`: direct replacement. Adds `tension_target` and `phase` fields.
- `auto_keyframe_planner.py`: direct replacement. Adds expressiveness filtering/bonus, tension-aware dynamic weights and homogeneity penalty.
- `planner_edit_loop.py`: optional direct replacement. Adds retrieved/generated expressiveness diagnostics.
- `choreorag_unit_prior.py`: new helper file for Phase 4.
- `generate_controlled_phase4.diff`: small patch for `generate_controlled.py` to activate 45-frame soft unit priors.

## Phase 1: Expressiveness replacement for raw energy

Rebuild DB:

```bash
python build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out data/dunhuang_choreo_unit_rag/index_u45_s15_expr.npz \
  --checkpoint runs/train_stage45/stage45_riskfix_v1_e504/weights/train-10.pt \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_model BAAI/bge-small-zh-v1.5 \
  --text_device cuda \
  --expr_w_energy 0.30 \
  --expr_w_upper 0.30 \
  --expr_w_spatial 0.20 \
  --expr_w_turning 0.15 \
  --expr_w_root 0.05 \
  --expr_w_lower 0.00
```

Run inference with high-expressiveness retrieval:

```bash
export EDGE_UNIT_MIN_EXPRESSIVENESS=0.42
export EDGE_UNIT_EXPRESSIVENESS_BONUS=0.25
export EDGE_UNIT_MIN_ENERGY=-1
export EDGE_UNIT_ENERGY_BONUS=0
```

If feet slide too much, rebuild DB with lower `--expr_w_root`, or raise `EDGE_UNIT_CONTACT_PHASE_WEIGHT`.

## Phase 2: Music tension / phase-aware planner

Generate or load a plan with tension fields:

```bash
python music_choreo_planner.py \
  --audio_feature test_music_bank/dunhuangwu3.npy \
  --num_frames 150 \
  --out output/choreo_plan/dunhuangwu3_tension_plan.json \
  --style_hint "敦煌舞，飞天感，上肢舒展，重心稳定"
```

Enable tension-aware weighting:

```bash
export EDGE_CHOREO_PLAN_JSON=output/choreo_plan/dunhuangwu3_tension_plan.json
export EDGE_TENSION_AWARE_PLANNER=1
export EDGE_UNIT_EXPRESSIVENESS_BONUS=0.12
export EDGE_UNIT_MIN_EXPRESSIVENESS=-1
```

The planner will dynamically raise expressiveness in `attack`, preserve trajectory/diversity in `flow`, and restore stability/contact/entry-exit weights in `pose`.

## Phase 3: Homogeneity penalty for long sequences

Only enable for 240+ frame generations:

```bash
export EDGE_UNIT_HOMOGENEITY_WEIGHT=0.25
export EDGE_UNIT_HOMOGENEITY_MIN_FRAMES=240
```

## Phase 4: 45-frame soft unit prior

Apply the generate_controlled patch and copy `choreorag_unit_prior.py` into the repo root:

```bash
cp choreorag_unit_prior.py /home/disk/lsm/storage/EDGE/
cd /home/disk/lsm/storage/EDGE
git apply /path/to/generate_controlled_phase4.diff
```

Enable weak upper-body temporal prior:

```bash
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_STRENGTH=0.06
export EDGE_UNIT_PRIOR_FEATURES=upper
export EDGE_UNIT_PRIOR_MAX_LEN=45
export EDGE_UNIT_PRIOR_DECAY_GAMMA=1.0
```

Recommended with:

```bash
--mid_keyframe_strength 0.12
--infer_keyframe_width 0
--no_tto
```

## Diagnostics

After generation, run:

```bash
python planner_edit_loop.py \
  --motion output/your_motion.npy \
  --plan_json output/your_auto_mid_plan.json \
  --target_traj output/your_target_traj.npy \
  --out output/diagnostics_expr.json
```

Track:

- `retrieved_expressiveness_mean`
- `retrieved_energy_mean`
- `generated_upper_activity`
- `generated_lower_activity`
- `generated_root_speed`
- `transition_jerk`
- `contact_phase_break`
- `freezing_score`
