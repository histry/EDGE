# EDGE ChoreoRAG V3 replacement files

Copy this folder into the EDGE repository root, preserving paths:

```bash
cp choreorag_unit_prior.py /path/to/EDGE/choreorag_unit_prior.py
cp v10_choreo_planner_formal_patch.py /path/to/EDGE/v10_choreo_planner_formal_patch.py
cp postprocess_footlock.py /path/to/EDGE/postprocess_footlock.py
cp edge_experiment_guard.py /path/to/EDGE/edge_experiment_guard.py
cp generate_controlled_v9.py /path/to/EDGE/generate_controlled_v9.py
cp generate_v10_choreo.py /path/to/EDGE/generate_v10_choreo.py
cp generate_v9_baseline.py /path/to/EDGE/generate_v9_baseline.py
cp scripts/eval_generated_motions.py /path/to/EDGE/scripts/eval_generated_motions.py
cp scripts/validate_formal_run.py /path/to/EDGE/scripts/validate_formal_run.py
mkdir -p /path/to/EDGE/docs
cp docs/human_evaluation_template.md /path/to/EDGE/docs/human_evaluation_template.md
```

## What changed

1. Activity vs. Jerk
   - `v10_choreo_planner_formal_patch.py` replaces linear-only transition cost at runtime with a jerk-proxy nonlinear penalty.
   - It estimates boundary jerk from pose jump, velocity mismatch, contact discontinuity, and root direction discontinuity.

2. DCT Prior
   - `choreorag_unit_prior.py` now supports soft spectral decay:
     - `EDGE_UNIT_PRIOR_DCT_DECAY=soft_exp|hard`
     - `EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH=3.0`
   - Formal V10 wrapper defaults to `soft_exp`.

3. Root-Lower Decoupling
   - `postprocess_footlock.py` keeps the old API but adds contact-aware dynamic trajectory blending.
   - During contact, trajectory pull is weak; during swing, trajectory pull returns to the base value.

4. GPU / Hook Safety
   - `edge_experiment_guard.py` now makes `TextContextGradMonitor` a context manager.
   - It removes hooks in `__exit__`, uses `.item()` for scalar reports, and optionally empties CUDA cache.

5. Evaluation
   - `scripts/eval_generated_motions.py` keeps raw/final separation and reports `foot_slide_proxy_rate` explicitly.
   - `docs/human_evaluation_template.md` adds blind A/B user-study dimensions.

## Formal V10 recommended command

```bash
EDGE_RUN_MODE=formal \
EDGE_EXPERIMENT_PROFILE=v10 \
EDGE_STRICT_EXPERIMENT_GUARD=1 \
EDGE_STRICT_RUNTIME_PATCHES=1 \
EDGE_UNIT_SOFT_PRIOR=1 \
EDGE_UNIT_PRIOR_REQUIRED=1 \
EDGE_UNIT_PRIOR_TEMPORAL=1 \
EDGE_UNIT_PRIOR_DCT=1 \
EDGE_UNIT_PRIOR_DCT_DECAY=soft_exp \
EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH=3.0 \
EDGE_UNIT_PRIOR_LOW_FREQ_K=4 \
EDGE_UNIT_PRIOR_FEATURES=upper+torso \
EDGE_UNIT_PRIOR_STRENGTH=0.006 \
EDGE_V10_JERK_PENALTY=1 \
EDGE_V10_JERK_MODE=quadratic \
EDGE_V10_JERK_THRESHOLD=0.18 \
EDGE_V10_JERK_PENALTY_WEIGHT=0.35 \
EDGE_V10_JERK_PENALTY_SCALE=8.0 \
EDGE_V10_POSE_SAFE_THRESHOLD=0.40 \
EDGE_V10_POSE_QUAD_WEIGHT=5.0 \
EDGE_DYNAMIC_TRAJ_BLEND=1 \
EDGE_TRAJ_KEEP_CONTACT=0.05 \
EDGE_TRAJ_KEEP_AIR=0.35 \
EDGE_TRAJ_BLEND_SMOOTH=5 \
EDGE_FORMAL_AUTO_EVAL=1 \
EDGE_FORMAL_EVAL_REQUIRED=1 \
EDGE_V10_RAG_DB=data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
python generate_v10_choreo.py \
  --checkpoint runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt \
  --start_pose /path/to/start.npy \
  --end_pose /path/to/end.npy \
  --audio /path/to/audio.wav \
  --feature_type hybrid \
  --trajectory '0,0;0.5,0.7;-0.3,1.2;0,1.6' \
  --sampler ddim \
  --out output/v10_formal/demo.npy
```

Validate evidence:

```bash
python scripts/validate_formal_run.py --prefix output/v10_formal/demo --require_context
```

## Clean V9 baseline

```bash
python generate_v9_baseline.py \
  --checkpoint /path/to/v9_checkpoint.pt \
  --start_pose /path/to/start.npy \
  --end_pose /path/to/end.npy \
  --audio /path/to/audio.wav \
  --feature_type hybrid \
  --trajectory '0,0;0.5,0.7;-0.3,1.2;0,1.6' \
  --sampler ddim \
  --out output/v9_baseline/demo.npy
```

## Text/Pose Context RAG adapter training switches

```bash
EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
EDGE_TEXT_CONTEXT_TRAIN_SELF=1 \
EDGE_TEXT_CONTEXT_REPORT_JSON=logs/text_context_report.json \
EDGE_TEXT_CONTEXT_GRAD_JSON=logs/text_context_grad.json \
EDGE_TEXT_CONTEXT_EVIDENCE_JSON=logs/text_context_evidence.json \
EDGE_TEXT_CONTEXT_REQUIRE_GRAD=1 \
EDGE_TEXT_CONTEXT_MIN_GRAD_NORM=1e-10 \
EDGE_EMPTY_CUDA_CACHE_ON_MONITOR_CLOSE=1 \
python train.py --train_stage adapter ...
```

For strict final proof:

```bash
EDGE_TEXT_CONTEXT_REQUIRE_GATE=1 \
EDGE_TEXT_CONTEXT_MIN_GATE_ABS=1e-4
```

## Differentiable contact loss note

A true differentiable foot-contact loss is intentionally not wired into `train.py` in this patch, because it requires confirming the training data's contact labels and differentiable FK path.  Treat it as Future Work / next training-stage work:

`Loss_contact = sum_t C_t * ||V_foot_world^t||^2`.
