# EDGE Formal Experiment Contracts

Copy these files into the root of `histry/EDGE`.

## Files

- `choreorag_unit_prior.py` — replacement with required temporal-prior assert and JSON report.
- `generate_v10_choreo.py` — replacement V10 wrapper with formal-mode legacy auto-mid blocking and explicit unit prior paths.
- `v10_choreo_planner_formal_patch.py` — new optional runtime patch for nonlinear transition-jerk penalty in planner.
- `edge_text_context_training_fix.py` — replacement with final grad/gate evidence contract.
- `text_context_rag_io_patch.py` — replacement with `normal/no_context/shuffled/wrong_text/zero_text` ablation modes.
- `scripts/eval_generated_motions.py` — new raw-vs-final evaluator.
- `scripts/run_context_rag_ablation.sh` — new four-way context ablation launcher.
- `scripts/analyze_context_rag_ablation.py` — new ablation summary tool.
- `configs/experiment_profiles/*.yaml` — reference env profiles.

## Formal V10 env

```bash
export EDGE_RUN_MODE=formal
export EDGE_EXPERIMENT_PROFILE=v10
export EDGE_STRICT_EXPERIMENT_GUARD=1
export EDGE_STRICT_RUNTIME_PATCHES=1
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_CONTEXT_REQUIRED=1
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_REQUIRED=1
export EDGE_UNIT_PRIOR_TEMPORAL=1
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_LOW_FREQ_K=4
export EDGE_UNIT_PRIOR_FEATURES=upper+torso
export EDGE_UNIT_PRIOR_STRENGTH=0.006
export EDGE_V10_JERK_PENALTY=1
```

## Raw vs final evaluation

```bash
python scripts/eval_generated_motions.py \
  --raw_motion output/demo_raw.npy \
  --final_motion output/demo.npy \
  --target_traj output/demo_target_traj.npy \
  --meta output/demo_meta.json \
  --out output/demo_eval.json \
  --formal
```

## Text/Pose Context RAG ablation

```bash
CHECKPOINT=runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt \
START_POSE=test_keyframes/demo_dyl002_start.npy \
END_POSE=test_keyframes/demo_dyl002_end.npy \
AUDIO=dunhuangwu2.wav \
OUT_DIR=output/context_ablation_wu2 \
EDGE_V10_RAG_DB=data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
bash scripts/run_context_rag_ablation.sh
```
