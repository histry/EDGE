# EDGE NextGen V11 Experiments

This patch bundle adds four next-generation experimental directions:

1. Differentiable Contact Loss
2. Beat-guided Sampling
3. Adaptive Planner / stronger Beam Search
4. Explicit Cross-Attention RAG

## 1. Differentiable Contact Loss

Current EDGE already has contact regression and FK-based foot sliding loss. This
bundle replaces the foot sliding term with an explicit differentiable contact
velocity objective:

```text
L_contact = mean C_t * ||v_foot_world||^2
```

Run training with:

```bash
EDGE_DIFF_CONTACT_LOSS=1 \
EDGE_DCL_USE_FK_CONTACT_LABELS=0 \
EDGE_DCL_HORIZONTAL_ONLY=1 \
python train.py \
  --train_stage adapter \
  --foot_loss_weight 0.5 \
  ...
```

Use `EDGE_DCL_USE_FK_CONTACT_LABELS=1` only after confirming FK-derived labels
are stable on your dataset.

## 2. Beat-guided Sampling

Inference only. No retraining required.

```bash
EDGE_BEAT_GUIDANCE=1 \
EDGE_BEAT_GUIDANCE_WEIGHT=0.02 \
EDGE_BEAT_MASK_QUANTILE=0.88 \
python generate_v11_choreo.py ...
```

Start with small weights. Large weights can improve BeatAlign but may increase
jerk.

## 3. Adaptive Planner

Use beam as default for formal experiments:

```bash
EDGE_V10_SEARCH_METHOD=beam \
EDGE_V10_BEAM_WIDTH=8 \
EDGE_V10_JERK_PENALTY=1 \
EDGE_V10_JERK_PENALTY_WEIGHT=0.6 \
EDGE_V10_JERK_PENALTY_SCALE=12.0 \
EDGE_V10_ADAPTIVE_PLANNER=1 \
python generate_v11_choreo.py ...
```

The current planner does not pass per-frame metadata into `transition_cost`;
therefore adaptive frame-aware behavior is provided as a scaffold and becomes
active when `weights["target_frame"]` is available in future planner calls.

## 4. Explicit Cross-Attention RAG

V10 already appends retrieved context tokens to decoder memory. V11 adds an
additional explicit per-layer cross-attention branch.

```bash
EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
EDGE_V11_CROSS_ATTN_RAG=1 \
EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=1 \
EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0 \
python train.py --train_stage adapter --adapter_train_decoder ...
```

For inference:

```bash
EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
EDGE_V11_CROSS_ATTN_RAG=1 \
python generate_v11_choreo.py ...
```

Because the gate is zero-initialized by default, this branch needs adapter
training before claiming improvement.

## Recommended ablations

- Contact loss off/on
- Beat guidance weight: 0, 0.01, 0.02, 0.05
- Greedy vs beam
- Nonlinear jerk 0.35 vs 0.6
- V10 memory context vs V11 cross-attention RAG
