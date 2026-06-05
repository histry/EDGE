# EDGE-Dunhuang: Quality-Gated ChoreoRAG for Dunhuang Dance Motion Generation

This repository extends the original EDGE music-conditioned dance diffusion framework toward controllable 3D Dunhuang dance choreography generation.

The current project focus is **not** strict paired Dunhuang music-to-dance generation. Because a large, clean, temporally aligned Dunhuang music-motion dataset is not available, the system is positioned as:

> Low-resource, weak-music-conditioned, retrieval-augmented and support-aware 3D Dunhuang dance choreography generation with quality-gated temporal priors.

In Chinese:

> 面向低资源敦煌舞场景的弱音乐条件、文本桥接、检索增强、支撑感知、质量门控的 3D 短段落编舞生成。

---

## 1. Current Research Direction

Earlier experiments tried to improve controllability by stacking many conditions:

```text
trajectory control
+ start/end/mid keyframes
+ beat guidance
+ TextBridge / Text-Pose RAG
+ postprocess footlock / trajectory anchor
```

This made the system controllable in isolated metrics, but visual results often suffered from:

- hard snapping near middle keyframes;
- root dragging along S-curve trajectories;
- lower body not following the root path;
- static or near-static free DDPM samples;
- high event-score retrieval that still produced tail freeze or root-drag;
- postprocess results that looked better than the native generated motion.

The latest direction is therefore:

```text
45-frame temporal unit reconstruction
→ FK-visible / support-chain diagnosis
→ HF-event-aware motion-unit retrieval
→ retrieved temporal prior guided sampling
→ support-prior quality gate
→ quality-gated ChoreoRAG rerank
```

The current core method name is:

```text
Support-Prior Quality-Gated ChoreoRAG
```

or:

```text
QC-filtered Support-aware Retrieved Temporal Prior
```

---

## 2. Current Main Conclusion

The latest experiments show:

```text
V3I free DDPM sampling still collapses to near-static average motion.
```

However:

```text
V3I + QC-filtered support-shift temporal prior + body_no_rootxz guided sampling
significantly improves visible root path, lower/upper activity, contact switch and visual motion quality.
```

Therefore, the immediate next step is **not** to blindly train V3J. The next step is to formalize the prior selection process:

```text
Support-Prior Quality Gate
→ automatic prior pool
→ ChoreoRAG rerank
→ controlled ablation
→ only then V3J training if the gate is reliable
```

---

## 3. Motion Representation

The current Dunhuang motion representation is 151-dimensional:

```text
[0:4]    foot contact channels
[4:7]    root xyz
[7:151]  24 joints × 6D rotations
```

Important index convention:

```text
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
trajectory plane = physical X/Z ground-plane
```

Do **not** treat image-plane X/Y as the physical trajectory plane.

---

## 4. Key Code Components

### 4.1 `train.py`

The training entrypoint now supports explicit experiment profiles:

```bash
EDGE_TRAIN_PROFILE=legacy
EDGE_TRAIN_PROFILE=v3_unit_recon
```

The current V3/V3H/V3I route should use:

```bash
export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
```

This profile intentionally avoids the old patch-heavy path and installs only the clean temporal-unit route:

```text
edge_safety_patch
edge_recon_contract_patch
unit_reconstruction_patch
v3_loss_stability_patch
v3c_visible_fk_patch
v3f_body_centered_response_patch
hf_event_contrastive_patch
```

It also disables accidental sparse controls:

```text
audio_pairing_mode = none
mmr_loss_weight = 0
disable_traj_cond = True
trajectory loss = 0
keyframe / mid-keyframe prob = 0
beat guidance = 0
energy loss = 0
root-lower coupling = 0
```

### 4.2 `dataset/dance_dataset.py`

The Dunhuang dataset loader supports:

- strict 151D motion contract;
- direct loading from `motion`, `motion_151`, `poses`, or `unit_motions_physical`;
- original-source-level train/validation split to avoid leakage from overlapping windows;
- X/Z ground-plane trajectory enforcement.

### 4.3 `unit_reconstruction_patch.py`

This patch changes the training objective from sparse endpoint/keyframe-driven control to clean 45-frame temporal unit reconstruction.

It:

- removes keyframe / trajectory / beat / RAG controls during V3 training;
- supports `EDGE_X0_RECON_LOSS`;
- adds DCT low-frequency temporal reconstruction;
- encourages the model to learn a full motion phrase rather than isolated poses.

### 4.4 `v3c_visible_fk_patch.py`

This patch addresses an important failure mode:

```text
151D feature-space motion can increase while the FK-rendered skeleton still looks static.
```

It adds FK-visible diagnosis and losses for:

- upper / torso / hand visible range;
- lower-body support-chain motion;
- foot range and foot speed;
- contact switching;
- robust Huber losses and per-term caps.

### 4.5 `model/model.py`

The model still supports trajectory-conditioned generation in the legacy path. The trajectory branch encodes:

```text
[X, Z, ΔX, ΔZ] → trajectory tokens
```

and includes:

- trajectory projection;
- trajectory encoder;
- per-layer trajectory adapters;
- root trajectory generator.

For the current V3 unit reconstruction route, trajectory is deliberately disabled because motion-unit quality must be fixed before reintroducing long-range spatial control.

### 4.6 `model/diffusion.py`

The diffusion module supports:

- epsilon prediction and optional x0 reconstruction loss;
- keyframe constraints;
- trajectory loss;
- contact / FK / foot-sliding / anti-freeze / energy / stability losses;
- classifier-free guidance and energy CFG.

For strict reconstruction sanity tests, use:

```bash
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.8
```

---

## 5. Experiment History Summary

### 5.1 Early controllable generation

Completed:

- start/end keyframe control;
- auto-mid keyframe planning;
- MMR-RAG / ChoreoRAG retrieval;
- trajectory anchor / TTO / Leg IK;
- beat guidance;
- TextBridge and Text/Pose Context RAG.

Main conclusion:

```text
Adding more control signals does not automatically improve motion quality.
Sparse controls often improve local metrics but cause snapping, root dragging, freezing, or unnatural transitions.
```

### 5.2 V7 / V8 / V8B

| Stage | Goal | Conclusion |
|---|---|---|
| V7 lower-fixed | fix lower velocity / root-lower coupling | more stable, still low motion diversity |
| V8 high-activity | increase action strength | activity increased but jerk exploded |
| V8B smooth | balance activity and smoothness | more stable, still not enough for full choreography |

### 5.3 V9 / V10 / V11

Completed:

- RAG Summary Token;
- Unified Choreo Planner;
- TextBridge candidate filtering;
- Text/Pose Context RAG adapter training path.

Main conclusion:

```text
The RAG injection path is connected, but soft context alone does not guarantee physical support execution.
```

### 5.4 V12 / V13

Completed:

- source-file split fix;
- X/Z trajectory contract;
- functional choreography metrics;
- support-expression / turn-expression diagnosis.

Main conclusion:

```text
The core challenge is trajectory-support-expression coupling:
root movement, weight shift, foot support and Dunhuang upper-body expression must become one choreographic event.
```

### 5.5 V3 / V3H / V3I

Current latest route:

```text
V3 clean 45-frame temporal unit reconstruction
→ V3C FK-visible diagnosis
→ V3H support-chain learning
→ HF-event retrieval
→ V3I HF-event support continuation
→ V15 support-prior quality gate
```

Current best baseline before V15:

```text
checkpoint = V3I train-120
prior      = support_shift u32781
mask       = body_no_rootxz
strength   = 0.35
start_frac = 0.70
gamma      = 1.3
```

---

## 6. Recommended Next Plan: V15 Support-Prior Quality RAG

### 6.1 Build support quality sidecar

Target file:

```text
data/dunhuang_choreo_unit_rag/index_v14_hf_event_u45_s15.support_quality_gate.npz
```

Recommended fields:

```text
support_prior_quality
good_gate
hf_event_score_norm
tail_activity_ratio
root_lower_ratio
jump_p95
jerk_p95
root_path
lower_activity
torso_activity
upper_activity
contact_switch
support_context_score
functional_coupling_score
reject_reason
```

### 6.2 Export support-quality prior pool

Target directory:

```text
data/dunhuang_bvh/support_quality_prior_pool_v15_u45/
```

Suggested subsets:

```text
top12_support_shift
top24_support_shift
top50_mixed_support_expr
top500_train_candidate
```

### 6.3 V15 guided sampling matrix

Recommended first experiment:

```text
top12 support-quality priors
× checkpoint e80 / e120
× mask body_no_rootxz / torso_upper
× strength 0.35
= 48 rendered videos
```

Pass criteria:

```text
at least 8/12 priors are visually acceptable;
no obvious line-like degeneration;
no obvious tail-freeze;
no root-drag without support;
quality score correlates with human visual ranking;
body_no_rootxz_s0.35 remains stable.
```

### 6.4 Rerank formula

A recommended ChoreoRAG rerank formula:

```text
final_score =
    w_text      * text_score
  + w_hf        * hf_event_score_norm
  + w_support   * support_context_score
  + w_quality   * support_prior_quality
  + w_func      * functional_coupling_score
  - w_jump      * jump_penalty
  - w_jerk      * jerk_penalty
  - w_freeze    * tail_freeze_penalty
  - w_rootdrag  * root_drag_penalty
```

Initial weights can be:

```text
w_text     = 0.15
w_hf       = 0.15
w_support  = 0.25
w_quality  = 0.30
w_func     = 0.15
w_jump     = 0.20
w_jerk     = 0.20
w_freeze   = 0.35
w_rootdrag = 0.35
```

These weights should be validated by ablation and human scoring.

---

## 7. Ablation Protocol

| Group | Description | Expected result |
|---|---|---|
| A | V3I free DDPM | near-static collapse |
| B | HF-event top prior only | event-strong but may freeze/root-drag |
| C | support-quality prior only | better support stability |
| D | HF-event + support quality gate | better candidate stability |
| E | D + body_no_rootxz guided sampling | current main method |

Metrics:

```text
root_path
lower_activity
torso_activity
upper_activity
contact_switch
jump_p95
jerk_p95
tail_activity_ratio
root_lower_ratio
support_expression_coupling
turn_expression_response
human visual score
```

---

## 8. When to Start V3J Training

Do **not** start V3J blindly.

Start V3J only if:

```text
V15 auto top12/top24 mostly looks normal;
support_prior_quality matches human visual ranking;
bad priors are filtered automatically;
top prior pool no longer depends on manual selection;
body_no_rootxz_s0.35 is stable across e80/e120.
```

Then V3J should train on:

```text
support-quality-gated top500 / top1000 temporal units
```

rather than the whole HF-event boosted dataset.

---

## 9. Claim Boundaries

Safe claims:

- This is a low-resource, weak-music-conditioned Dunhuang dance generation prototype.
- The system uses music rhythm / onset / energy / high-frequency event cues as weak timing signals.
- Dunhuang style is mainly injected through motion-unit retrieval, TextBridge and temporal priors.
- FK-visible and support-chain metrics are necessary because 151D feature metrics can be misleading.
- Quality-gated temporal priors currently outperform free DDPM sampling in visible motion quality.

Unsafe claims:

- The model learns strict Dunhuang music-to-dance mapping.
- Postprocessed ADE=0 proves native trajectory control.
- HF-event top retrieval is always a good prior.
- Free DDPM sampling already generates stable Dunhuang dance.
- Hard middle keyframes are enough for choreographic motion.

---

## 10. Suggested Paper Titles

```text
Quality-Gated ChoreoRAG: Support-aware Temporal Prior Retrieval for Low-resource Dunhuang Dance Generation
```

```text
Support-aware Retrieved Temporal Priors for Low-resource Classical Dance Motion Generation
```

```text
From Keyframes to Quality-Gated Temporal Priors: Retrieval-Augmented Dunhuang Dance Motion Generation
```

---

## 11. Immediate TODO

```text
[ ] Freeze V3I train-120 + support_shift + body_no_rootxz + s0.35 baseline
[ ] Build support quality sidecar
[ ] Export top12/top24/top500 support-quality prior pools
[ ] Run 48 V15 guided sampling videos
[ ] Render and manually score videos
[ ] Check correlation between quality score and visual ranking
[ ] Write support-prior quality into ChoreoRAG rerank
[ ] Run A/B/C/D/E ablation
[ ] Start V3J only after V15 passes stability criteria
```

---

## 12. Summary

The project is currently not bottlenecked by another stronger loss or another control signal. The bottleneck is:

```text
which retrieved temporal prior is safe and useful for diffusion sampling?
```

Therefore, the latest direction is:

```text
Support-Prior Quality-Gated ChoreoRAG
```

This should be the main README direction until V15 is validated.
