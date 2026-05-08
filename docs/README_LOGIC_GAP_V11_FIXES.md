# EDGE V11.1 logic-gap fixes

This package addresses four current logic gaps:

1. **Foot sliding / physical realism**
   - DCL no longer blindly trusts dense target contact channels.
   - Use `EDGE_DCL_CONTACT_SOURCE=auto`.

2. **BeatAlign**
   - Beat-guided sampling now supports full-body energy and scheduled guidance.

3. **Planner target-frame propagation**
   - Adaptive planner reads `target_frame` when available and infers it from caller stack as a bridge.

4. **Root/lower conflict**
   - Dynamic trajectory blending now uses a tolerance band so contact frames can deviate slightly from the global target trajectory.

## Replace

```bash
cp losses/contact_loss.py /home/disk/lsm/storage/EDGE/losses/contact_loss.py
cp differentiable_contact_loss_patch.py /home/disk/lsm/storage/EDGE/differentiable_contact_loss_patch.py
cp beat_guided_sampling_patch.py /home/disk/lsm/storage/EDGE/beat_guided_sampling_patch.py
cp v10_choreo_planner_formal_patch.py /home/disk/lsm/storage/EDGE/v10_choreo_planner_formal_patch.py
cp postprocess_footlock.py /home/disk/lsm/storage/EDGE/postprocess_footlock.py
```

## DCL training

```bash
EDGE_DIFF_CONTACT_LOSS=1 \
EDGE_DCL_CONTACT_SOURCE=auto \
EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85 \
EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height \
EDGE_DCL_HORIZONTAL_ONLY=1 \
EDGE_DCL_VERBOSE=1 \
python train.py ...
```

If logs show `contact_source=auto_fallback_pred_fk_height`, target contact labels were too dense and the patch is working.

## Beat guidance

```bash
EDGE_BEAT_GUIDANCE=1 \
EDGE_BEAT_GUIDANCE_WEIGHT=0.03 \
EDGE_BEAT_GUIDANCE_TARGET=1.35 \
EDGE_BEAT_GUIDANCE_FEATURES=all \
python generate_v11_choreo.py ...
```

## Adaptive planner

```bash
EDGE_V10_JERK_PENALTY=1 \
EDGE_V10_ADAPTIVE_PLANNER=1 \
EDGE_V10_ADAPTIVE_NEAR_POSE_SCALE=3.0 \
EDGE_V10_ADAPTIVE_NEAR_JERK_SCALE=0.1 \
EDGE_V10_SEARCH_METHOD=beam \
EDGE_V10_BEAM_WIDTH=8 \
python generate_v11_choreo.py ...
```

## Dynamic trajectory tolerance

```bash
EDGE_DYNAMIC_TRAJ_BLEND=1 \
EDGE_TRAJ_KEEP_CONTACT=0.02 \
EDGE_TRAJ_KEEP_AIR=0.35 \
EDGE_TRAJ_TOLERANCE_ENABLE=1 \
EDGE_TRAJ_TOLERANCE_M=0.04 \
EDGE_TRAJ_TOLERANCE_CONTACT_M=0.08
```
