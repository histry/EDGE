# V39B Footplant-Gated Contact Stability Patch

This patch is a direct response to the V39 audit where contact ratio became too broad, false long footplants were locked, floor penetration remained rejected, and Butterworth filtering worsened hand jitter.

## New logic

1. Segment-level footplant rejection:
   - Reject contact segments whose XZ anchor error is too large.
   - Reject very long moving-contact segments before root locking.

2. Safer contact confidence:
   - Stricter hysteresis thresholds.
   - Lower label weight; higher height/speed evidence.

3. Gentler root correction:
   - Lower foot-lock strength.
   - Lower root correction step/acceleration limits.
   - Lower support root damping.

4. Floor handling:
   - Lower floor quantile by default to avoid over-estimated floor.
   - Higher but smoother allowed floor lift.

5. Residual Butterworth rollback:
   - Low-pass is wrist/hand only by default.
   - Automatically rolls back if p95 joint or hand jerk gets worse beyond tolerance.

## Install

```bash
cd /mnt/data/V39B_Footplant_Gated_Stability_Patch
mkdir -p /tmp/V39B_Footplant_Gated_Stability_Patch
cp -r * /tmp/V39B_Footplant_Gated_Stability_Patch/

bash install_v39b_footplant_gate_patch.sh /home/disk/lsm/storage/EDGE
```

## Reprocess one result

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v39b_footplant_gate_env.sh

bash scripts/run_v39b_reprocess_motion.sh \
  output/v38_source_aware_full_train_20260625_212818/v34_contact_inr/dunhuangwu2_v26.npy
```

## Key environment switches

```bash
export V39_REJECT_MOVING_FOOTPLANTS=1
export V39_MAX_FOOTPLANT_SEGMENT_ERROR=0.055
export V39_MAX_FOOTPLANT_SEGMENT_P95=0.110
export V39_MAX_FOOTPLANT_SEGMENT_FRAMES=120
export V39_BUTTERWORTH_ROLLBACK_IF_WORSE=1
```
