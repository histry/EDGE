# V46.43 Physics-Consistent Stability

This package is a direct replacement / extension package for the EDGE V46.42 chain.
It includes V46.38 routing, V46.41 stage-anchored TGT, V46.42 stability alignment,
and V46.43 physics-consistent corrections.

## Problems fixed

### 1. Tweedie derivative noise amplification
Early-abort no longer treats high-order derivative spikes on a half-denoised
Tweedie/intermediate probe as a fatal signal by itself.  V46.43 uses:

- low-pass filtered probe for oracle decisions only;
- robust p95/p99 derivative statistics instead of raw max;
- relaxed early-abort thresholds;
- multiple probe points;
- consecutive fatal low-frequency barriers before abort;
- derivative-only barriers are diagnostics by default.

### 2. MSA rubber-band / moonwalk paradox
Stage anchoring no longer drags root coordinates to the prior frame-by-frame.
V46.43 applies only low-frequency drift correction, gates out high-energy/leap
frames, and caps the velocity of the anchor correction.

### 3. HN-DPO static mode collapse
The V46.43 HN-DPO fine-tuning tool adds kinetic and motion-density preservation:

- KE(pred) must not fall below a ratio of max(KE(snapshot), KE(preferred));
- motion density must remain close to preferred transition;
- velocity-shape loss discourages freezing or lazy-dancer behavior.

## Main files

```text
tools/apply_v46_43_physics_consistent_stability_patch.py
tools/v46_43_train_hn_dpo_diffusion.py
tools/v46_43_verify_physics_stability_report.py
scripts/run_v46_43_physics_consistent_full_retrain.sh
```

The package also includes the V46.38/V46.41/V46.42 tools required by the run script.

## Install

```bash
cd /home/disk/lsm/storage/EDGE
unzip -o /mnt/data/EDGE_V46_43_PHYSICS_CONSISTENT_STABILITY_solution.zip -d /mnt/data/
cp /mnt/data/v46_43_solution/tools/*.py tools/
cp /mnt/data/v46_43_solution/scripts/*.sh scripts/
chmod +x tools/apply_v46_43_physics_consistent_stability_patch.py tools/v46_43_*.py scripts/run_v46_43_physics_consistent_full_retrain.sh
```

## Full retrain

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export V46_DEVICE=cuda
export PYTHONUNBUFFERED=1

export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V26_V23_CKPT="./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt"

nohup bash scripts/run_v46_43_physics_consistent_full_retrain.sh \
  > output/v46_43_physics_consistent_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Important switches

```bash
export V46_43_EARLY_ABORT_LOWPASS_SIGMA=2.25
export V46_43_EARLY_ABORT_RELAX=4.0
export V46_43_EARLY_ABORT_CONSECUTIVE_FATAL=2
export V46_43_EARLY_ABORT_ALLOW_DERIVATIVE_ONLY_FATAL=0

export V46_43_MSA_LEAP_SPEED_THRESH=0.070
export V46_43_MSA_LEAP_DILATE=4
export V46_43_MSA_LEAP_MIN_GATE=0.0
export V46_43_MSA_CORRECTION_LOWPASS_SIGMA=10.0
export V46_43_MSA_MAX_CORRECTION_VEL_MPF=0.006

export V46_43_ENABLE_HN_DPO_FINETUNE=0
export V46_43_HN_DPO_STEPS=1800
export V46_43_HN_DPO_KINETIC_WEIGHT=0.22
export V46_43_HN_DPO_MOTION_DENSITY_WEIGHT=0.18
```

For the first main experiment, keep HN-DPO disabled.  Generate hard-negative
pairs first, then fine-tune in a second pass.

## Verify

```bash
python tools/v46_43_verify_physics_stability_report.py \
  --report "$RUN_ROOT/dunhuangwu2_v46_43_diffusion_ik.report.json" \
  --require_v46_38_routing \
  --require_v46_42_metadata \
  --require_v46_43_metadata \
  --out "$RUN_ROOT/V46_43_PHYSICS_STABILITY_VERIFY.json"
```
