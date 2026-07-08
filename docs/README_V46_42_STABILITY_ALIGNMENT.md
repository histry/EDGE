# V46.42 Stability Alignment for Stage-Anchored Guided TGT

This package fixes three vulnerabilities found in V46.41:

1. **Tweedie Jitter False Positives**
   - Early-abort no longer applies final strict KBO directly to a noisy intermediate probe.
   - The probe is first low-pass filtered and then checked with relaxed early thresholds.

2. **Rubber-band Effect in Macroscopic Stage Anchoring**
   - MSA strength is now dynamically modulated by MSSD slot semantics, music energy, boundary accent, tension, speed factor, and local root velocity.
   - High-energy/climax/percussive/turning/leap-like windows get weak anchoring; calm/pose/sustained windows get strong anchoring.

3. **Static Mode Collapse in HN-DPO**
   - The HN-DPO fine-tuning tool adds kinetic-energy preservation.
   - The predicted x0 must maintain at least a configurable ratio of reference kinetic energy, default 80%.

## Files

```text
tools/canonicalize_chang_e_bvh_rot_only_meter.py
tools/apply_v46_38_complete_routing_patch.py
tools/apply_v46_41_stage_anchor_guided_tgt_patch.py
tools/apply_v46_42_stability_alignment_patch.py
tools/v46_42_train_hn_dpo_diffusion.py
tools/v46_42_verify_tgt_kbo_report.py
tools/v46_42_extract_hn_dpo_pairs.py
scripts/run_v46_42_stage_anchor_guided_full_retrain.sh
```

## Install

```bash
cd /home/disk/lsm/storage/EDGE
unzip -o /mnt/data/EDGE_V46_42_STABILITY_ALIGNMENT_solution.zip -d /mnt/data/
cp /mnt/data/v46_42_solution/tools/*.py tools/
cp /mnt/data/v46_42_solution/scripts/*.sh scripts/
chmod +x tools/apply_v46_42_stability_alignment_patch.py tools/v46_42_*.py scripts/run_v46_42_stage_anchor_guided_full_retrain.sh
```

## Full run

```bash
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V26_V23_CKPT="./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt"

nohup bash scripts/run_v46_42_stage_anchor_guided_full_retrain.sh \
  > output/v46_42_stage_anchor_guided_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Important switches

```bash
# Early-abort jitter fixes
export V46_42_EARLY_ABORT_KBO_SMOOTH_SIGMA=1.35
export V46_42_EARLY_ABORT_KBO_RELAX=3.0

# Dynamic MSA against rubber-band effect
export V46_42_MSA_HIGH_ENERGY_SCALE=0.22
export V46_42_MSA_DYNAMIC_ATTENUATION=0.75
export V46_42_MSA_ROOT_SPEED_RELAX_THRESH=0.045
export V46_42_MSA_MIN_WEIGHT=0.05

# Optional kinetic HN-DPO
export V46_42_ENABLE_HN_DPO_FINETUNE=1
export V46_42_HN_DPO_STEPS=1500
export V46_42_HN_DPO_KINETIC_WEIGHT=0.18
export V46_42_HN_DPO_KINETIC_FLOOR_RATIO=0.80
```

## Verify

```bash
python tools/v46_42_verify_tgt_kbo_report.py \
  --report "$RUN_ROOT/dunhuangwu2_v46_42_diffusion_ik.report.json" \
  --require_v46_38_routing \
  --require_v46_41_tokens \
  --require_v46_42_metadata \
  --out "$RUN_ROOT/V46_42_TGT_KBO_VERIFY.json"
```
