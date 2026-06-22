# EDGE V34 Rhythm Repair Replacement Files

Base repo checked: https://github.com/histry/EDGE.git
Base commit: a224308bf0cc60ccaf571ffc58612e1eec62d634

## Why this patch exists

Dense Boundary fixed many hard physical seam failures, but the generated full-song result exposed a new retrieval-level failure: safe low-motion snippets dominate the beam. The typical symptom is: the dancer moves quickly during the first second of a slot, then holds a pose for the remaining slot; several `pose_hold`, `calm_flow`, and `neutral_flow` slots in a row look like 6-15 seconds of visual stagnation.

This patch adds a pure inference-time rhythm degradation penalty. It does not require rebuilding the V34 event library and does not require retraining.

## Files to replace/copy

Copy these files into `/home/disk/lsm/storage/EDGE` preserving paths:

- `tools/v34_warp_aware_retrieval.py`
- `scripts/launch_v34_inpaint_blend.sh`
- `scripts/launch_v34_graceful_pipeline.sh`
- `scripts/launch_v34_semantic_router.sh`
- `scripts/launch_v34_rhythm_repair.sh`  (new file)

## Main code changes

### 1. Retrieval-time rhythm descriptors

`tools/v34_warp_aware_retrieval.py` now computes, for each candidate snippet:

- `rhythm_mean_energy`
- `rhythm_p90_energy`
- `rhythm_first1s_energy_ratio`
- `rhythm_tail_mean_energy`

These are computed from already-loaded motion tensors, so no database rebuild is needed.

### 2. Motion density preserving penalty

The score receives three anti-collapse terms:

- `hold_penalty`: penalizes candidates with high first-second energy ratio and low tail energy.
- `density_penalty`: penalizes long slots whose mean motion energy is too low.
- `streak_penalty`: penalizes consecutive static tags: `pose_hold,calm_flow,neutral_flow`.

The schedule report records all debug fields under both top-level slot fields and `transition_meta.rhythm_degradation`.

### 3. Shortlist-level and beam-level protection

The candidate-local `hold_penalty` and `density_penalty` are subtracted before top-k shortlist pruning, preventing static snippets from blocking better candidates before the beam search loop.

The history-dependent `streak_penalty` is applied during beam expansion, so repeated pose-hold chains are penalized path-wise.

## Important environment variables

Default launcher values:

```bash
export V34_RHYTHM_DEGRADATION_PENALTY=1
export V34_RHYTHM_WEIGHT=1.0
export V34_HOLD_PENALTY_WEIGHT=3.50
export V34_STREAK_PENALTY_WEIGHT=2.00
export V34_DENSITY_PENALTY_WEIGHT=4.00
export V34_HOLD_FIRST1S_RATIO_LIMIT=0.65
export V34_HOLD_TAIL_ENERGY_LIMIT=0.020
export V34_MIN_SLOT_MEAN_ENERGY=0.015
export V34_STATIC_STREAK_ALLOW=2
export V34_STATIC_EVENT_TAGS=pose_hold,calm_flow,neutral_flow
```

For the dedicated rhythm-repair inference launcher, boundary weight defaults to:

```bash
export V34_BOUNDARY_COMPAT_WEIGHT=1.15
```

This keeps Dense Boundary active but reduces its tendency to over-prefer low-motion safe snippets.

## Verify after replacement

On the Linux EDGE server:

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile tools/v34_warp_aware_retrieval.py
bash -n \
  scripts/launch_v34_inpaint_blend.sh \
  scripts/launch_v34_graceful_pipeline.sh \
  scripts/launch_v34_semantic_router.sh \
  scripts/launch_v34_rhythm_repair.sh
```

## Recommended first run: pure inference rhythm repair

```bash
cd /home/disk/lsm/storage/EDGE

tmux kill-session -t v34_rhythm_repair 2>/dev/null || true

tmux new-session -d -s v34_rhythm_repair "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export RUN_ID=v34_rhythm_repair_infer_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p \"\$RUN_ROOT\"
echo \"\$RUN_ROOT\" > output/LATEST_V34_RHYTHM_REPAIR.txt

bash scripts/launch_v34_rhythm_repair.sh 2>&1 | tee -a \"\$RUN_ROOT/outer.log\"
'"

RUN_ROOT=$(cat output/LATEST_V34_RHYTHM_REPAIR.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -f "$RUN_ROOT/outer.log"
```

## Morning inspection

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_RHYTHM_REPAIR.txt)

find "$RUN_ROOT" -type f \
  \( -name "*.schedule_report.json" \
     -o -name "*.boundary_v34.json" \
     -o -name "*.contact_metrics.json" \
     -o -name "*.jitter.json" \
     -o -name "*.mp4" \) | sort
```

Check whether the previously bad ranges are now lower in `rhythm_first1s_energy_ratio`, `rhythm_degradation_penalty`, and static streak count.
