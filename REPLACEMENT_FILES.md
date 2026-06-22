# EDGE V34 Dense Boundary Risk Replacement - 2026-06-21

This replacement package is based on the previous dynamic-relax package and the
current `histry/EDGE` V34 code path.  It targets the failure found in the
Dunhuang whole-song boundary clips: visually bad boundaries can still receive a
zero retrieval risk when every metric remains just inside its hard threshold.

## Core Diagnosis

The old boundary score used a ReLU/excess ratio:

```python
max(0.0, value / limit - 1.0)
```

That is correct for hard rejection, but wrong for ranking.  It makes `0.01 *
limit` and `0.99 * limit` both score `0.0`.  Beam search therefore cannot prefer
a safe interior stitch over a near-threshold stitch.

This package decouples the two roles:

- hard feasibility gate: still uses direct threshold checks, e.g. `pose <= pose_limit`.
- ranking score: uses dense convex risk potential, `min((value / limit) ** gamma, cap)`.
- semantic continuity: remains excess/ReLU by default, so natural style flow is not over-penalized.
- inpainting trigger: uses both dense score and near-threshold visual ratios.

## Files To Replace On Server

Copy these files into `/home/disk/lsm/storage/EDGE`:

```text
tools/v34_boundary_compatibility.py
tools/v34_warp_aware_retrieval.py
tools/v34_boundary_inpainting.py
tools/schedule_v34_whole_song.py
tools/v34_gpu_candidate_cache.py
scripts/launch_v34_semantic_router.sh
scripts/launch_v34_graceful_pipeline.sh
scripts/launch_v34_inpaint_blend.sh
scripts/resume_v34_inference_v33ckpt.sh
```

## What Changed

### `tools/v34_boundary_compatibility.py`

- Adds `_excess_ratio()` for hard-gate-compatible excess scoring.
- Adds `_dense_ratio()` and `_ranking_ratio()` for physical boundary ranking.
- Physical terms now use dense score by default:
  - pose
  - velocity
  - acceleration
  - contact
  - contact_binary
  - support_count
  - yaw
  - transition
- Semantic terms use `_excess_ratio()` by default:
  - body
  - activity
  - turn
- Keeps hard checks unchanged.
- Writes diagnostic fields into schedule reports:
  - `score_mode`
  - `dense_power`
  - `dense_cap`
  - `dense_semantic_score`
  - `terms`
  - `excess_terms`

### `tools/v34_warp_aware_retrieval.py`

- Makes semantic scoring explicitly use `_excess_ratio()`.
- Keeps adaptive relax-on-empty from the previous package.
- Adds minimum-violation rescue top-k for relaxed candidates:
  - `V34_RELAX_RESCUE_TOP_K=768`
- This prevents an empty strict feasible set from reintroducing thousands of bad relaxed branches into the beam.

### `tools/v34_boundary_inpainting.py`

- Changes default `V34_INPAINT_COMPAT_SCORE_TRIGGER` from `0.10` to `0.45`.
- Adds visual heuristic triggers based on boundary metric ratios:
  - pose ratio
  - yaw ratio
  - contact ratio
  - transition-cost ratio
- Default trigger ratio is `0.80`, so near-threshold bad boundaries are sent to masked boundary inpainting even when hard checks pass.

### `tools/schedule_v34_whole_song.py`

Adds policy fields to `schedule_report.json`:

```json
"dense_boundary_score": true,
"dense_boundary_power": 2.0,
"dense_boundary_cap": 4.0,
"dense_semantic_score": false,
"inpaint_compat_score_trigger": 0.45,
"inpaint_visual_heuristic": true,
"relax_rescue_top_k": 768
```

### Launch scripts

The four scripts now default to:

```bash
export V34_COMPAT_DENSE_SCORE=1
export V34_COMPAT_DENSE_POWER=2.0
export V34_COMPAT_DENSE_CAP=4.0
export V34_COMPAT_DENSE_SEMANTIC_SCORE=0
export V34_RELAX_RESCUE_TOP_K=768
export V34_INPAINT_COMPAT_SCORE_TRIGGER=0.45
export V34_INPAINT_VISUAL_HEURISTIC=1
export V34_INPAINT_VISUAL_POSE_RATIO=0.80
export V34_INPAINT_VISUAL_YAW_RATIO=0.80
export V34_INPAINT_VISUAL_CONTACT_RATIO=0.80
export V34_INPAINT_VISUAL_TRANSITION_RATIO=0.80
```

## Replace Commands

Run these on the server after copying this package directory there, or copy the
listed files manually.

```bash
cd /home/disk/lsm/storage/EDGE

cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/tools/v34_boundary_compatibility.py tools/v34_boundary_compatibility.py
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/tools/v34_warp_aware_retrieval.py tools/v34_warp_aware_retrieval.py
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/tools/v34_boundary_inpainting.py tools/v34_boundary_inpainting.py
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/tools/schedule_v34_whole_song.py tools/schedule_v34_whole_song.py
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/tools/v34_gpu_candidate_cache.py tools/v34_gpu_candidate_cache.py
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/scripts/launch_v34_semantic_router.sh scripts/launch_v34_semantic_router.sh
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/scripts/launch_v34_graceful_pipeline.sh scripts/launch_v34_graceful_pipeline.sh
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/scripts/launch_v34_inpaint_blend.sh scripts/launch_v34_inpaint_blend.sh
cp -f /PATH/TO/EDGE_V34_DENSE_BOUNDARY_REPLACEMENT_20260621/scripts/resume_v34_inference_v33ckpt.sh scripts/resume_v34_inference_v33ckpt.sh
```

## Verify After Replacement

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  tools/v34_boundary_compatibility.py \
  tools/v34_warp_aware_retrieval.py \
  tools/v34_boundary_inpainting.py \
  tools/schedule_v34_whole_song.py \
  tools/v34_gpu_candidate_cache.py

bash -n \
  scripts/launch_v34_semantic_router.sh \
  scripts/launch_v34_graceful_pipeline.sh \
  scripts/launch_v34_inpaint_blend.sh \
  scripts/resume_v34_inference_v33ckpt.sh
```

## Recommended Overnight Inference Run

This first run reuses the current event library and checkpoint.  Do this before
full retraining because the current failure is a retrieval/inpainting routing
issue, not a proven training-capacity issue.

```bash
cd /home/disk/lsm/storage/EDGE

tmux kill-session -t v34_dense_boundary_overnight 2>/dev/null || true

tmux new-session -d -s v34_dense_boundary_overnight "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V34_TRAIN=0
export V34_BUILD_EVENT_LIBRARY=0
export V34_OVERWRITE_EVENT_LIBRARY=0

export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_HARD_PRUNE=1
export V34_COMPAT_SEMANTIC_HARD_PRUNE=1
export V34_SEMANTIC_EDGE=1
export V34_SEMANTIC_EDGE_HARD_PRUNE=1

export V34_COMPAT_DENSE_SCORE=1
export V34_COMPAT_DENSE_POWER=2.0
export V34_COMPAT_DENSE_CAP=4.0
export V34_COMPAT_DENSE_SEMANTIC_SCORE=0
export V34_BOUNDARY_COMPAT_WEIGHT=1.50
export V34_SEMANTIC_EDGE_WEIGHT=1.25

export V34_RELAX_CONSTRAINTS_ON_EMPTY=1
export V34_RELAX_COMPAT_ON_EMPTY=1
export V34_RELAX_SEMANTIC_ON_EMPTY=1
export V34_RELAX_COMPAT_PENALTY_WEIGHT=5.00
export V34_RELAX_SEMANTIC_PENALTY_WEIGHT=4.50
export V34_RELAX_CONTACT_PENALTY_WEIGHT=2.00
export V34_RELAX_RESCUE_TOP_K=768

export V34_WARP_HARD_PRUNE=1
export V34_WARP_MIN=0.92
export V34_WARP_MAX=1.12
export V34_WARP_RELAX_ON_EMPTY=1
export V34_WARP_RELAX_MIN=0.82
export V34_WARP_RELAX_MAX=1.30

export V34_BOUNDARY_INPAINT=1
export V34_INPAINT_ON_RELAXED_CONSTRAINT=1
export V34_INPAINT_COMPAT_SCORE_TRIGGER=0.45
export V34_INPAINT_VISUAL_HEURISTIC=1
export V34_INPAINT_VISUAL_POSE_RATIO=0.80
export V34_INPAINT_VISUAL_YAW_RATIO=0.80
export V34_INPAINT_VISUAL_CONTACT_RATIO=0.80
export V34_INPAINT_VISUAL_TRANSITION_RATIO=0.80

export V34_LATENT_SNIPPET_BLEND=1
export V34_USE_GPU_RETRIEVAL=1
export V34_FAIL_ON_UNSAFE_BOUNDARY=0
export V34_DEFER_UNSAFE_BOUNDARY=1
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE=1

export RUN_ID=v34_dense_boundary_overnight_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p "\$RUN_ROOT"
echo "\$RUN_ROOT" > output/LATEST_V34_DENSE_BOUNDARY_OVERNIGHT.txt

echo "[DENSE BOUNDARY START] \$(date) RUN_ROOT=\$RUN_ROOT" | tee -a "\$RUN_ROOT/outer.log"
bash scripts/launch_v34_semantic_router.sh 2>&1 | tee -a "\$RUN_ROOT/outer.log"
'"

sleep 5
RUN_ROOT=$(cat output/LATEST_V34_DENSE_BOUNDARY_OVERNIGHT.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -f "$RUN_ROOT/outer.log"
```

## Morning Inspection

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_DENSE_BOUNDARY_OVERNIGHT.txt)
echo "RUN_ROOT=$RUN_ROOT"

grep -nE "DENSE|V34-RELAX|constraint_relax|visual_heuristic|compat_trigger|PASS|DONE|ERROR|Traceback|RuntimeError|SAVED" \
  "$RUN_ROOT/run.log" "$RUN_ROOT/outer.log" 2>/dev/null | tail -260

find "$RUN_ROOT" -type f \
  \( -name "*.schedule_report.json" \
     -o -name "*.boundary_v34.json" \
     -o -name "*.public_metrics.json" \
     -o -name "*.frequency_foot.json" \
     -o -name "*.contact_metrics.json" \
     -o -name "*.jitter.json" \
     -o -name "*.mp4" \
     -o -name "*SUMMARY.json" \) | sort
```

## Quantile Check For Trigger Tuning

Use this after the run to decide whether `V34_INPAINT_COMPAT_SCORE_TRIGGER=0.45`
should move toward `0.35` or `0.60`.

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_DENSE_BOUNDARY_OVERNIGHT.txt)

python - <<'PY'
import json, pathlib, statistics
run = pathlib.Path(open("output/LATEST_V34_DENSE_BOUNDARY_OVERNIGHT.txt").read().strip())
for rpt in sorted(run.glob("**/*.schedule_report.json")):
    data = json.loads(rpt.read_text())
    scores, visual = [], []
    for part in data.get("schedule", []):
        scores.append(float(part.get("boundary_compat_score", 0.0)))
    for i, metrics in enumerate(data.get("boundary_metrics", []), start=1):
        inpaint = metrics.get("v34_boundary_inpainting", {}) or {}
        trig = inpaint.get("trigger", {}) or {}
        if trig.get("visual_heuristic"):
            visual.append(i)
    if not scores:
        continue
    vals = sorted(scores)
    def q(p):
        return vals[min(len(vals)-1, int(round((len(vals)-1)*p)))]
    print("\n==", rpt, "==")
    print("n", len(vals), "q50", round(q(0.50), 4), "q75", round(q(0.75), 4), "q90", round(q(0.90), 4), "q95", round(q(0.95), 4), "max", round(vals[-1], 4))
    print("visual_heuristic_slots", visual[:80], "count", len(visual))
PY
```

## Expected Effect

- The bad-boundary class where `pose/yaw/contact` are close to the limit but still below it should no longer receive `boundary_compat_score=0.0`.
- Some problematic boundaries should be avoided directly by beam search.
- Unavoidable near-threshold boundaries should be sent to masked inpainting through `compat_trigger` or `visual_heuristic`.
- If many slots still trigger visual inpainting, tune in this order:
  1. Lower `V34_BOUNDARY_COMPAT_WEIGHT` from `1.50` to `1.20` if motion becomes too conservative.
  2. Raise `V34_INPAINT_VISUAL_*_RATIO` from `0.80` to `0.88` if inpainting is over-triggered.
  3. Lower `V34_INPAINT_COMPAT_SCORE_TRIGGER` from `0.45` to `0.35` if bad boundary clips remain but are not inpainted.
