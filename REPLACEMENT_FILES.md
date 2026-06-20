# EDGE V34 Dynamic Relax Replacement - 2026-06-20

This replacement package is based on GitHub `histry/EDGE` main commit
`c3757bf4f3cfae82371158eea772cd1191ee7892`.

## What Changed

- `tools/v34_warp_aware_retrieval.py`
  - Adds adaptive constraint relaxation for Event-RAG retrieval.
  - The scheduler first keeps strict candidates only.
  - If the strict feasible set is empty, it uses candidates rejected by semantic/contact/boundary hard gates as soft-penalized rescue branches.
  - Writes `transition_meta.constraint_relaxation`, `semantic_relaxed`, `compat_relaxed`, and `contact_relaxed`.

- `tools/v34_boundary_inpainting.py`
  - Forces masked boundary inpainting when a boundary was saved by adaptive relaxation.

- `tools/schedule_v34_whole_song.py`
  - Adds policy flags to `schedule_report.json`.

- launch scripts
  - Keep hard pruning enabled by default.
  - Enable relax-on-empty by default.
  - Keep strict warp active.

## Files To Replace On Server

Copy these files into `/home/disk/lsm/storage/EDGE`:

```text
tools/v34_warp_aware_retrieval.py
tools/v34_boundary_inpainting.py
tools/schedule_v34_whole_song.py
tools/v34_boundary_compatibility.py
tools/v34_gpu_candidate_cache.py
scripts/launch_v34_semantic_router.sh
scripts/launch_v34_graceful_pipeline.sh
scripts/launch_v34_inpaint_blend.sh
scripts/resume_v34_inference_v33ckpt.sh
```

## Verify After Replacement

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  tools/v34_warp_aware_retrieval.py \
  tools/v34_boundary_inpainting.py \
  tools/schedule_v34_whole_song.py \
  tools/v34_boundary_compatibility.py \
  tools/v34_gpu_candidate_cache.py

bash -n \
  scripts/launch_v34_semantic_router.sh \
  scripts/launch_v34_graceful_pipeline.sh \
  scripts/launch_v34_inpaint_blend.sh \
  scripts/resume_v34_inference_v33ckpt.sh
```

## Recommended Overnight Run

This run reuses the existing checkpoint and event library first. Do this before full retraining.

```bash
cd /home/disk/lsm/storage/EDGE

tmux kill-session -t v34_dynamic_relax_overnight 2>/dev/null || true

tmux new-session -d -s v34_dynamic_relax_overnight "bash -lc '
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

export V34_RELAX_CONSTRAINTS_ON_EMPTY=1
export V34_RELAX_COMPAT_ON_EMPTY=1
export V34_RELAX_SEMANTIC_ON_EMPTY=1
export V34_RELAX_COMPAT_PENALTY_WEIGHT=5.00
export V34_RELAX_SEMANTIC_PENALTY_WEIGHT=4.50
export V34_RELAX_CONTACT_PENALTY_WEIGHT=2.00

export V34_WARP_HARD_PRUNE=1
export V34_WARP_MIN=0.92
export V34_WARP_MAX=1.12
export V34_WARP_RELAX_ON_EMPTY=1
export V34_WARP_RELAX_MIN=0.82
export V34_WARP_RELAX_MAX=1.30

export V34_BOUNDARY_INPAINT=1
export V34_INPAINT_ON_RELAXED_CONSTRAINT=1
export V34_LATENT_SNIPPET_BLEND=1
export V34_USE_GPU_RETRIEVAL=1
export V34_FAIL_ON_UNSAFE_BOUNDARY=0
export V34_DEFER_UNSAFE_BOUNDARY=1
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE=1

export RUN_ID=v34_dynamic_relax_overnight_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p \"\$RUN_ROOT\"
echo \"\$RUN_ROOT\" > output/LATEST_V34_DYNAMIC_RELAX_OVERNIGHT.txt

echo \"[DYNAMIC RELAX START] \$(date) RUN_ROOT=\$RUN_ROOT\" | tee -a \"\$RUN_ROOT/outer.log\"
bash scripts/launch_v34_semantic_router.sh 2>&1 | tee -a \"\$RUN_ROOT/outer.log\"
'"

sleep 5
RUN_ROOT=$(cat output/LATEST_V34_DYNAMIC_RELAX_OVERNIGHT.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -f "$RUN_ROOT/outer.log"
```

## Morning Inspection

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_DYNAMIC_RELAX_OVERNIGHT.txt)
echo "RUN_ROOT=$RUN_ROOT"

grep -nE "V34-RELAX|constraint_relax|semantic_relaxed|contact_relaxed|PASS|DONE|ERROR|Traceback|RuntimeError|SAVED" \
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

## Count Relaxed Boundaries

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_DYNAMIC_RELAX_OVERNIGHT.txt)

python - <<'PY'
import json, pathlib
run = pathlib.Path(open("output/LATEST_V34_DYNAMIC_RELAX_OVERNIGHT.txt").read().strip())
for p in sorted(run.glob("**/*.schedule_report.json")):
    data = json.loads(p.read_text())
    rows = []
    for part in data.get("schedule", []):
        tm = part.get("transition_meta", {}) or {}
        rel = tm.get("constraint_relaxation", {}) or {}
        if rel.get("used_due_to_empty_strict") or rel.get("active"):
            rows.append({
                "slot": part.get("slot"),
                "event_id": part.get("event_id"),
                "reasons": rel.get("reasons", []),
                "penalty": rel.get("penalty"),
                "semantic_relaxed": tm.get("semantic_relaxed"),
                "compat_relaxed": tm.get("compat_relaxed"),
                "contact_relaxed": tm.get("contact_relaxed"),
            })
    print("\\n==", p, "==")
    print("relaxed_boundaries", len(rows))
    for r in rows[:20]:
        print(r)
PY
```
