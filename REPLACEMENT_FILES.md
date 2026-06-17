# V34 Boundary-Compatible Event-RAG Patch

This patch targets the failure mode where V34 strict warp succeeds but adjacent
retrieved snippets are physically incompatible, so Contact-INR falls back or
cannot repair a hard pose/contact/velocity jump.

## Files to copy into the EDGE repository

Copy these files to the same relative paths under your EDGE root:

- `tools/v34_warp_aware_retrieval.py`
- `tools/v34_boundary_compatibility.py`
- `tools/v34_gpu_candidate_cache.py`
- `scripts/launch_v34_boundary_compat.sh`

Optional installer from the patch directory:

```bash
python install_v34_boundary_compat_patch.py --edge_root /home/disk/lsm/storage/EDGE
```

The installer backs up overwritten files to:

```text
/home/disk/lsm/storage/EDGE/backup_v34_boundary_compat_YYYYMMDD_HHMMSS
```

## Default research policy

The new retrieval policy is enabled by default:

```bash
export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_HARD_PRUNE=1
export V34_BOUNDARY_COMPAT_WEIGHT=1.20
export V34_USE_GPU_RETRIEVAL=1
```

Main hard gates:

```bash
export V34_COMPAT_MAX_POSE_JUMP=0.42
export V34_COMPAT_MAX_VELOCITY_JUMP=0.060
export V34_COMPAT_MAX_ACCELERATION_JUMP=0.120
export V34_COMPAT_MAX_CONTACT_JUMP=0.62
export V34_COMPAT_MAX_YAW_GAP_DEG=62
export V34_COMPAT_MAX_TRANSITION_COST=0.95
```

For ablation:

```bash
export V34_BOUNDARY_COMPAT=0
```

For soft penalty only:

```bash
export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_HARD_PRUNE=0
```

## Recommended launch

Fast checkpoint reuse:

```bash
cd /home/disk/lsm/storage/EDGE
bash scripts/launch_v34_boundary_compat.sh
```

Full retraining / overnight research run:

```bash
cd /home/disk/lsm/storage/EDGE
V34_TRAIN=1 bash scripts/launch_v34_boundary_compat.sh
```

## What to inspect after the run

Each selected schedule part now records:

- `boundary_compat_enabled`
- `boundary_compat_hard_prune`
- `boundary_compat_score`
- `boundary_compat_meta`
- `transition_meta.boundary_compatibility`

Useful checks:

```bash
RUN_ROOT=$(cat output/LATEST_V34_1_INFERENCE_LAUNCH.txt 2>/dev/null || cat output/LATEST_V34_OVERNIGHT_LAUNCH.txt)
grep -nE "compat_rejected|Unsafe V34|Traceback|RuntimeError|\\[SAVED\\]|\\[PASS\\]" "$RUN_ROOT"/run.log "$RUN_ROOT"/launcher.log 2>/dev/null | tail -120
find "$RUN_ROOT" -type f -name "*.schedule_report.json" -o -name "*.boundary_v34.json" | sort
```

The expected improvement is not that Contact-INR becomes stronger.  The expected
improvement is that hard-incompatible snippet pairs are removed before
Contact-INR, so the post-handshake absolute gate sees fewer catastrophic
boundaries and fewer unsafe fallbacks.
