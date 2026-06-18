# V34 Graceful Degradation + Deferred Veto Replacement Files

These files are meant to be copied into `/home/disk/lsm/storage/EDGE` with the
same relative paths.

## Files

- `tools/schedule_v34_whole_song.py`
- `tools/v34_boundary_inpainting.py`
- `scripts/launch_v34_inpaint_blend.sh`
- `scripts/resume_v34_inference_v33ckpt.sh`
- `scripts/run_v34_full_research.sh`
- `scripts/launch_v34_graceful_pipeline.sh`

## What Changed

1. Boundary inpainting is now protected from a destructive post-inpaint
   handshake. If `_adaptive_exit_handshake` makes the absolute boundary gate
   unsafe, the scheduler falls back to the pre-handshake inpainted sequence and
   records the rejected handshake in the schedule report.
2. Unsafe boundary vetoes are deferred by default instead of terminating the
   whole run. The report records `v34_deferred_vetoes`, so bad slots remain
   auditable.
3. Inpainting acceptance now allows physically meaningful improvements, not
   only a strict max-risk-ratio pass.
4. Inference now prefers the newest V34 full-rebuild checkpoint before falling
   back to the old V33 checkpoint.
5. The full research script writes explicit stage markers and checks whether
   both `.npy` and `.schedule_report.json` artifacts exist for every target.
6. All delivered shell scripts use LF line endings.

## Replace On Server

```bash
cd /home/disk/lsm/storage/EDGE

cp /path/to/replacement/tools/schedule_v34_whole_song.py tools/schedule_v34_whole_song.py
cp /path/to/replacement/tools/v34_boundary_inpainting.py tools/v34_boundary_inpainting.py
cp /path/to/replacement/scripts/launch_v34_inpaint_blend.sh scripts/launch_v34_inpaint_blend.sh
cp /path/to/replacement/scripts/resume_v34_inference_v33ckpt.sh scripts/resume_v34_inference_v33ckpt.sh
cp /path/to/replacement/scripts/run_v34_full_research.sh scripts/run_v34_full_research.sh
cp /path/to/replacement/scripts/launch_v34_graceful_pipeline.sh scripts/launch_v34_graceful_pipeline.sh

chmod +x scripts/launch_v34_inpaint_blend.sh \
  scripts/resume_v34_inference_v33ckpt.sh \
  scripts/run_v34_full_research.sh \
  scripts/launch_v34_graceful_pipeline.sh
```

## Verify After Replacement

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  tools/schedule_v34_whole_song.py \
  tools/v34_boundary_inpainting.py

bash -n \
  scripts/launch_v34_inpaint_blend.sh \
  scripts/resume_v34_inference_v33ckpt.sh \
  scripts/run_v34_full_research.sh \
  scripts/launch_v34_graceful_pipeline.sh
```

## Recommended Inference, Reuse Latest V34 Checkpoint

```bash
cd /home/disk/lsm/storage/EDGE

tmux new-session -d -s v34_graceful_infer "bash -lc '
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
export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_HARD_PRUNE=1
export V34_BOUNDARY_INPAINT=1
export V34_LATENT_SNIPPET_BLEND=1
export V34_USE_GPU_RETRIEVAL=1
export V34_FAIL_ON_UNSAFE_BOUNDARY=0
export V34_DEFER_UNSAFE_BOUNDARY=1
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE=1

export RUN_ID=v34_graceful_infer_\$(date +%Y%m%d_%H%M%S)
bash scripts/launch_v34_graceful_pipeline.sh
'"
```

## Full Rebuild + Retrain

```bash
cd /home/disk/lsm/storage/EDGE

tmux new-session -d -s v34_graceful_full "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V34_TRAIN=1
export V34_BUILD_EVENT_LIBRARY=1
export V34_OVERWRITE_EVENT_LIBRARY=1
export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_HARD_PRUNE=1
export V34_BOUNDARY_INPAINT=1
export V34_LATENT_SNIPPET_BLEND=1
export V34_USE_GPU_RETRIEVAL=1
export V34_FAIL_ON_UNSAFE_BOUNDARY=0
export V34_DEFER_UNSAFE_BOUNDARY=1
export V34_HANDSHAKE_FALLBACK_ON_UNSAFE=1

export RUN_ID=v34_graceful_full_rebuild_\$(date +%Y%m%d_%H%M%S)
bash scripts/launch_v34_graceful_pipeline.sh
'"
```

## Monitor

```bash
cd /home/disk/lsm/storage/EDGE

RUN_ROOT=$(cat output/LATEST_V34_GRACEFUL_PIPELINE.txt)
echo "RUN_ROOT=$RUN_ROOT"

grep -nE "\[STAGE START\]|\[STAGE DONE\]|\[STAGE ERROR\]|\[V34 CKPT\]|graceful_degradation|deferred_veto|PASS|DONE|ERROR|Traceback|RuntimeError|SAVED" \
  "$RUN_ROOT/run.log" "$RUN_ROOT/outer.log" 2>/dev/null | tail -200

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
