#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

if [ -f /home/disk/lsm/conda_envs/edge/bin/activate ]; then
  source /home/disk/lsm/conda_envs/edge/bin/activate
fi

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

# ===== 共享 4090：尽量关掉额外推理分支，降低显存 =====
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=0
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENERGY_COND=0
export EDGE_DISABLE_TRAJ_COND=1
export EDGE_FINAL_KEYFRAME_PROJECT=1

EXP_DIR="runs/train_nextgen/v15_onset_phrase_safe_recon_si20_20260526_153307"
CKPT="$EXP_DIR/weights/train-260.pt"

RUN_ROOT="output/night_v15_onset_phrase_20260526_001018"
PRIOR=$(find "$RUN_ROOT" -name "dunhuangwu2_v15_onset_phrase_prior.npy" | head -1)

if [ -z "${PRIOR:-}" ]; then
  PRIOR=$(find "$RUN_ROOT" -name "*_v15_onset_phrase_prior.npy" | head -1)
fi

if [ ! -f "$CKPT" ]; then
  echo "ERROR: checkpoint not found: $CKPT"
  exit 1
fi

if [ ! -f "$PRIOR" ]; then
  echo "ERROR: prior not found under $RUN_ROOT"
  exit 1
fi

EVAL_ROOT="output/eval_v15_ck260_minimal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_ROOT"/poses "$EVAL_ROOT"/motions "$EVAL_ROOT"/logs "$EVAL_ROOT"/reports

echo "============================================================"
echo "Minimal V15 eval"
echo "CKPT=$CKPT"
echo "PRIOR=$PRIOR"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "============================================================"

python - "$PRIOR" "$EVAL_ROOT/poses/start.npy" "$EVAL_ROOT/poses/end.npy" <<'PY'
import sys
import numpy as np
from pathlib import Path

prior, start_out, end_out = map(Path, sys.argv[1:])
m = np.load(prior, allow_pickle=True)
if m.ndim == 0 and isinstance(m.item(), dict):
    obj = m.item()
    m = obj.get("motion", obj.get("pose", m))
m = np.asarray(m, dtype=np.float32)
assert m.ndim == 2 and m.shape[1] == 151, m.shape
np.save(start_out, m[0].astype(np.float32))
np.save(end_out, m[-1].astype(np.float32))
print("saved start/end pose")
PY

python generate_controlled.py \
  --checkpoint "$CKPT" \
  --music test_music_bank/dunhuangwu2.wav \
  --start_pose "$EVAL_ROOT/poses/start.npy" \
  --end_pose "$EVAL_ROOT/poses/end.npy" \
  --out "$EVAL_ROOT/motions/dunhuangwu2_ck260.npy" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 150 \
  --num_frames 150 \
  --mixed_precision fp16 \
  --sampler ddim \
  --guidance_weight 1.0 \
  --pose_space physical \
  --disable_traj_cond \
  --no_tto \
  --no_ema \
  --trajectory "0,0;0,0" \
  --root_xz_reference "$PRIOR" \
  --save_normalized_motion \
  2>&1 | tee "$EVAL_ROOT/logs/generate_ck260.log"

python - "$EVAL_ROOT/motions/dunhuangwu2_ck260.npy" "$PRIOR" "$EVAL_ROOT/reports/metrics.json" <<'PY'
import sys, json
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def load(p):
    x = np.load(p, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        obj = x.item()
        x = obj.get("motion", obj.get("pose", x))
    return np.asarray(x, dtype=np.float32)

def metric(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1)

    out = {
        "frames": int(len(m)),
        "root_max_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "root_final_x": float(root[-1, 0]),
        "root_final_z": float(root[-1, 1]),
        "global_root_jump_p95": float(np.percentile(droot, 95)) if len(droot) else 0.0,
        "global_rot_jump_p95": float(np.percentile(drot, 95)) if len(drot) else 0.0,
        "segment_activity_mean": float(drot.mean()) if len(drot) else 0.0,
    }

    for b in [35, 70, 74, 105, 108, 140, 142]:
        if 2 <= b < len(m) - 2:
            lo = max(1, b - 2)
            hi = min(len(m), b + 3)
            local = np.linalg.norm(rot[lo:hi] - rot[lo-1:hi-1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local.max())

    return out

gen = load(sys.argv[1])
prior = load(sys.argv[2])

report = {
    "generated_path": sys.argv[1],
    "prior_path": sys.argv[2],
    "generated": metric(gen),
    "prior": metric(prior),
}

Path(sys.argv[3]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

echo "DONE"
echo "EVAL_ROOT=$EVAL_ROOT"
