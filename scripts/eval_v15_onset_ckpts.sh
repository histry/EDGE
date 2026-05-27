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

EXP_DIR="runs/train_nextgen/v15_onset_phrase_safe_recon_si20_20260526_153307"
RUN_ROOT="output/night_v15_onset_phrase_20260526_001018"

EVAL_ROOT="output/eval_v15_onset_ckpts_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_ROOT"/poses "$EVAL_ROOT"/motions "$EVAL_ROOT"/reports "$EVAL_ROOT"/logs

echo "============================================================"
echo "V15 onset checkpoint evaluation"
echo "EXP_DIR=$EXP_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "============================================================"

mapfile -t PRIOR_LIST < <(find "$RUN_ROOT" -name "*_v15_onset_phrase_prior.npy" | sort)

if [ "${#PRIOR_LIST[@]}" -lt 1 ]; then
  echo "ERROR: no *_v15_onset_phrase_prior.npy found under $RUN_ROOT"
  exit 1
fi

echo "Found priors:"
printf '  %s\n' "${PRIOR_LIST[@]}"

cat > "$EVAL_ROOT/eval_motion_metrics.py" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT_SLICE = slice(7, 151)

def load_motion(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        obj = arr.item()
        arr = obj.get("motion", obj.get("pose", arr))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {arr.shape}")
    return arr

def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT_SLICE]
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
            local_rot = np.linalg.norm(rot[lo:hi] - rot[lo-1:hi-1], axis=1)
            local_root = np.linalg.norm(root[lo:hi] - root[lo-1:hi-1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local_rot.max())
            out[f"boundary_{b}_local_root_jump_max"] = float(local_root.max())
    return out

def main():
    paths = [Path(p) for p in sys.argv[1:]]
    rows = []
    for p in paths:
        m = load_motion(p)
        r = metrics(m)
        r["path"] = str(p)
        rows.append(r)
        out = p.with_suffix(".metrics.json")
        out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {p} ===")
        print(json.dumps(r, ensure_ascii=False, indent=2))
    if rows:
        summary = Path(paths[0]).parent / "summary_metrics.json"
        summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved summary: {summary}")

if __name__ == "__main__":
    main()
PY

echo "Extract start/end poses..."
for PRIOR in "${PRIOR_LIST[@]}"; do
  TAG=$(basename "$PRIOR" _v15_onset_phrase_prior.npy)
  python - "$PRIOR" "$EVAL_ROOT/poses/${TAG}_start.npy" "$EVAL_ROOT/poses/${TAG}_end.npy" <<'PY'
import sys
import numpy as np
from pathlib import Path

prior, start_out, end_out = map(Path, sys.argv[1:])
m = np.load(prior, allow_pickle=True)
if m.ndim == 0 and isinstance(m.item(), dict):
    obj = m.item()
    m = obj.get("motion", obj.get("pose", m))
m = np.asarray(m, dtype=np.float32)
assert m.ndim == 2 and m.shape[1] == 151, (prior, m.shape)

np.save(start_out, m[0].astype(np.float32))
np.save(end_out, m[-1].astype(np.float32))
print(f"{prior} -> {start_out}, {end_out}")
PY
done

echo "Evaluate original priors..."
python "$EVAL_ROOT/eval_motion_metrics.py" "${PRIOR_LIST[@]}" | tee "$EVAL_ROOT/reports/prior_metrics.log"

CKPTS=(220 240 260 300)

echo "Run controlled generation for candidate checkpoints..."
for CK in "${CKPTS[@]}"; do
  CKPT="$EXP_DIR/weights/train-${CK}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "Skip missing ckpt: $CKPT"
    continue
  fi

  for PRIOR in "${PRIOR_LIST[@]}"; do
    TAG=$(basename "$PRIOR" _v15_onset_phrase_prior.npy)

    MUSIC="test_music_bank/${TAG}.wav"
    if [ ! -f "$MUSIC" ]; then
      # fallback：有些 tag 可能不是完整音乐名
      MUSIC="test_music_bank/dunhuangwu2.wav"
    fi

    START="$EVAL_ROOT/poses/${TAG}_start.npy"
    END="$EVAL_ROOT/poses/${TAG}_end.npy"
    OUT="$EVAL_ROOT/motions/${TAG}_ck${CK}.npy"

    echo "============================================================"
    echo "CK=$CK"
    echo "PRIOR=$PRIOR"
    echo "MUSIC=$MUSIC"
    echo "OUT=$OUT"
    echo "============================================================"

    export EDGE_FINAL_KEYFRAME_PROJECT=1
    export EDGE_DISABLE_TRAJ_COND=1

    python generate_controlled.py \
      --checkpoint "$CKPT" \
      --music "$MUSIC" \
      --start_pose "$START" \
      --end_pose "$END" \
      --out "$OUT" \
      --feature_type hybrid \
      --audio_dim 803 \
      --seq_len 150 \
      --num_frames 150 \
      --mixed_precision bf16 \
      --sampler ddim \
      --guidance_weight 1.0 \
      --pose_space physical \
      --disable_traj_cond \
      --no_tto \
      --trajectory "0,0;0,0" \
      --root_xz_reference "$PRIOR" \
      --save_normalized_motion \
      2>&1 | tee "$EVAL_ROOT/logs/${TAG}_ck${CK}.log"

  done
done

echo "Evaluate generated motions..."
mapfile -t GEN_LIST < <(find "$EVAL_ROOT/motions" -name "*.npy" ! -name "*_raw.npy" ! -name "*_target_traj.npy" ! -name "*_norm.npy" | sort)

if [ "${#GEN_LIST[@]}" -gt 0 ]; then
  python "$EVAL_ROOT/eval_motion_metrics.py" "${GEN_LIST[@]}" | tee "$EVAL_ROOT/reports/generated_metrics.log"
else
  echo "ERROR: no generated final motions found"
fi

echo "============================================================"
echo "DONE"
echo "EVAL_ROOT=$EVAL_ROOT"
echo "Reports:"
echo "  $EVAL_ROOT/reports/prior_metrics.log"
echo "  $EVAL_ROOT/reports/generated_metrics.log"
echo "============================================================"
