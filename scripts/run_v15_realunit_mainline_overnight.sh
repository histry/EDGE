#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

if [ -f /home/disk/lsm/conda_envs/edge/bin/activate ]; then
  source /home/disk/lsm/conda_envs/edge/bin/activate
else
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate edge
fi

export PYTHONPATH=$PWD:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

DATE=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="output/night_v15_realunit_mainline_${DATE}"
LOG_ROOT="logs/night_v15_realunit_mainline_${DATE}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

exec > >(stdbuf -oL -eL tee -a "$LOG_ROOT/master.log") 2>&1

echo "============================================================"
echo "V15 Real-Unit Style Mainline Overnight"
echo "DATE=$DATE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "============================================================"

echo "[1/8] Write real-unit pool builder..."

mkdir -p tools

cat > tools/build_v15_realunit_style_pool.py <<'PY'
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def as_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def find_motion_array(npz):
    candidates = []
    for k in npz.files:
        arr = npz[k]
        if not isinstance(arr, np.ndarray):
            continue

        if arr.ndim == 3 and arr.shape[-1] == 151:
            candidates.append((k, arr))
        elif arr.ndim == 2 and arr.shape[-1] == 151:
            # [N,151] single-frame fallback, less preferred
            candidates.append((k, arr[:, None, :]))

    if not candidates:
        raise RuntimeError("No motion array [N,T,151] or [N,151] found. Keys=" + ",".join(npz.files))

    priority = [
        "unit_motions_physical",
        "motions_physical",
        "motion_151",
        "unit_motions",
        "motions",
        "motion",
        "poses",
    ]

    for name in priority:
        for k, arr in candidates:
            if k == name:
                return k, arr

    candidates.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    return candidates[0]

def crop_or_pad(m, target_len):
    m = np.asarray(m, dtype=np.float32)
    if len(m) == target_len:
        return m
    if len(m) > target_len:
        s = max(0, (len(m) - target_len) // 2)
        return m[s:s + target_len]
    pad = np.repeat(m[-1:], target_len - len(m), axis=0)
    return np.concatenate([m, pad], axis=0)

def motion_metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1) if len(m) > 1 else np.zeros(1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1) if len(m) > 1 else np.zeros(1)

    rot_j = rot.reshape(len(rot), 24, 6)
    lower = rot_j[:, 0:8].reshape(len(rot), -1)
    torso = rot_j[:, 8:14].reshape(len(rot), -1)
    upper = rot_j[:, 14:24].reshape(len(rot), -1)

    def act(x):
        if len(x) <= 1:
            return 0.0
        return float(np.linalg.norm(x[1:] - x[:-1], axis=1).mean())

    root_radius = float(np.linalg.norm(root - root[:1], axis=1).max())
    activity = float(drot.mean())
    upper_activity = act(upper)
    torso_activity = act(torso)
    lower_activity = act(lower)

    return {
        "root_radius": root_radius,
        "root_final": float(np.linalg.norm(root[-1] - root[0])),
        "root_speed_mean": float(droot.mean()),
        "global_rot_jump_p95": float(np.percentile(drot, 95)),
        "global_rot_jump_max": float(drot.max()),
        "activity": activity,
        "upper_activity": upper_activity,
        "torso_activity": torso_activity,
        "lower_activity": lower_activity,
        "contact_switch": float(np.abs(m[1:, CONTACT] - m[:-1, CONTACT]).sum(axis=1).mean()) if len(m) > 1 else 0.0,
    }

def score(met, args):
    if met["root_radius"] > args.max_root_radius:
        return None
    if met["global_rot_jump_p95"] > args.max_rot_jump_p95:
        return None
    if met["activity"] < args.min_activity:
        return None
    if met["upper_activity"] < args.min_upper_activity:
        return None
    if met["torso_activity"] < args.min_torso_activity:
        return None

    expr = 1.50 * met["upper_activity"] + 1.10 * met["torso_activity"] + 0.30 * met["lower_activity"]
    root_safe = np.exp(-5.0 * met["root_radius"])
    jump_safe = np.exp(-1.2 * met["global_rot_jump_p95"])
    non_static = min(met["activity"] / 0.16, 1.0)
    return float(expr * root_safe * jump_safe * (0.35 + 0.65 * non_static))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_len", type=int, default=45)
    ap.add_argument("--top_k", type=int, default=900)
    ap.add_argument("--render_top_k", type=int, default=24)
    ap.add_argument("--audio_dim", type=int, default=803)

    ap.add_argument("--max_root_radius", type=float, default=0.25)
    ap.add_argument("--max_rot_jump_p95", type=float, default=1.10)
    ap.add_argument("--min_activity", type=float, default=0.030)
    ap.add_argument("--min_upper_activity", type=float, default=0.020)
    ap.add_argument("--min_torso_activity", type=float, default=0.008)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    pkl_dir = out_dir / "dunhuang_realunit_style_pkl"
    npy_dir = out_dir / "top_unit_npy"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    motion_key, motions = find_motion_array(npz)
    print(f"motion_key={motion_key}, shape={motions.shape}")

    rows = []
    for i in range(motions.shape[0]):
        m = crop_or_pad(motions[i], args.target_len)
        met = motion_metrics(m)
        sc = score(met, args)
        if sc is None:
            continue
        rows.append((sc, i, m, met))

    rows.sort(key=lambda x: x[0], reverse=True)
    selected = rows[:args.top_k]

    report_items = []
    for rank, (sc, idx, m, met) in enumerate(selected, start=1):
        item = {
            "motion": m.astype(np.float32),
            "motion_151": m.astype(np.float32),
            "poses": m.astype(np.float32),
            "audio_feature": np.zeros((len(m), args.audio_dim), dtype=np.float32),
            "original_filename": f"realunit_idx{idx:06d}",
            "source_file": f"realunit_idx{idx:06d}",
            "metadata": {
                "rank": rank,
                "source_npz": str(args.npz),
                "source_index": int(idx),
                "score": float(sc),
                **met,
            }
        }

        pkl_path = pkl_dir / f"realunit_rank{rank:04d}_idx{idx:06d}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(item, f)

        if rank <= args.render_top_k:
            np.save(npy_dir / f"realunit_rank{rank:04d}_idx{idx:06d}.npy", m[None].astype(np.float32))

        report_items.append({
            "rank": rank,
            "source_index": int(idx),
            "score": float(sc),
            "pkl": str(pkl_path),
            **met,
        })

    report = {
        "npz": str(args.npz),
        "motion_key": motion_key,
        "input_shape": list(motions.shape),
        "target_len": args.target_len,
        "selected": len(selected),
        "pkl_dir": str(pkl_dir),
        "top_unit_npy_dir": str(npy_dir),
        "thresholds": {
            "max_root_radius": args.max_root_radius,
            "max_rot_jump_p95": args.max_rot_jump_p95,
            "min_activity": args.min_activity,
            "min_upper_activity": args.min_upper_activity,
            "min_torso_activity": args.min_torso_activity,
        },
        "items": report_items,
    }

    report_path = out_dir / "realunit_style_pool_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"selected={len(selected)}")
    print(f"pkl_dir={pkl_dir}")
    print(f"top_unit_npy_dir={npy_dir}")
    print(f"report={report_path}")

if __name__ == "__main__":
    main()
PY

python -m py_compile tools/build_v15_realunit_style_pool.py

echo "[2/8] Select source NPZ..."

SRC_NPZ=""
for p in \
  data/dunhuang_choreo_unit_rag/index_v13_functional_u45_s15.npz \
  data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
  data/dunhuang_choreo_unit_rag/index_v12_footstep_u45_s15.npz \
  output/night_v15_onset_phrase_20260526_001018/physics_aware_prior_pool.npz
do
  if [ -f "$p" ]; then
    SRC_NPZ="$p"
    break
  fi
done

if [ -z "$SRC_NPZ" ]; then
  echo "ERROR: no source npz found."
  exit 1
fi

echo "SRC_NPZ=$SRC_NPZ"

echo "[3/8] Build visual/style-first real-unit pool..."

python tools/build_v15_realunit_style_pool.py \
  --npz "$SRC_NPZ" \
  --out_dir "$RUN_ROOT" \
  --target_len 45 \
  --top_k 900 \
  --render_top_k 24 \
  --max_root_radius 0.25 \
  --max_rot_jump_p95 1.10 \
  --min_activity 0.030 \
  --min_upper_activity 0.020 \
  --min_torso_activity 0.008 \
  --audio_dim 803 \
  2>&1 | tee "$LOG_ROOT/build_realunit_pool.log"

DATA_PATH="$RUN_ROOT/dunhuang_realunit_style_pkl"
PKL_COUNT=$(find "$DATA_PATH" -name "*.pkl" | wc -l | tr -d ' ')
echo "PKL_COUNT=$PKL_COUNT"

if [ "$PKL_COUNT" -lt 30 ]; then
  echo "ERROR: too few real units selected. Need loosen thresholds."
  exit 2
fi

echo "[4/8] Render top selected real-unit priors for visual inspection..."

VIS_DIR="$RUN_ROOT/top_unit_renders"
mkdir -p "$VIS_DIR"

count=0
for npy in "$RUN_ROOT"/top_unit_npy/*.npy; do
  [ -f "$npy" ] || continue
  count=$((count + 1))
  stem=$(basename "$npy" .npy)

  python render_from_npy.py \
    --motion "$npy" \
    --audio test_music_bank/dunhuangwu2.wav \
    --output "$VIS_DIR/${stem}_follow.mp4" \
    --camera_mode follow \
    || true

  if [ "$count" -ge 12 ]; then
    break
  fi
done

echo "[5/8] Wait for shared GPU..."

GPU_ID=${GPU_ID:-0}
MIN_GPU_FREE_MB=${MIN_GPU_FREE_MB:-18500}
CHECK_INTERVAL_SEC=${CHECK_INTERVAL_SEC:-300}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-14}

START_TS=$(date +%s)
MAX_WAIT_SEC=$((MAX_WAIT_HOURS * 3600))

while true; do
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))

  GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
  GPU_USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')

  echo "[$(date '+%F %T')] GPU used=${GPU_USED_MB}MiB free=${GPU_FREE_MB}MiB elapsed=${ELAPSED}s"

  if [ "$GPU_FREE_MB" -ge "$MIN_GPU_FREE_MB" ]; then
    echo "GPU free enough."
    break
  fi

  if [ "$ELAPSED" -ge "$MAX_WAIT_SEC" ]; then
    echo "Timeout waiting GPU."
    exit 3
  fi

  echo "GPU busy. Do not kill other users. Sleep ${CHECK_INTERVAL_SEC}s..."
  sleep "$CHECK_INTERVAL_SEC"
done

sleep 10
GPU_FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | head -1 | tr -d ' ')
if [ "$GPU_FREE_MB" -lt "$MIN_GPU_FREE_MB" ]; then
  echo "GPU was occupied again. Exit safely."
  exit 4
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "[6/8] Train real-unit reconstruction model..."

export EDGE_DUNHUANG_STRICT_SPLIT=0
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=0
export EDGE_TRAJECTORY_PLANE=xz

export EDGE_TRAIN_PROFILE=v3_unit_recon
export EDGE_V3_UNIT_RECON=1
export EDGE_X0_RECON_LOSS=1
export EDGE_X0_RECON_LOSS_WEIGHT=0.55
export EDGE_V3_DCT_KEEP=8
export EDGE_V3_TEMPORAL_FEATURES=upper_torso
export EDGE_V3_VELOCITY_WEIGHT=0.04
export EDGE_V3_ACCEL_WEIGHT=0.008

EXP_NAME="v15_realunit_style_recon_${DATE}"

BASE_CKPT=""
for p in \
  runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt \
  runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt
do
  if [ -f "$p" ]; then
    BASE_CKPT="$p"
    break
  fi
done

EXTRA_CKPT_ARGS=()
if [ -n "$BASE_CKPT" ] && python train.py --help 2>&1 | grep -q -- "--checkpoint"; then
  EXTRA_CKPT_ARGS=(--checkpoint "$BASE_CKPT")
  echo "Using base checkpoint: $BASE_CKPT"
else
  echo "No compatible --checkpoint arg or base checkpoint found. Training without base checkpoint."
fi

python train.py \
  "${EXTRA_CKPT_ARGS[@]}" \
  --project runs/train_nextgen \
  --exp_name "$EXP_NAME" \
  --data_path "$DATA_PATH" \
  --processed_data_dir "$RUN_ROOT/dataset_cache" \
  --render_dir "$RUN_ROOT/renders" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --batch_size 4 \
  --epochs 220 \
  --learning_rate 1.5e-4 \
  --weight_decay 0.02 \
  --mixed_precision bf16 \
  --cond_drop_prob 0.10 \
  --audio_pairing_mode none \
  --mmr_loss_weight 0.0 \
  --disable_traj_cond \
  --keyframe_condition_prob 0.0 \
  --keyframe_loss_weight 0.0 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --beat_guidance_weight 0.0 \
  --sync_loss_weight 0.0 \
  --energy_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --contact_loss_weight 0.2 \
  --foot_loss_weight 0.5 \
  --traj_aug_prob 0.0 \
  --save_interval 20 \
  --val_batches 2 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --force_reload \
  --no_cache \
  2>&1 | tee "$LOG_ROOT/train.log"

echo "[7/8] Build next-morning conclusion..."

EXP_DIR="runs/train_nextgen/$EXP_NAME"
CKPT_DIR="$EXP_DIR/weights"

BEST_LINE=$(grep -E "Validation \| Val Loss:" "$LOG_ROOT/train.log" | tail -20 | sort -t':' -k2,2n | head -1 || true)
LAST_CKPT=$(ls -1 "$CKPT_DIR"/train-*.pt 2>/dev/null | sort -V | tail -1 || true)

cat > "$RUN_ROOT/NEXT_MORNING_CONCLUSION.md" <<EOF
# V15 Real-Unit Style Mainline Overnight Conclusion

## Run

- RUN_ROOT: $RUN_ROOT
- LOG_ROOT: $LOG_ROOT
- EXP_NAME: $EXP_NAME
- DATA_PATH: $DATA_PATH
- PKL_COUNT: $PKL_COUNT
- SRC_NPZ: $SRC_NPZ

## Main Goal

This run shifts the project from generated-prior smoothing to real Dunhuang motion-unit learning.

Pipeline:

real Dunhuang motion units
→ visual/style-first prior pool selection
→ real-unit reconstruction training
→ prior-based final render candidate pool

## Selected Pool

The selected real-unit pool is stored in:

$RUN_ROOT/realunit_style_pool_report.json

Top rendered unit videos are stored in:

$RUN_ROOT/top_unit_renders/

Use these videos for visual/style-first selection.

## Training Result

Checkpoint directory:

$CKPT_DIR

Last checkpoint:

$LAST_CKPT

Best recent validation line:

$BEST_LINE

## How to judge tomorrow

1. First inspect top_unit_renders:
   - choose units that look most like Dunhuang dance;
   - reject weak, static, jittery, or non-style units.

2. Then inspect train.log:
   - if val loss steadily decreases and checkpoints exist, training is valid;
   - if render samples are weak, the pool is still useful for prior-based render.

3. Main conclusion expected:
   - If real-unit pool videos are visually better than ck260/SDEdit outputs, the project should continue as visual-first prior selection.
   - If reconstruction samples also retain style, this checkpoint can replace train-260 as a better style-preserving refiner.

## Current recommendation

Do not use ck260 as final generator.
Use real-unit prior pool as the main source of final Dunhuang motion.
EOF

echo "[8/8] Pack reports and top videos..."

zip -j "$RUN_ROOT/next_morning_package.zip" \
  "$RUN_ROOT/NEXT_MORNING_CONCLUSION.md" \
  "$RUN_ROOT/realunit_style_pool_report.json" \
  "$LOG_ROOT/master.log" \
  "$LOG_ROOT/train.log" \
  "$VIS_DIR"/*.mp4 \
  2>/dev/null || true

echo "============================================================"
echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "EXP_NAME=$EXP_NAME"
echo "CHECKPOINT_DIR=$CKPT_DIR"
echo "NEXT_MORNING=$RUN_ROOT/NEXT_MORNING_CONCLUSION.md"
echo "PACKAGE=$RUN_ROOT/next_morning_package.zip"
echo "============================================================"
