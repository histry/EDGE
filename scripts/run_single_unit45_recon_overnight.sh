#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/disk/lsm/storage/EDGE

RUN_NAME="strict_single_unit45_recon_v11_3000steps_b1"
DATA_DIR="data/dunhuang_bvh/single_unit45_recon_physical"
ASSET_ENV="output/single_unit45_recon/asset_paths.env"
OUT_DIR="output/single_unit45_recon"
MUSIC="test_music_bank/dunhuangwu2.wav"
METRICS="$OUT_DIR/${RUN_NAME}_metrics.csv"

mkdir -p "$DATA_DIR" "$OUT_DIR/videos" logs

source "$ASSET_ENV"

echo "========== CONFIG =========="
echo "RUN_NAME=$RUN_NAME"
echo "GT_CLIP=$GT_CLIP"
echo "GT_ROOTLOCK=$GT_ROOTLOCK"
echo "START_POSE=$START_POSE"
echo "END_POSE=$END_POSE"
echo "MID_POSES=$MID_POSES"
echo "MID_FRAMES=$MID_FRAMES"
echo "SEQ_LEN=${SEQ_LEN:-45}"

echo "========== STEP 0: GPU BEFORE =========="
nvidia-smi || true

echo "========== STEP 1: make single-unit pkl dataset =========="
python - <<'PY'
import pickle
from pathlib import Path
import numpy as np

gt = Path("output/single_unit45_recon/gt_clip.npy")
out_dir = Path("data/dunhuang_bvh/single_unit45_recon_physical")
out_dir.mkdir(parents=True, exist_ok=True)

clip = np.load(gt).astype(np.float32)
assert clip.shape == (45, 151), f"expected [45,151], got {clip.shape}"

payload = {
    # 多写几个常见键，兼容 DunhuangDataset 的不同读取逻辑
    "motion": clip,
    "motion_151": clip,
    "poses": clip,
    "pose": clip,
    "data": clip,
    "unit_motions_physical": clip,
    "original_filename": "rag_unit55370_physical",
    "source_file": "rag_unit55370_physical",
    "source": "rag_unit55370_physical",
    "name": "rag_unit55370_physical",
    "unit_id": 55370,
    "seq_len": 45,
    "note": "single 45-frame physical RAG unit for strict reconstruction sanity",
}

with open(out_dir / "rag_unit55370_physical.pkl", "wb") as f:
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ wrote", out_dir / "rag_unit55370_physical.pkl")
print("clip shape:", clip.shape)
print("root path xz len:", float(np.linalg.norm(np.diff(clip[:, [4,6]], axis=0), axis=1).sum()))
PY

echo "========== STEP 2: dataset smoke =========="
EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1 \
EDGE_DUNHUANG_STRICT_SPLIT=1 \
EDGE_TRAJECTORY_PLANE=xz \
python - <<'PY'
from dataset.dance_dataset import DunhuangDataset

ds = DunhuangDataset(
    "data/dunhuang_bvh/single_unit45_recon_physical",
    train=True,
    seq_len=45,
    audio_dim=803,
    return_traj=False,
    audio_pairing_mode="none",
)
print("✅ dataset len:", len(ds))
x, cond, fn, wav = ds[0]
print("✅ sample motion:", tuple(x.shape))
print("✅ cond keys:", sorted(cond.keys()))
assert tuple(x.shape) == (45, 151), tuple(x.shape)
PY

echo "========== STEP 3: render GT again =========="
python render_from_npy.py \
  --motion "$GT_CLIP" \
  --audio "$MUSIC" \
  --output "$OUT_DIR/videos/gt_unit45_fixed.mp4" \
  --camera_mode fixed \
  2>&1 | tee "logs/${RUN_NAME}_render_gt_fixed.log" || true

python render_from_npy.py \
  --motion "$GT_ROOTLOCK" \
  --audio "$MUSIC" \
  --output "$OUT_DIR/videos/gt_unit45_rootlock_follow.mp4" \
  --camera_mode follow \
  2>&1 | tee "logs/${RUN_NAME}_render_gt_rootlock_follow.log" || true

echo "========== STEP 4: choose base checkpoint =========="
BASE_CKPT="runs/train_nextgen/v11_inplace_expr_e3_b1_nockpt_quick/weights/train-3.pt"
if [ ! -f "$BASE_CKPT" ]; then
  BASE_CKPT="runs/train_nextgen/v11_stationary_subset_overfit_e30_b1_allow1src/weights/train-30.pt"
fi
if [ ! -f "$BASE_CKPT" ]; then
  BASE_CKPT="runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt"
fi
if [ ! -f "$BASE_CKPT" ]; then
  echo "❌ No base checkpoint found. Edit BASE_CKPT in this script."
  exit 1
fi
echo "BASE_CKPT=$BASE_CKPT"

echo "========== STEP 5: train 45-frame strict overfit for 3000 effective steps =========="
rm -rf "runs/train_nextgen/${RUN_NAME}"

unset EDGE_DIFF_CONTACT_LOSS || true
unset EDGE_BEAT_GUIDANCE || true
unset EDGE_UNIT_SOFT_PRIOR || true
unset EDGE_UNIT_PRIOR_REQUIRED || true

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1 \
EDGE_DUNHUANG_STRICT_SPLIT=1 \
EDGE_TRAJECTORY_PLANE=xz \
EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1 \
EDGE_AUDIO_DEVICE=cpu \
EDGE_DYNAMIC_TRAJ_CFG=0 \
EDGE_GAIT_PHASE_COND=0 \
EDGE_GAIT_CONTACT_LOSS=0 \
EDGE_TRAJ_PHYSICS_FEATURES=0 \
EDGE_TRAJ_FOURIER_FEATURES=0 \
EDGE_TRAJ_SPARSE_WAYPOINT=0 \
EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
EDGE_ENABLE_RAG_SUMMARY_TOKEN=1 \
EDGE_V11_CROSS_ATTN_RAG=1 \
EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=0 \
EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0 \
EDGE_TEXT_CONTEXT_TRAIN_SELF=1 \
EDGE_TEXT_CONTEXT_REQUIRE_GRAD=1 \
EDGE_TEXT_CONTEXT_MIN_GRAD_NORM=1e-10 \
python train.py \
  --project runs/train_nextgen \
  --exp_name "$RUN_NAME" \
  --train_stage adapter \
  --adapter_train_decoder \
  --checkpoint "$BASE_CKPT" \
  --data_path "$DATA_DIR" \
  --seq_len 45 \
  --feature_type hybrid \
  --audio_dim 803 \
  --audio_pairing_mode none \
  --batch_size 1 \
  --epochs 3000 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --enable_rag_summary_token \
  --rag_summary_drop_prob 0.0 \
  --cond_drop_prob 0.0 \
  --keyframe_condition_prob 1.0 \
  --keyframe_condition_width 1 \
  --keyframe_loss_weight 5.0 \
  --mid_keyframe_condition_prob 1.0 \
  --mid_keyframe_count 3 \
  --mid_keyframe_condition_width 1 \
  --mid_keyframe_selection motion_peak \
  --trajectory_loss_weight 0.0 \
  --trajectory_velocity_loss_weight 0.0 \
  --disable_traj_cond \
  --traj_aug_prob 0.0 \
  --energy_condition_prob 0.0 \
  --energy_loss_weight 0.0 \
  --mmr_loss_weight 0.0 \
  --contact_loss_weight 0.0 \
  --foot_loss_weight 0.0 \
  --sync_loss_weight 0.0 \
  --root_lower_coupling_loss_weight 0.0 \
  --save_interval 250 \
  --val_batches 0 \
  --max_train_batches 1 \
  --train_num_workers 0 \
  --val_num_workers 0 \
  --mixed_precision bf16 \
  2>&1 | tee "logs/${RUN_NAME}_train.log"

echo "========== STEP 6: write metric helper =========="
cat > scripts/rootlock_and_recon_metrics_unit45.py <<'PY'
import argparse
import csv
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def path_len_xz(m):
    return float(np.linalg.norm(np.diff(m[:, [ROOT_X, ROOT_Z]], axis=0), axis=1).sum())

def diff_norm(m, sl):
    if len(m) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(m[:, sl], axis=0), axis=-1).mean())

def jerk(m, sl):
    if len(m) < 3:
        return 0.0
    j = m[2:, sl] - 2 * m[1:-1, sl] + m[:-2, sl]
    return float(np.linalg.norm(j, axis=-1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out_rootlock", required=True)
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    m = np.load(args.motion).astype(np.float32)
    gt = np.load(args.gt).astype(np.float32)

    n = min(len(m), len(gt))
    m = m[:n]
    gt = gt[:n]

    rl = m.copy()
    rl[:, ROOT_X] = rl[0, ROOT_X]
    rl[:, ROOT_Z] = rl[0, ROOT_Z]
    Path(args.out_rootlock).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_rootlock, rl)

    row = {
        "tag": args.tag,
        "motion": args.motion,
        "T": n,
        "root_path_raw": path_len_xz(m),
        "root_path_rootlock": path_len_xz(rl),
        "rot_activity_raw": diff_norm(m, ROT),
        "rot_activity_rootlock": diff_norm(rl, ROT),
        "rot_jerk_raw": jerk(m, ROT),
        "rot_jerk_rootlock": jerk(rl, ROT),
        "mse_all_vs_gt": float(np.mean((m - gt) ** 2)),
        "mse_rot_vs_gt": float(np.mean((m[:, ROT] - gt[:, ROT]) ** 2)),
        "mse_rootxz_vs_gt": float(np.mean((m[:, [ROOT_X, ROOT_Z]] - gt[:, [ROOT_X, ROOT_Z]]) ** 2)),
        "nan_count": int(np.isnan(m).sum()),
    }

    csv_path = Path(args.metrics_csv)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("✅ metrics:", row)

if __name__ == "__main__":
    main()
PY

echo "========== STEP 7: controlled reconstruction generation =========="
rm -f "$METRICS"

for E in 250 500 1000 1500 2000 2500 3000; do
  CKPT="runs/train_nextgen/${RUN_NAME}/weights/train-${E}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "skip missing checkpoint $CKPT"
    continue
  fi

  for POSE_SPACE in physical normalized; do
    TAG="e${E}_${POSE_SPACE}"
    OUT_NPY="$OUT_DIR/${RUN_NAME}_${TAG}.npy"
    ROOTLOCK_NPY="$OUT_DIR/${RUN_NAME}_${TAG}_rootlock.npy"
    GEN_LOG="logs/${RUN_NAME}_gen_${TAG}.log"

    echo "----- generate $TAG -----"

    set +e
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1 \
    EDGE_AUDIO_DEVICE=cpu \
    EDGE_DYNAMIC_TRAJ_CFG=0 \
    EDGE_GAIT_PHASE_COND=0 \
    EDGE_GAIT_CONTACT_LOSS=0 \
    EDGE_TRAJ_PHYSICS_FEATURES=0 \
    EDGE_TRAJ_FOURIER_FEATURES=0 \
    EDGE_TRAJ_SPARSE_WAYPOINT=0 \
    EDGE_ENABLE_TEXT_CONTEXT_RAG=1 \
    EDGE_ENABLE_RAG_SUMMARY_TOKEN=1 \
    EDGE_V11_CROSS_ATTN_RAG=1 \
    EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=0 \
    EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0 \
    EDGE_STRICT_EXPERIMENT_GUARD=1 \
    python generate_controlled_v9.py \
      --checkpoint "$CKPT" \
      --music "$MUSIC" \
      --start_pose "$START_POSE" \
      --end_pose "$END_POSE" \
      --mid_poses "$MID_POSES" \
      --mid_pose_frames "$MID_FRAMES" \
      --out "$OUT_NPY" \
      --feature_type hybrid \
      --audio_dim 803 \
      --seq_len 45 \
      --num_frames 45 \
      --sampler ddim \
      --no_tto \
      --endpoint_keyframe_strength 1.0 \
      --mid_keyframe_strength 0.80 \
      --infer_keyframe_width 0 \
      --pose_space "$POSE_SPACE" \
      --mixed_precision bf16 \
      2>&1 | tee "$GEN_LOG"
    STATUS=${PIPESTATUS[0]}
    set -e

    if grep -E "kept newly initialized keys|ignored unexpected keys" "$GEN_LOG" >/dev/null 2>&1; then
      echo "⚠️ WARNING: checkpoint architecture mismatch found in $GEN_LOG"
    fi

    if [ "$STATUS" -ne 0 ] || [ ! -f "$OUT_NPY" ]; then
      echo "⚠️ generation failed for $TAG"
      continue
    fi

    python scripts/rootlock_and_recon_metrics_unit45.py \
      --motion "$OUT_NPY" \
      --gt "$GT_CLIP" \
      --out_rootlock "$ROOTLOCK_NPY" \
      --metrics_csv "$METRICS" \
      --tag "$TAG" \
      2>&1 | tee "logs/${RUN_NAME}_metrics_${TAG}.log"

    python render_from_npy.py \
      --motion "$OUT_NPY" \
      --audio "$MUSIC" \
      --output "$OUT_DIR/videos/${RUN_NAME}_${TAG}_fixed.mp4" \
      --camera_mode fixed \
      2>&1 | tee "logs/${RUN_NAME}_render_${TAG}_fixed.log" || true

    python render_from_npy.py \
      --motion "$ROOTLOCK_NPY" \
      --audio "$MUSIC" \
      --output "$OUT_DIR/videos/${RUN_NAME}_${TAG}_rootlock_follow.mp4" \
      --camera_mode follow \
      2>&1 | tee "logs/${RUN_NAME}_render_${TAG}_rootlock_follow.log" || true
  done
done

echo "========== DONE =========="
echo "Videos:"
ls -lh "$OUT_DIR/videos" || true
echo
echo "Metrics:"
cat "$METRICS" || true
echo
echo "Logs:"
echo "logs/${RUN_NAME}_train.log"
echo "logs/${RUN_NAME}_gen_*.log"
