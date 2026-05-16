#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

source output/single_unit45_recon/asset_paths.env || true

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

EXP="${V15_EXP:-strict_single_unit45_recon_v15_bridge_dense4_bodyrot_10000steps_b1}"
EPOCHS="${V15_EPOCHS:-10000}"
SAVE_INTERVAL="${V15_SAVE_INTERVAL:-1000}"

DATA_PATH="${V15_DATA_PATH:-data/dunhuang_bvh/single_unit45_recon_physical}"
PROCESSED_DATA_DIR="${V15_PROCESSED_DATA_DIR:-data/dataset_backups/}"

echo "============================================================"
echo "V15 clean run start: $(date)"
echo "PWD=$PWD"
echo "EXP=$EXP"
echo "EPOCHS=$EPOCHS"
echo "DATA_PATH=$DATA_PATH"
echo "PROCESSED_DATA_DIR=$PROCESSED_DATA_DIR"
echo "============================================================"

python -m py_compile \
  edge_single_unit_recon_patch.py \
  train_single_recon.py \
  generate_single_recon.py \
  model/diffusion.py \
  EDGE.py

export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=1

export EDGE_ENABLE_TEXT_CONTEXT_RAG=0
export EDGE_V11_CROSS_ATTN_RAG=0
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
export EDGE_AUDIO_DEVICE=cpu

export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_GAIT_PHASE_COND=0
export EDGE_GAIT_CONTACT_LOSS=0
export EDGE_TRAJ_PHYSICS_FEATURES=0
export EDGE_TRAJ_FOURIER_FEATURES=0
export EDGE_TRAJ_SPARSE_WAYPOINT=0
export EDGE_TRAJ_BEV_COND=0

export EDGE_RECON_TRAIN_DENSE_KEYFRAMES=1
export EDGE_RECON_TRAIN_DENSE_STRIDE=4
export EDGE_RECON_TRAIN_ROOT_XZ_ALL=1
export EDGE_RECON_TRAIN_HARD_FEATURES=rot+root_y+contacts

export EDGE_RECON_BRIDGE_COND=1
export EDGE_RECON_BRIDGE_FEATURES=rot+root_y
export EDGE_RECON_BRIDGE_STRENGTH=0.35

export EDGE_RECON_EXTRA_LOSS=1
export EDGE_RECON_EXTRA_X0_W=50
export EDGE_RECON_EXTRA_VEL_W=25
export EDGE_RECON_EXTRA_ACC_W=5
export EDGE_RECON_EXTRA_KEY_NEIGHBOR_W=10
export EDGE_RECON_KEY_NEIGHBOR_RADIUS=1

export EDGE_RECON_LOSS_ROOT_XZ_W=0.0
export EDGE_RECON_LOSS_ROOT_Y_W=1.0
export EDGE_RECON_LOSS_CONTACT_W=0.25
export EDGE_RECON_LOSS_PELVIS_W=4.0
export EDGE_RECON_LOSS_LOWER_W=2.0
export EDGE_RECON_LOSS_TORSO_W=4.0
export EDGE_RECON_LOSS_UPPER_W=8.0
export EDGE_RECON_EXTRA_DEBUG=1

TRAIN_LOG="logs/train_${EXP}.log"

echo "============================================================"
echo "Start clean training: $(date)"
echo "TRAIN_LOG=$TRAIN_LOG"
echo "============================================================"

python train_single_recon.py \
  --project runs/train_nextgen \
  --exp_name "$EXP" \
  --data_path "$DATA_PATH" \
  --processed_data_dir "$PROCESSED_DATA_DIR" \
  --seq_len 45 \
  --batch_size 1 \
  --epochs "$EPOCHS" \
  --save_interval "$SAVE_INTERVAL" \
  --val_batches 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --mixed_precision bf16 \
  --audio_pairing_mode none \
  --cond_drop_prob 0.0 \
  --disable_traj_cond \
  --traj_aug_prob 0.0 \
  --keyframe_condition_prob 1.0 \
  --keyframe_condition_width 1 \
  --keyframe_loss_weight 2.0 \
  --mid_keyframe_condition_prob 1.0 \
  --mid_keyframe_count 10 \
  --mid_keyframe_condition_width 0 \
  --mid_keyframe_selection motion_peak \
  --contact_loss_weight 0.0 \
  --foot_loss_weight 0.0 \
  --sync_loss_weight 0.0 \
  --mmr_loss_weight 0.0 \
  --train_stage full \
  --enable_rag_summary_token \
  --rag_summary_drop_prob 0.0 \
  --hard_keyframe_project \
  --train_num_workers 0 \
  --val_num_workers 0 \
  2>&1 | tee "$TRAIN_LOG"

# IMPORTANT:
# EDGE.py uses increment_path(), so the actual run dir may be EXP, EXP2, EXP3...
# Parse the real checkpoint path from the training log instead of assuming $EXP.
CKPT="$(grep -oP '权重已保存:\s*\K.*train-[0-9]+\.pt' "$TRAIN_LOG" | tail -1 || true)"

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "⚠️ Could not parse checkpoint from log. Falling back to newest train-${EPOCHS}.pt under runs/train_nextgen/${EXP}*"
  CKPT="$(find runs/train_nextgen -path "*/${EXP}*/weights/train-${EPOCHS}.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
  echo "❌ checkpoint not found after training."
  echo "Recent candidate checkpoints:"
  find runs/train_nextgen -path "*/weights/train-*.pt" -type f -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort | tail -20 || true
  exit 1
fi

SAVE_DIR="$(dirname "$(dirname "$CKPT")")"
RUN_NAME="$(basename "$SAVE_DIR")"

echo "============================================================"
echo "Training finished: $(date)"
echo "SAVE_DIR=$SAVE_DIR"
echo "RUN_NAME=$RUN_NAME"
echo "CKPT=$CKPT"
echo "============================================================"

mkdir -p test_keyframes/single_unit45_recon/dense4

python - <<'PY'
import numpy as np
from pathlib import Path

gt = np.load("output/single_unit45_recon/gt_clip.npy").astype("float32")
out = Path("test_keyframes/single_unit45_recon/dense4")
out.mkdir(parents=True, exist_ok=True)

frames = list(range(4, 44, 4))
for f in frames:
    np.save(out / f"mid_{f:03d}.npy", gt[f])

print("dense4 frames:", frames)
PY

MID_POSES="test_keyframes/single_unit45_recon/dense4/mid_004.npy,test_keyframes/single_unit45_recon/dense4/mid_008.npy,test_keyframes/single_unit45_recon/dense4/mid_012.npy,test_keyframes/single_unit45_recon/dense4/mid_016.npy,test_keyframes/single_unit45_recon/dense4/mid_020.npy,test_keyframes/single_unit45_recon/dense4/mid_024.npy,test_keyframes/single_unit45_recon/dense4/mid_028.npy,test_keyframes/single_unit45_recon/dense4/mid_032.npy,test_keyframes/single_unit45_recon/dense4/mid_036.npy,test_keyframes/single_unit45_recon/dense4/mid_040.npy"
MID_FRAMES="4,8,12,16,20,24,28,32,36,40"

export EDGE_HARD_KEYFRAME_PROJECT=1
export EDGE_INFER_PROJECT_XSTART=1
export EDGE_RECON_BRIDGE_COND=1
export EDGE_RECON_BRIDGE_FEATURES=rot+root_y
export EDGE_RECON_BRIDGE_STRENGTH=0.35

OUT_BASE="output/single_unit45_recon_norm/e${EPOCHS}_v15_bridge_dense4_${RUN_NAME}"

python generate_single_recon.py \
  --checkpoint "$CKPT" \
  --no_ema \
  --music test_music_bank/dunhuangwu2.wav \
  --start_pose test_keyframes/single_unit45_recon/start.npy \
  --end_pose test_keyframes/single_unit45_recon/end.npy \
  --mid_poses "$MID_POSES" \
  --mid_pose_frames "$MID_FRAMES" \
  --out "${OUT_BASE}.npy" \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --num_frames 45 \
  --sampler ddim \
  --guidance_weight 1.0 \
  --pose_space physical \
  --endpoint_keyframe_strength 1.0 \
  --mid_keyframe_strength 1.0 \
  --infer_keyframe_width 0 \
  --hard_keyframe_project \
  --infer_project_xstart \
  --disable_traj_cond \
  --root_xz_reference output/single_unit45_recon/gt_clip.npy \
  --no_tto \
  2>&1 | tee "logs/gen_e${EPOCHS}_v15_bridge_dense4_${RUN_NAME}.log"

cat > scripts/eval_v15_bridge_dense4.py <<PY
import numpy as np

path = "${OUT_BASE}.npy"
pred = np.load(path).astype("float32")
gt = np.load("output/single_unit45_recon/gt_clip.npy").astype("float32")

KEYS = {0,4,8,12,16,20,24,28,32,36,40,44}

def root_path(x):
    return float(np.linalg.norm(np.diff(x[:, [4, 6]], axis=0), axis=1).sum())

def rot_idx(joints):
    out = []
    for j in joints:
        s = 7 + 6 * j
        out.extend(range(s, s + 6))
    return out

parts = {
    "all_rot": list(range(7, 151)),
    "pelvis": rot_idx([0]),
    "lower": rot_idx([1,2,4,5,7,8,10,11]),
    "torso": rot_idx([3,6,9]),
    "upper": rot_idx([12,13,14,15,16,17,18,19,20,21,22,23]),
}

print("path:", path)
print("phys MSE:", float(np.mean((pred - gt) ** 2)))
print("phys rot MSE:", float(np.mean((pred[:, 7:151] - gt[:, 7:151]) ** 2)))
print("phys rootXZ MSE:", float(np.mean((pred[:, [4, 6]] - gt[:, [4, 6]]) ** 2)))
print("root path pred:", root_path(pred))
print("root path gt:", root_path(gt))

print("\\n=== MSE by part ===")
for name, idx in parts.items():
    print(f"{name:8s}: {float(np.mean((pred[:, idx] - gt[:, idx]) ** 2)):.8f}")

print("\\n=== frame-to-frame jump, pred vs gt ===")
pred_jump = np.linalg.norm(np.diff(pred[:, 7:151], axis=0), axis=1)
gt_jump = np.linalg.norm(np.diff(gt[:, 7:151], axis=0), axis=1)
ratio = pred_jump / (gt_jump + 1e-8)

top = np.argsort(-ratio)[:15]
for i in top:
    print(
        f"{i:02d}->{i+1:02d} "
        f"pred_jump={pred_jump[i]:.6f} "
        f"gt_jump={gt_jump[i]:.6f} "
        f"ratio={ratio[i]:.2f} "
        f"end_is_key={(i+1) in KEYS}"
    )

print("\\n=== keyframes ===")
for f in sorted(KEYS):
    all_mse = float(np.mean((pred[f] - gt[f]) ** 2))
    rot_mse = float(np.mean((pred[f, 7:151] - gt[f, 7:151]) ** 2))
    root_mse = float(np.mean((pred[f, [4,6]] - gt[f, [4,6]]) ** 2))
    print(f"frame {f:02d}: all={all_mse:.8f}, rot={rot_mse:.8f}, rootxz={root_mse:.8f}")
PY

python scripts/eval_v15_bridge_dense4.py \
  2>&1 | tee "logs/eval_e${EPOCHS}_v15_bridge_dense4_${RUN_NAME}.log"

python render_from_npy.py \
  --motion "${OUT_BASE}.npy" \
  --audio test_music_bank/dunhuangwu2.wav \
  --output "${OUT_BASE}_fixed.mp4" \
  --camera_mode fixed \
  2>&1 | tee "logs/render_e${EPOCHS}_v15_bridge_dense4_${RUN_NAME}_fixed.log"

echo "============================================================"
echo "DONE: $(date)"
echo "Checkpoint: $CKPT"
echo "Motion: ${OUT_BASE}.npy"
echo "Video: ${OUT_BASE}_fixed.mp4"
echo "Eval: logs/eval_e${EPOCHS}_v15_bridge_dense4_${RUN_NAME}.log"
echo "============================================================"
