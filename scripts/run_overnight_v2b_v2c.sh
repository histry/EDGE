#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

RUN_TAG="overnight_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TAG}"
mkdir -p "$LOG_DIR"

echo "===== EDGE overnight run: $RUN_TAG =====" | tee "$LOG_DIR/README.log"
date | tee -a "$LOG_DIR/README.log"

# ============================================================
# 0. Global safe env
# ============================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# v2b endpoint-continuous evaluation must stay clean.
export EDGE_TRAJ_EVENT_COND=0
export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_DYNAMIC_TRAJ_CFG=0
export EDGE_TRAJ_PHYSICS_FEATURES=0
export EDGE_TRAJ_FOURIER_FEATURES=0
export EDGE_TRAJ_SPARSE_WAYPOINT=0
export EDGE_TRAJ_BEV_COND=0

export EDGE_DISABLE_GAIT_TRAJECTORY_WRAPPER=1
export EDGE_UNIT_SOFT_PRIOR=0
export EDGE_ENABLE_TEXT_CONTEXT_RAG=0
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=0
export EDGE_BEAT_GUIDANCE=0
export EDGE_ENERGY_COND=0

# ============================================================
# 1. Make sure eval unit list exists
# ============================================================

mkdir -p output/whitelist_candidate_units_v2

cat > output/whitelist_candidate_units_v2/eval_units_v2b_small.txt <<'EOF'
55370
49122
61437
70219
69373
76207
EOF

# ============================================================
# 2. Full checkpoint sweep: 200/400/600/800/1000/1200
# ============================================================

echo "===== Stage 1: v2b endpoint-only checkpoint sweep =====" | tee -a "$LOG_DIR/README.log"

for step in 200 400 600 800 1000 1200; do
  CKPT="runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-${step}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "skip missing $CKPT" | tee -a "$LOG_DIR/README.log"
    continue
  fi

  echo "----- eval checkpoint train-${step}.pt -----" | tee -a "$LOG_DIR/README.log"

  python scripts/eval_whitelist_v2b_endpoint.py \
    --ckpt "$CKPT" \
    --unit_list output/whitelist_candidate_units_v2/eval_units_v2b_small.txt \
    --unit_root output/whitelist_candidate_units_v2 \
    --unit_root data/dunhuang_bvh/stationary_whitelist_v2_27units \
    --unit_root data/dunhuang_choreo_unit_rag \
    --out_dir "output/stationary_whitelist_v2b_endpoint_eval/e${step}_small" \
    --sampler ddim \
    2>&1 | tee "$LOG_DIR/eval_v2b_e${step}_small.log"
done

# ============================================================
# 3. Summarize metrics and select best checkpoint / units
# ============================================================

echo "===== Stage 2: summarize v2b endpoint eval =====" | tee -a "$LOG_DIR/README.log"

python - <<'PY' | tee "$LOG_DIR/v2b_endpoint_summary.csv"
import json
import math
from pathlib import Path
from collections import defaultdict

root = Path("output/stationary_whitelist_v2b_endpoint_eval")

rows = []
for d in sorted(root.glob("e*_small")):
    step = int(d.name.split("_")[0].replace("e", ""))
    for p in sorted(d.glob("unit_*_eval.json")):
        j = json.loads(p.read_text())
        uid = p.stem.replace("unit_", "").replace("_eval", "")

        # Conservative score:
        # - phys/lower/upper lower is better;
        # - jump close to 1 is better, too low means under-motion/frozen;
        # - jerk around <=1 is good, high jerk bad.
        phys = float(j.get("phys_mse", 999))
        upper = float(j.get("upper_mse", 999))
        lower = float(j.get("lower_mse", 999))
        jump = float(j.get("jump_ratio_p95", 999))
        jerk = float(j.get("jerk_ratio_p95", 999))
        rootxz = float(j.get("rootxz_mse", 999))

        score = (
            1.00 * phys
            + 0.35 * upper
            + 0.45 * lower
            + 0.30 * abs(jump - 1.0)
            + 0.35 * max(0.0, jerk - 1.0)
            + 0.10 * rootxz
        )

        rows.append({
            "step": step,
            "unit": uid,
            "phys_mse": phys,
            "rootxz_mse": rootxz,
            "jump_p95": jump,
            "jerk_p95": jerk,
            "upper_mse": upper,
            "lower_mse": lower,
            "score": score,
        })

print("step,unit,phys_mse,rootxz_mse,jump_p95,jerk_p95,upper_mse,lower_mse,score")
for r in sorted(rows, key=lambda x: (x["step"], x["unit"])):
    print(",".join(str(r[k]) for k in ["step","unit","phys_mse","rootxz_mse","jump_p95","jerk_p95","upper_mse","lower_mse","score"]))

by = defaultdict(list)
for r in rows:
    by[r["step"]].append(r)

summary = []
for step, arr in sorted(by.items()):
    mean = {}
    for k in ["phys_mse","rootxz_mse","jump_p95","jerk_p95","upper_mse","lower_mse","score"]:
        vals = [x[k] for x in arr if math.isfinite(x[k])]
        mean[k] = sum(vals) / max(len(vals), 1)
    summary.append((step, len(arr), mean))

Path("output/stationary_whitelist_v2b_endpoint_eval").mkdir(parents=True, exist_ok=True)

with open("output/stationary_whitelist_v2b_endpoint_eval/v2b_step_summary.txt", "w") as f:
    f.write("step,n,phys_mse,rootxz_mse,jump_p95,jerk_p95,upper_mse,lower_mse,score\n")
    for step, n, m in summary:
        f.write(f"{step},{n},{m['phys_mse']},{m['rootxz_mse']},{m['jump_p95']},{m['jerk_p95']},{m['upper_mse']},{m['lower_mse']},{m['score']}\n")

best_step = min(summary, key=lambda x: x[2]["score"])[0] if summary else None
best_units = sorted(rows, key=lambda x: x["score"])[:8]

with open("output/stationary_whitelist_v2b_endpoint_eval/best_step.txt", "w") as f:
    f.write(str(best_step) + "\n")

with open("output/stationary_whitelist_v2b_endpoint_eval/best_units_by_score.txt", "w") as f:
    for r in best_units:
        f.write(f"{r['unit']}\n")

with open("output/stationary_whitelist_v2b_endpoint_eval/best_units_by_score_verbose.csv", "w") as f:
    f.write("step,unit,phys_mse,rootxz_mse,jump_p95,jerk_p95,upper_mse,lower_mse,score\n")
    for r in best_units:
        f.write(",".join(str(r[k]) for k in ["step","unit","phys_mse","rootxz_mse","jump_p95","jerk_p95","upper_mse","lower_mse","score"]) + "\n")

print("")
print("# BEST_STEP", best_step)
print("# BEST_UNITS", ",".join([r["unit"] for r in best_units]))
PY

BEST_STEP="$(cat output/stationary_whitelist_v2b_endpoint_eval/best_step.txt | tr -d '[:space:]')"
echo "BEST_STEP=$BEST_STEP" | tee -a "$LOG_DIR/README.log"

if [ -z "$BEST_STEP" ] || [ "$BEST_STEP" = "None" ]; then
  echo "No best step found. Stop before v2c." | tee -a "$LOG_DIR/README.log"
  exit 1
fi

BEST_CKPT="runs/train_nextgen/stationary_whitelist_v2b_endpoint_continuous_from_v16_27units_b2_fix/weights/train-${BEST_STEP}.pt"
echo "BEST_CKPT=$BEST_CKPT" | tee -a "$LOG_DIR/README.log"

# ============================================================
# 4. Optional render selected endpoint-only outputs for morning QC
# ============================================================

echo "===== Stage 3: render selected v2b outputs for QC =====" | tee -a "$LOG_DIR/README.log"

mkdir -p "output/stationary_whitelist_v2b_endpoint_eval/render_best_e${BEST_STEP}"

for uid in $(cat output/whitelist_candidate_units_v2/eval_units_v2b_small.txt); do
  MOTION="output/stationary_whitelist_v2b_endpoint_eval/e${BEST_STEP}_small/unit_${uid}.npy"
  if [ -f "$MOTION" ]; then
    python render_from_npy.py \
      --motion "$MOTION" \
      --audio test_music_bank/dunhuangwu2.wav \
      --output "output/stationary_whitelist_v2b_endpoint_eval/render_best_e${BEST_STEP}/unit_${uid}_fixed.mp4" \
      --camera_mode fixed \
      2>&1 | tee "$LOG_DIR/render_v2b_e${BEST_STEP}_unit_${uid}.log" || true
  fi
done

# ============================================================
# 5. Prepare best-8 unit list for tomorrow / next training
# ============================================================

cp output/stationary_whitelist_v2b_endpoint_eval/best_units_by_score.txt \
   output/whitelist_candidate_units_v2/best8_from_v2b_endpoint_eval.txt

echo "best8 list:"
cat output/whitelist_candidate_units_v2/best8_from_v2b_endpoint_eval.txt | tee -a "$LOG_DIR/README.log"

# ============================================================
# 6. Smoke-test new trajectory-event patch
# ============================================================

echo "===== Stage 4: smoke test native trajectory-event patch =====" | tee -a "$LOG_DIR/README.log"

export EDGE_TRAJ_PHYSICS_FEATURES=1
export EDGE_TRAJ_FOURIER_FEATURES=0
export EDGE_TRAJ_SPARSE_WAYPOINT=1
export EDGE_TRAJ_EVENT_COND=1
export EDGE_TRAJ_EVENT_AUTODETECT=1
export EDGE_TRAJ_EVENT_INIT_GATE=0.0
export EDGE_TRAJ_EVENT_ZERO_LAST=1

export EDGE_TURN_EVENT_COUNT=3
export EDGE_TURN_SUPPORT_LAG=8
export EDGE_TURN_EXPR_LAG=4
export EDGE_TURN_MIN_GAP=18
export EDGE_TURN_GATE_SIGMA=5.0

export EDGE_WEAK_TRAJ_ENERGY=0
export EDGE_DYNAMIC_TRAJ_CFG=0

python - <<'PY' 2>&1 | tee "$LOG_DIR/smoke_patch_import.log"
import os
os.environ["EDGE_TRAJ_EVENT_COND"] = "1"
os.environ["EDGE_WEAK_TRAJ_ENERGY"] = "0"
import sitecustomize
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
print("event_patch", getattr(DanceDecoder, "_edge_trajectory_event_condition_patch_installed", False))
print("weak_energy_patch", getattr(GaussianDiffusion, "_edge_weak_traj_energy_patch_installed", False))
PY

# ============================================================
# 7. Conservative v2c native trajectory-event training
# ============================================================
# This is intentionally conservative:
# - start from best v2b checkpoint;
# - no weak energy guidance;
# - no dynamic CFG;
# - no mid keyframes;
# - low trajectory/root-lower losses.
#
# If your dataset path differs, edit V2C_DATA_PATH below.

echo "===== Stage 5: start v2c native trajectory-event training =====" | tee -a "$LOG_DIR/README.log"

V2C_DATA_PATH="data/dunhuang_bvh/stationary_whitelist_v2_27units"
if [ ! -d "$V2C_DATA_PATH" ]; then
  echo "WARNING: $V2C_DATA_PATH not found. Falling back to data/dunhuang_bvh/processed" | tee -a "$LOG_DIR/README.log"
  V2C_DATA_PATH="data/dunhuang_bvh/processed"
fi

python train.py \
  --project runs/train_nextgen \
  --exp_name "v2c_native_traj_event_from_v2b_e${BEST_STEP}_overnight" \
  --data_path "$V2C_DATA_PATH" \
  --processed_data_dir data/dataset_backups \
  --feature_type hybrid \
  --audio_dim 803 \
  --seq_len 45 \
  --checkpoint "$BEST_CKPT" \
  --batch_size 2 \
  --epochs 600 \
  --save_interval 50 \
  --val_batches 4 \
  --learning_rate 1e-5 \
  --weight_decay 0.02 \
  --cond_drop_prob 0.10 \
  --audio_pairing_mode proxy \
  --mmr_loss_weight 0.0 \
  --keyframe_condition_prob 0.7 \
  --keyframe_condition_width 1 \
  --keyframe_loss_weight 0.15 \
  --mid_keyframe_condition_prob 0.0 \
  --mid_keyframe_count 0 \
  --mid_keyframe_condition_width 0 \
  --trajectory_loss_weight 0.05 \
  --trajectory_velocity_loss_weight 0.02 \
  --root_lower_coupling_loss_weight 0.05 \
  --root_lower_min_motion 0.006 \
  --contact_loss_weight 0.15 \
  --foot_loss_weight 0.20 \
  --sync_loss_weight 0.10 \
  --energy_condition_prob 0.0 \
  --energy_condition_drop_prob 0.0 \
  --energy_loss_weight 0.0 \
  --disable_traj_cond \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --train_num_workers 0 \
  --val_num_workers 0 \
  2>&1 | tee "$LOG_DIR/train_v2c_native_traj_event.log"

echo "===== overnight finished =====" | tee -a "$LOG_DIR/README.log"
date | tee -a "$LOG_DIR/README.log"
