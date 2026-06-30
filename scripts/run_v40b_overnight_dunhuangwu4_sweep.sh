#!/usr/bin/env bash
set -u

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# ===== 必需 checkpoint =====
export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_V23_CKPT="output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V27_TRANSITION_DIFFUSION_CKPT="output/v33_event_contact_20260611_204533/v32_contact_inr_training/checkpoints/best.pt"

# ===== V40B 源头索引 =====
export V40B_SOURCE_INDEX_JSON="data/v34_source_aware/v34_shared_event_index_source_aware.json"
export V40B_SOURCE_INDEX_NPZ="data/v34_source_aware/v34_shared_event_index_source_aware.npz"

# ===== 只跑 dunhuangwu4 =====
export V40B_MUSIC="test_music_bank/dunhuangwu4.wav"
export V40B_KEYS="dunhuangwu4"

# ===== 禁止 V41 污染 =====
unset V41_BEAT_SUPPORT_STABILIZER || true
unset V41_BEAT_DECOUPLED_SUPPORT_STABILIZER || true
unset V41_ROOT_XZ_FILTER_STRENGTH || true
unset V41_RELOCK_STRENGTH || true
unset V41_SUPPORT_LOWER_STRENGTH || true

# ===== V34 / V40B 规划参数 =====
export V26_MUSIC="test_music_bank/dunhuangwu4.wav"
export V32_KEYS="dunhuangwu4"
export V26_MIN_TIME_WARP=0.82
export V26_MAX_TIME_WARP=1.30
export V26_BEAM_SIZE=64
export V26_CANDIDATE_TOP_K=1024
export V34_WARP_PREFILTER_TOP_K=1024

# 保持 transition diffusion
export V27_TRANSITION_DIFFUSION=1
export V32_INR_TRUST=0.25
export V32_INFERENCE_STEPS=40

# 保持 V40 后处理，不上 V41
export V34_MOTION_QUALITY_POSTPROCESS=1
export V34_ROOT_Y_PHYSICS=1
export V34_CONTACT_LOCK=1
export V34_FLOOR_CLEARANCE=1
export V38_BUTTERWORTH_FILTER=1
export V40_FLOOR_AWARE_LEG_IK=1
export V40_ANKLE_PITCH_CLAMP=1

# V40 后处理参数：不要再极端加 IK，避免 root_y_delta 继续膨胀
export V34_FLOOR_MAX_LIFT=0.36
export V40_FLOOR_IK_STEPS=55
export V40_FLOOR_IK_LR=0.010
export V40_FLOOR_IK_FLOOR_WEIGHT=42.0
export V40_FLOOR_IK_ROOT_WEIGHT=0.025
export V40_FLOOR_IK_ROT_WEIGHT=0.32
export V40_FLOOR_IK_TEMPORAL_WEIGHT=0.09
export V40_FLOOR_IK_ANATOMY_WEIGHT=28.0
export V40_FLOOR_IK_MAX_ROOT_DELTA=0.22
export V40_FLOOR_IK_CONTEXT_FRAMES=5
export V40_FLOOR_IK_LOWER_JOINTS="lhip,rhip,lknee,rknee,lankle,rankle,ltoes,rtoes"

export V40_ANKLE_CLAMP_MAX_DEG=38.0
export V40_ANKLE_CLAMP_CONTACT_MAX=0.60
export V40_ANKLE_CLAMP_TRIGGER_M=0.008

export V34_CONTACT_LOCK_STRENGTH=0.70
export V39_ROOT_CORR_MAX_STEP=0.009
export V39_ROOT_CORR_MAX_ACCEL=0.003
export V39_SUPPORT_ROOT_VELOCITY_DAMPING=0.04

MASTER_ROOT="output/v40b_overnight_dunhuangwu4_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_ROOT"
echo "$MASTER_ROOT" > output/LATEST_V40B_OVERNIGHT_D4_SWEEP.txt

echo "[V40B OVERNIGHT START] $(date)"
echo "[MASTER_ROOT] $MASTER_ROOT"
echo "[ROUTER] $V26_ROUTER_CKPT"
echo "[V23] $V26_V23_CKPT"
echo "[PLANNER] $V26_PLANNER_CKPT"
echo "[TRANSITION] $V27_TRANSITION_DIFFUSION_CKPT"

# 阈值由保守到激进
for THR in 0.050 0.045 0.040 0.035; do
  TAG="thr_${THR/./p}"
  RUN_ROOT="$MASTER_ROOT/$TAG"
  mkdir -p "$RUN_ROOT"

  echo
  echo "============================================================"
  echo "[V40B RUN] threshold=$THR  run_root=$RUN_ROOT  $(date)"
  echo "============================================================"

  export RUN_ROOT="$RUN_ROOT"
  export RUN_ID="v40b_d4_${TAG}"
  export V40B_NATIVE_FLOOR_REMOVE_THRESHOLD="$THR"

  # soft threshold 比 remove 阈值略低
  python - <<PY > "$RUN_ROOT/threshold.env"
thr=float("$THR")
soft=max(0.020, thr-0.010)
print(f"export V40B_NATIVE_FLOOR_SOFT_THRESHOLD={soft:.3f}")
PY
  source "$RUN_ROOT/threshold.env"

  # 激进阈值允许稍大删除比例
  if python - <<PY
thr=float("$THR")
raise SystemExit(0 if thr <= 0.040 else 1)
PY
  then
    export V40B_MAX_REMOVE_FRACTION=0.45
  else
    export V40B_MAX_REMOVE_FRACTION=0.30
  fi

  echo "[THRESHOLD] remove=$V40B_NATIVE_FLOOR_REMOVE_THRESHOLD soft=$V40B_NATIVE_FLOOR_SOFT_THRESHOLD max_remove_fraction=$V40B_MAX_REMOVE_FRACTION"

  bash scripts/run_v40b_native_floor_reroute.sh > "$RUN_ROOT/run.log" 2>&1
  CODE=$?

  echo "[RUN EXIT] threshold=$THR code=$CODE"

  if [[ -f "$RUN_ROOT/v40b_pruned_index/v40b_native_floor_prune_audit.json" ]]; then
    python - <<PY
import json, os
f="$RUN_ROOT/v40b_pruned_index/v40b_native_floor_prune_audit.json"
a=json.load(open(f))
s=a.get("summary", a)
print("[PRUNE SUMMARY]", json.dumps({
    "num_before": s.get("num_before"),
    "num_after": s.get("num_after"),
    "removed": s.get("removed"),
    "remove_fraction": s.get("remove_fraction"),
    "max_pen": s.get("native_floor_penetration_max_m"),
    "p95_pen": s.get("native_floor_penetration_p95_m"),
    "num_over_remove": s.get("num_over_remove_threshold"),
}, ensure_ascii=False))
PY
  fi

  python - <<PY
import json, glob, os
root="$RUN_ROOT/v40b_native_floor_reroute"
files=glob.glob(root+"/*motion_quality_postprocess*.json")
print("[RESULT JSON COUNT]", len(files))
for f in sorted(files):
    try:
        d=json.load(open(f))
    except Exception as e:
        print("[BAD JSON]", f, e)
        continue
    post=d.get("post_audit",{})
    pf=d.get("planner_feedback",{})
    print("[RESULT]", os.path.basename(f),
          "accepted=", pf.get("accepted"),
          "reject=", pf.get("reject_reasons"),
          "foot_pen=", post.get("foot_penetration_min_m"),
          "skate_p95=", post.get("foot_skate_p95_mpf"),
          "jerk_p95=", post.get("mean_joint_jerk_p95"),
          "local_ik_after=", d.get("local_floor_ik",{}).get("after",{}).get("max_penetration"),
          "has_v41=", "beat_decoupled_support_stabilizer" in d)
PY

  # 如果已经 accepted，就继续跑后面的阈值也可以；这里不中断，方便整晚比较最优。
done

echo
echo "[V40B OVERNIGHT DONE] $(date)"
echo "[MASTER_ROOT] $MASTER_ROOT"
