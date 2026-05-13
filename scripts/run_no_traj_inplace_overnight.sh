#!/usr/bin/env bash
set -uo pipefail

cd /home/disk/lsm/storage/EDGE

mkdir -p logs
mkdir -p output/no_traj_inplace_overnight
mkdir -p output/no_traj_inplace_overnight/videos

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "❌ Please run: conda activate edge"
  exit 1
fi

PY="$CONDA_PREFIX/bin/python"
GEN="generate_controlled_v9.py"

OUTROOT="output/no_traj_inplace_overnight"
SUMMARY_CSV="$OUTROOT/summary_metrics.csv"
REPORT_MD="$OUTROOT/README_RESULTS.md"

CKPT_V10="${CKPT_V10:-runs/train_stage45/v10_text_context_rag_adapter_e8_fix2_full/weights/train-4.pt}"
CKPT_V12="${CKPT_V12:-runs/train_stage45/v12_no_leakage_xz_source_split/weights/train-50.pt}"
RAG_DB="${RAG_DB:-data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz}"

START_POSE="${START_POSE:-test_keyframes/demo_dyl002_start.npy}"
END_POSE_LOOP="${END_POSE_LOOP:-test_keyframes/demo_dyl002_start.npy}"
END_POSE_REAL="${END_POSE_REAL:-test_keyframes/demo_dyl002_end.npy}"

# 晚上默认跑三首音乐。若只想先跑一首，可运行前设置：
# MUSIC_NAMES="dunhuangwu2" bash scripts/run_no_traj_inplace_overnight.sh
MUSIC_NAMES="${MUSIC_NAMES:-dunhuangwu2 dunhuangwu3 dunhuangwu4}"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ------------------------------------------------------------
# 0. 全局清理 trajectory / turn-event / gait 污染
# ------------------------------------------------------------

clear_traj_env () {
  while IFS='=' read -r key val; do
    case "$key" in
      EDGE_TURN*) unset "$key" ;;
    esac
  done < <(env)

  export EDGE_DYNAMIC_TRAJ_CFG=0
  export EDGE_GAIT_PHASE_COND=0
  export EDGE_GAIT_CONTACT_LOSS=0
  export EDGE_TRAJ_PHYSICS_FEATURES=0
  export EDGE_TRAJ_FOURIER_FEATURES=0
  export EDGE_TRAJ_SPARSE_WAYPOINT=0
  export EDGE_TRAJECTORY_REP=

  unset EDGE_TURN_EVENT_MODEL_ADAPTER
  unset EDGE_TURN_EVENT_TRAJ_TOKEN
  unset EDGE_TURN_EVENT_OUTPUT_ADAPTER
  unset EDGE_TURN_EVENT_FREEZE_BACKBONE
  unset EDGE_TURN_EVENT_PRESERVE_ROOT_XZ
}

clear_traj_env

export EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
export EDGE_AUDIO_DEVICE=cpu
export EDGE_EXPERIMENT_PROFILE=v10

# 不走 V10 wrapper，不传 trajectory，不传 target_traj
# 但允许 checkpoint 内已有 Text/Pose RAG 与 RAG Summary 分支
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1

echo "============================================================"
echo "No-trajectory in-place Dunhuang overnight experiment"
echo "python       = $PY"
echo "ckpt_v10     = $CKPT_V10"
echo "ckpt_v12     = $CKPT_V12"
echo "start_pose   = $START_POSE"
echo "end_loop     = $END_POSE_LOOP"
echo "music_names  = $MUSIC_NAMES"
echo "outroot      = $OUTROOT"
echo "============================================================"

echo ""
echo "Remaining EDGE_TURN env after cleanup:"
env | grep '^EDGE_TURN' || echo "  none"

"$PY" - <<'PY'
import sys
print("python =", sys.executable)
import torch, numpy
print("torch =", torch.__version__)
print("numpy =", numpy.__version__)
PY

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summarize_all () {
  "$PY" - <<'PY'
import csv
import numpy as np
from pathlib import Path

outroot = Path("output/no_traj_inplace_overnight")
rows = []

def load_motion(path):
    x = np.load(path)
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)

for p in sorted(outroot.glob("*.npy")):
    name = p.stem
    if (
        name.endswith("_raw")
        or name.endswith("_target_traj")
        or "_mid" in name
        or name.endswith("_unit")
    ):
        continue

    try:
        x = load_motion(p)
        if x.ndim != 2 or x.shape[-1] < 20:
            continue

        pose = x[:, 7:] if x.shape[-1] >= 151 else x
        dpose = np.diff(pose, axis=0)
        frame_energy = np.linalg.norm(dpose, axis=1)

        motion_energy = float(np.mean(frame_energy))
        motion_p95 = float(np.percentile(frame_energy, 95))
        freezing_rate = float(np.mean(frame_energy < 1e-3))

        if x.shape[-1] >= 151:
            root_xz = x[:, [4, 6]]
            root_path = float(np.sum(np.linalg.norm(np.diff(root_xz, axis=0), axis=1)))

            # proxy 分组，仅用于 overnight 快速筛选，不等于严格 anatomical group
            lower = x[:, 7:79]
            upper = x[:, 79:151]
            lower_activity = float(np.mean(np.linalg.norm(np.diff(lower, axis=0), axis=1)))
            upper_activity = float(np.mean(np.linalg.norm(np.diff(upper, axis=0), axis=1)))
        else:
            root_path = float("nan")
            lower_activity = float("nan")
            upper_activity = float("nan")

        jerk = float(np.mean(np.linalg.norm(np.diff(x, n=3, axis=0), axis=1))) if len(x) > 4 else float("nan")

        rows.append({
            "case": name,
            "motion_energy": motion_energy,
            "motion_p95": motion_p95,
            "upper_activity_proxy": upper_activity,
            "lower_activity_proxy": lower_activity,
            "root_path_proxy": root_path,
            "transition_jerk_proxy": jerk,
            "freezing_rate_proxy": freezing_rate,
            "path": str(p),
        })
    except Exception as e:
        rows.append({
            "case": name,
            "error": repr(e),
            "path": str(p),
        })

rows = sorted(rows, key=lambda r: r.get("motion_energy", -1), reverse=True)

csv_path = outroot / "summary_metrics.csv"
with csv_path.open("w", newline="") as f:
    fieldnames = [
        "case",
        "motion_energy",
        "motion_p95",
        "upper_activity_proxy",
        "lower_activity_proxy",
        "root_path_proxy",
        "transition_jerk_proxy",
        "freezing_rate_proxy",
        "path",
        "error",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

md_path = outroot / "README_RESULTS.md"
with md_path.open("w") as f:
    f.write("# No-trajectory in-place Dunhuang overnight results\n\n")
    f.write("本实验不传 `--trajectory`，不使用 `generate_v10_choreo.py`，不做 static trajectory。排序指标是快速 proxy，最终以视频为准。\n\n")
    f.write("| rank | case | motion_energy | upper_proxy | lower_proxy | root_path | jerk_proxy | freezing | follow_video |\n")
    f.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
    for i, r in enumerate(rows, 1):
        if "error" in r:
            continue
        case = r["case"]
        video = f"videos/{case}_follow.mp4"
        f.write(
            f"| {i} | {case} | "
            f"{r.get('motion_energy', float('nan')):.6f} | "
            f"{r.get('upper_activity_proxy', float('nan')):.6f} | "
            f"{r.get('lower_activity_proxy', float('nan')):.6f} | "
            f"{r.get('root_path_proxy', float('nan')):.6f} | "
            f"{r.get('transition_jerk_proxy', float('nan')):.6f} | "
            f"{r.get('freezing_rate_proxy', float('nan')):.6f} | "
            f"{video} |\n"
        )

print(f"✅ summary saved: {csv_path}")
print(f"✅ report saved: {md_path}")
print("\n==== TOP 15 by motion_energy ====")
for r in rows[:15]:
    if "error" in r:
        print(r["case"], r["error"])
    else:
        print(
            f"{r['case']}: "
            f"energy={r['motion_energy']:.6f}, "
            f"upper={r['upper_activity_proxy']:.6f}, "
            f"lower={r['lower_activity_proxy']:.6f}, "
            f"root={r['root_path_proxy']:.6f}, "
            f"jerk={r['transition_jerk_proxy']:.6f}, "
            f"freeze={r['freezing_rate_proxy']:.6f}"
        )
PY
}

# ------------------------------------------------------------
# Rendering
# ------------------------------------------------------------

render_all () {
  echo ""
  echo "============================================================"
  echo "Rendering motions"
  echo "============================================================"

  for MOTION in "$OUTROOT"/*.npy; do
    [ -f "$MOTION" ] || continue
    BASE=$(basename "$MOTION" .npy)

    case "$BASE" in
      *_raw|*_target_traj|*_mid*|*_unit*) continue ;;
    esac

    MUSIC_NAME="${BASE%%_*}"
    AUDIO="test_music_bank/${MUSIC_NAME}.wav"

    if [ ! -f "$AUDIO" ]; then
      AUDIO="test_music_bank/dunhuangwu2.wav"
    fi

    FOLLOW="$OUTROOT/videos/${BASE}_follow.mp4"
    FIXED="$OUTROOT/videos/${BASE}_fixed.mp4"

    if [ ! -f "$FOLLOW" ]; then
      echo "Rendering follow: $BASE"
      "$PY" render_from_npy.py \
        --motion "$MOTION" \
        --audio "$AUDIO" \
        --output "$FOLLOW" \
        --camera_mode follow \
        2>&1 | tee "logs/render_${BASE}_follow.log" || echo "⚠️ render follow failed: $BASE"
    fi

    if [ ! -f "$FIXED" ]; then
      echo "Rendering fixed: $BASE"
      "$PY" render_from_npy.py \
        --motion "$MOTION" \
        --audio "$AUDIO" \
        --output "$FIXED" \
        --camera_mode fixed \
        2>&1 | tee "logs/render_${BASE}_fixed.log" || echo "⚠️ render fixed failed: $BASE"
    fi
  done
}

# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------

run_gen () {
  MUSIC_NAME="$1"
  PHASE="$2"
  NAME="$3"
  CKPT="$4"
  ENDPOSE="$5"
  ENDPOINT="$6"
  BEAT="$7"
  ENERGY_SCALE="$8"
  CONTEXT_SCALE="$9"
  PRIOR="${10}"
  PRIOR_FEATURES="${11}"
  AUTO_MID_COUNT="${12}"
  MID_STRENGTH="${13}"

  MUSIC="test_music_bank/${MUSIC_NAME}.wav"
  OUT="$OUTROOT/${MUSIC_NAME}_${PHASE}_${NAME}.npy"

  if [ ! -f "$MUSIC" ]; then
    echo "⚠️ music not found, skip: $MUSIC"
    return 0
  fi

  if [ -f "$OUT" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "⏭️ skip existing: $OUT"
    return 0
  fi

  clear_traj_env

  export EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
  export EDGE_AUDIO_DEVICE=cpu
  export EDGE_EXPERIMENT_PROFILE=v10
  export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
  export EDGE_ENABLE_RAG_SUMMARY_TOKEN=1

  export EDGE_BEAT_GUIDANCE=0
  export EDGE_BEAT_GUIDANCE_WEIGHT=0

  if [ "$BEAT" != "0" ]; then
    export EDGE_BEAT_GUIDANCE=1
    export EDGE_BEAT_GUIDANCE_WEIGHT="$BEAT"
    export EDGE_BEAT_GUIDANCE_TARGET=1.35
    export EDGE_BEAT_GUIDANCE_FEATURES=all
  fi

  export EDGE_AUDIO_ENERGY_AS_COND=1
  export EDGE_MUSIC_TENSION_AS_ENERGY=1
  export EDGE_ENERGY_CFG_SCALE="$ENERGY_SCALE"

  export EDGE_CONTEXT_RAG_ENHANCE=1
  export EDGE_CONTEXT_RAG_SCALE="$CONTEXT_SCALE"

  export EDGE_UNIT_SOFT_PRIOR=1
  export EDGE_UNIT_PRIOR_REQUIRED=0
  export EDGE_UNIT_PRIOR_TEMPORAL=1
  export EDGE_UNIT_PRIOR_DCT=1
  export EDGE_UNIT_PRIOR_DCT_DECAY=soft_exp
  export EDGE_UNIT_PRIOR_DCT_DECAY_STRENGTH=3.0
  export EDGE_UNIT_PRIOR_LOW_FREQ_K=4
  export EDGE_UNIT_PRIOR_FEATURES="$PRIOR_FEATURES"
  export EDGE_UNIT_PRIOR_STRENGTH="$PRIOR"

  # TextBridge 偏原地敦煌表达
  export EDGE_TEXT_BRIDGE_MODE=force_topk
  export EDGE_TEXT_BRIDGE_TOPK=256
  export EDGE_TEXT_BRIDGE_TOP_K=256
  export EDGE_TEXT_BRIDGE_WEIGHT=1.0
  export EDGE_TEXT_QUERY="敦煌飞天，原地舞蹈，上肢舒展，躯干延展，手臂大幅开合，音乐重音响应，亮相收束，强表现力"

  echo ""
  echo "============================================================"
  echo "Generate:"
  echo "  music         = $MUSIC_NAME"
  echo "  phase         = $PHASE"
  echo "  name          = $NAME"
  echo "  ckpt          = $CKPT"
  echo "  endpoint      = $ENDPOINT"
  echo "  beat          = $BEAT"
  echo "  energy_scale  = $ENERGY_SCALE"
  echo "  context_scale = $CONTEXT_SCALE"
  echo "  prior         = $PRIOR"
  echo "  prior_features= $PRIOR_FEATURES"
  echo "  auto_mid_count= $AUTO_MID_COUNT"
  echo "  mid_strength  = $MID_STRENGTH"
  echo "  out           = $OUT"
  echo "============================================================"

  ARGS=(
    "$GEN"
    --checkpoint "$CKPT"
    --music "$MUSIC"
    --start_pose "$START_POSE"
    --end_pose "$ENDPOSE"
    --out "$OUT"
    --feature_type hybrid
    --sampler ddim
    --endpoint_keyframe_strength "$ENDPOINT"
    --no_tto
  )

  # 仍然不传 --trajectory
  # auto_mid 只作为 no-trajectory 动作内容提示，不走 V10 wrapper
  if [ "$AUTO_MID_COUNT" != "0" ]; then
    ARGS+=(
      --auto_mid_keyframes
      --rag_db "$RAG_DB"
      --auto_mid_count "$AUTO_MID_COUNT"
      --auto_mid_min_gap 30
      --auto_mid_source_gap 10
      --auto_mid_disallow_same_source
      --auto_mid_max_candidates 256
      --auto_mid_energy_weight 0.8
      --auto_mid_energy_target 0.8
      --auto_mid_energy_band 0.4
      --auto_mid_pose_weight 0.8
      --auto_mid_diversity_weight 0.4
      --auto_mid_contact_weight 0.2
      --mid_keyframe_strength "$MID_STRENGTH"
    )
  fi

  "$PY" "${ARGS[@]}" 2>&1 | tee "logs/${MUSIC_NAME}_${PHASE}_${NAME}.log"
  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" != "0" ]; then
    echo "⚠️ generation failed: ${MUSIC_NAME}_${PHASE}_${NAME}"
    return 0
  fi

  echo "✅ generated: $OUT"
}

# ------------------------------------------------------------
# Phase 1: start-loop baseline
# ------------------------------------------------------------

for M in $MUSIC_NAMES; do
  run_gen "$M" "p1" "v10_startloop_ep100_base" "$CKPT_V10" "$END_POSE_LOOP" 1.0 0 0.0 0.0 0.0 "upper+torso" 0 0.0
  run_gen "$M" "p1" "v10_startloop_ep050_base" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0 0.0 0.0 0.0 "upper+torso" 0 0.0
  run_gen "$M" "p1" "v10_startloop_ep020_base" "$CKPT_V10" "$END_POSE_LOOP" 0.2 0 0.0 0.0 0.0 "upper+torso" 0 0.0
done

# v12 干净基线，只跑 dunhuangwu2，判断是否 v10 分支导致保守
if [ -f "$CKPT_V12" ]; then
  run_gen "dunhuangwu2" "p1" "v12_startloop_ep050_base" "$CKPT_V12" "$END_POSE_LOOP" 0.5 0 0.0 0.0 0.0 "upper+torso" 0 0.0
fi

# ------------------------------------------------------------
# Phase 2: content boost without auto-mid
# ------------------------------------------------------------

for M in $MUSIC_NAMES; do
  run_gen "$M" "p2" "energy05_ctx05_noauto_ep050" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0 0.5 0.5 0.0 "upper+torso" 0 0.0
  run_gen "$M" "p2" "energy08_ctx05_noauto_ep020" "$CKPT_V10" "$END_POSE_LOOP" 0.2 0 0.8 0.5 0.0 "upper+torso" 0 0.0
done

# ------------------------------------------------------------
# Phase 3: no-trajectory RAG / unit-prior content boost
# ------------------------------------------------------------

for M in $MUSIC_NAMES; do
  run_gen "$M" "p3" "rag1_ep050_mid010_prior012_upper" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0 0.5 0.5 0.012 "upper" 1 0.10
  run_gen "$M" "p3" "rag1_ep050_mid010_prior020_uppertorso" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0 0.5 0.5 0.020 "upper+torso" 1 0.10
  run_gen "$M" "p3" "rag2_ep050_mid015_prior012_upper" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0 0.5 0.5 0.012 "upper" 2 0.15
  run_gen "$M" "p3" "rag2_ep020_mid015_prior012_upper" "$CKPT_V10" "$END_POSE_LOOP" 0.2 0 0.5 0.5 0.012 "upper" 2 0.15
done

# ------------------------------------------------------------
# Phase 4: beat guidance on content cases
# ------------------------------------------------------------

for M in $MUSIC_NAMES; do
  run_gen "$M" "p4" "rag1_ep050_prior012_beat001" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0.01 0.5 0.5 0.012 "upper" 1 0.10
  run_gen "$M" "p4" "rag1_ep050_prior012_beat003" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0.03 0.5 0.5 0.012 "upper" 1 0.10
  run_gen "$M" "p4" "rag2_ep050_prior012_beat001" "$CKPT_V10" "$END_POSE_LOOP" 0.5 0.01 0.5 0.5 0.012 "upper" 2 0.15
done

summarize_all
render_all
summarize_all

# ------------------------------------------------------------
# Phase 5: V11 Cross-Attention RAG expression training
# ------------------------------------------------------------

RUN_V11_TRAIN="${RUN_V11_TRAIN:-1}"
V11_EPOCHS="${V11_EPOCHS:-20}"

if [ "$RUN_V11_TRAIN" = "1" ]; then
  echo ""
  echo "============================================================"
  echo "Phase 5: V11 Cross-Attention RAG expression training"
  echo "============================================================"

  clear_traj_env

  unset EDGE_DIFF_CONTACT_LOSS
  unset EDGE_BEAT_GUIDANCE

  export EDGE_CHECKPOINT_COMPAT_CPU_MERGE=1
  export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
  export EDGE_V11_CROSS_ATTN_RAG=1
  export EDGE_V11_RAG_CROSS_ATTN_ZERO_INIT=1
  export EDGE_V11_RAG_CROSS_ATTN_WEIGHT=1.0
  export EDGE_TEXT_CONTEXT_TRAIN_SELF=1
  export EDGE_TEXT_CONTEXT_REQUIRE_GRAD=1
  export EDGE_TEXT_CONTEXT_MIN_GRAD_NORM=1e-10

  echo "---- V11 smoke ----"

  "$PY" train.py \
    --project runs/train_nextgen \
    --exp_name v11_xattn_inplace_expr_smoke_b1 \
    --train_stage adapter \
    --adapter_train_decoder \
    --checkpoint "$CKPT_V10" \
    --data_path data/dunhuang_bvh/processed \
    --feature_type hybrid \
    --audio_pairing_mode none \
    --batch_size 1 \
    --epochs 1 \
    --learning_rate 5e-6 \
    --enable_rag_summary_token \
    --save_interval 1 \
    --val_batches 0 \
    --max_train_batches 20 \
    --train_num_workers 0 \
    --val_num_workers 0 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    2>&1 | tee logs/v11_xattn_inplace_expr_smoke_b1.log

  SMOKE_STATUS=${PIPESTATUS[0]}

  if [ "$SMOKE_STATUS" = "0" ]; then
    echo "✅ V11 smoke passed. Start e${V11_EPOCHS} training."

    "$PY" train.py \
      --project runs/train_nextgen \
      --exp_name v11_xattn_inplace_expr_e${V11_EPOCHS}_b1 \
      --train_stage adapter \
      --adapter_train_decoder \
      --checkpoint "$CKPT_V10" \
      --data_path data/dunhuang_bvh/processed \
      --feature_type hybrid \
      --audio_pairing_mode none \
      --batch_size 1 \
      --epochs "$V11_EPOCHS" \
      --learning_rate 5e-6 \
      --enable_rag_summary_token \
      --save_interval 5 \
      --val_batches 5 \
      --train_num_workers 2 \
      --val_num_workers 1 \
      --mixed_precision bf16 \
      --gradient_checkpointing \
      2>&1 | tee "logs/v11_xattn_inplace_expr_e${V11_EPOCHS}_b1.log"

    TRAIN_STATUS=${PIPESTATUS[0]}

    CKPT_V11="runs/train_nextgen/v11_xattn_inplace_expr_e${V11_EPOCHS}_b1/weights/train-${V11_EPOCHS}.pt"

    if [ "$TRAIN_STATUS" = "0" ] && [ -f "$CKPT_V11" ]; then
      echo "✅ V11 checkpoint found: $CKPT_V11"

      for M in $MUSIC_NAMES; do
        run_gen "$M" "p5" "v11_startloop_ep050_base" "$CKPT_V11" "$END_POSE_LOOP" 0.5 0 0.0 0.0 0.0 "upper+torso" 0 0.0
        run_gen "$M" "p5" "v11_rag1_ep050_prior012_upper" "$CKPT_V11" "$END_POSE_LOOP" 0.5 0 0.5 0.5 0.012 "upper" 1 0.10
        run_gen "$M" "p5" "v11_rag1_ep050_prior012_beat001" "$CKPT_V11" "$END_POSE_LOOP" 0.5 0.01 0.5 0.5 0.012 "upper" 1 0.10
      done

      summarize_all
      render_all
      summarize_all
    else
      echo "⚠️ V11 training failed or final checkpoint not found."
      echo "Check partial checkpoints under runs/train_nextgen/v11_xattn_inplace_expr_e${V11_EPOCHS}_b1/weights/"
    fi
  else
    echo "⚠️ V11 smoke failed. Skip e${V11_EPOCHS} training."
  fi
fi

summarize_all

cat >> "$REPORT_MD" <<'MD'

## 明早检查顺序

1. 先看 `summary_metrics.csv` 中 `motion_energy` 和 `upper_activity_proxy` 排名前 10 的 case。
2. 再看对应 `videos/*_follow.mp4`。
3. 如果 Phase 1 能明显动，说明之前 static trajectory / V10 wrapper 可能压制动作。
4. 如果 Phase 3 能明显动，说明 RAG / auto-mid / Unit Prior 是有效的原地动作提示。
5. 如果 Phase 4 比 Phase 3 更贴音乐，保留 beat guidance，优先使用 `0.01`。
6. 如果 Phase 1–4 都只是抖，但 Phase 5 变好，说明必须训练 V11 Text/Pose RAG 表达分支。
7. 如果 Phase 5 仍弱，下一步应考虑 video-derived style RAG 或专门的原地 expressive training，不应急着恢复轨迹。

## 实验协议

本实验全程不传 `--trajectory`，不使用 `generate_v10_choreo.py`，不使用 static trajectory。
`auto_mid_keyframes` 只在 Phase 3/4/5 中作为 no-trajectory 动作内容提示，mid strength 很低，不作为轨迹控制。
MD

echo ""
echo "============================================================"
echo "✅ Overnight no-trajectory in-place experiment finished."
echo "Summary CSV:"
echo "  $SUMMARY_CSV"
echo "Report:"
echo "  $REPORT_MD"
echo "Videos:"
echo "  $OUTROOT/videos"
echo "Master log:"
echo "  logs/no_traj_inplace_overnight_master.log"
echo "============================================================"
