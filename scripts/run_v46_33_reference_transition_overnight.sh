#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${V46_RUN_ROOT:-output/v46_33_reference_transition_alltrain_${RUN_TS}}"
SPLIT_DIR="${V46_SPLIT_DIR:-change/splits_v46_33_reference_transition_${RUN_TS}}"
AUDIO="${V46_TARGET_AUDIO:-test_music_bank/dunhuangwu2.wav}"
CHANGE_DIR="${V46_CHANGE_DIR:-change}"
CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-160}"
REFINER_STEPS="${V46_REFINER_TRAIN_STEPS:-10000}"
DIFFUSION_STEPS_TRAIN="${V46_DIFFUSION_TRAIN_STEPS:-20000}"

mkdir -p "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/v46_33_reference_transition_overnight_${RUN_TS}.log"
PIDFILE="$RUN_ROOT/logs/v46_33_reference_transition_overnight.pid"

cat > "$RUN_ROOT/V46_33_RUN_ENV.txt" <<EOF
RUN_TS=$RUN_TS
RUN_ROOT=$RUN_ROOT
SPLIT_DIR=$SPLIT_DIR
AUDIO=$AUDIO
CHANGE_DIR=$CHANGE_DIR
CONTRASTIVE_EPOCHS=$CONTRASTIVE_EPOCHS
REFINER_STEPS=$REFINER_STEPS
DIFFUSION_STEPS_TRAIN=$DIFFUSION_STEPS_TRAIN
EOF

nohup bash -c '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

RUN_ROOT="'"$RUN_ROOT"'"
SPLIT_DIR="'"$SPLIT_DIR"'"
AUDIO="'"$AUDIO"'"
CHANGE_DIR="'"$CHANGE_DIR"'"
CONTRASTIVE_EPOCHS="'"$CONTRASTIVE_EPOCHS"'"
REFINER_STEPS="'"$REFINER_STEPS"'"
DIFFUSION_STEPS_TRAIN="'"$DIFFUSION_STEPS_TRAIN"'"

mkdir -p "$RUN_ROOT"

echo "[0/9] Apply patches"
if [[ -f tools/v46_research_contract_patch.py ]]; then
  python tools/v46_research_contract_patch.py
fi
if [[ -f tools/v46_json_safety_hotfix.py ]]; then
  python tools/v46_json_safety_hotfix.py
fi
python tools/v46_33_reference_transition_patch.py
python -m py_compile tools/v46_motionrag_diff.py tools/v46_33_reference_transition_patch.py tools/v46_33_relabel_change_event_db.py

echo "[1/9] Build source-level split manifest"
python tools/v46_make_change_splits.py \
  --motion_dir "$CHANGE_DIR" \
  --out_dir "$SPLIT_DIR" \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --allow_small

# Ensure all_manifest.csv exists. Some older split tools only write train/val/test.
if [[ ! -f "$SPLIT_DIR/all_manifest.csv" ]]; then
python - <<PY
from pathlib import Path
import csv
split_dir = Path("$SPLIT_DIR")
out = split_dir / "all_manifest.csv"
parts = [split_dir / "train_manifest.csv", split_dir / "val_manifest.csv", split_dir / "test_manifest.csv"]
rows = []
fieldnames = None
for p in parts:
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = r.fieldnames
        rows.extend(list(r))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print("[OK] wrote", out, "rows=", len(rows))
PY
fi

AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music change/music change/audio; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done

echo "[2/9] Build all-train Event-RAG DB"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
V46_DEVICE="${V46_DEVICE:-cuda}" \
V46_MANIFEST_SECONDARY_EVENT_SPLIT=1 \
V46_CHANG_E_BOUNDARY_EVENT_SPLIT=1 \
V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS="${V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS:-96}" \
V46_WINDOW_LEN="${V46_WINDOW_LEN:-120}" \
V46_HOP_LEN="${V46_HOP_LEN:-45}" \
V46_MIN_EVENT_FRAMES="${V46_MIN_EVENT_FRAMES:-45}" \
V46_MAX_EVENT_FRAMES="${V46_MAX_EVENT_FRAMES:-120}" \
V46_FILENAME_SEMANTIC_ENABLE=1 \
V46_CLASSIFICATION_SEMANTIC_ENABLE=1 \
V46_CHANG_E_EVENT_SEMANTIC_ENABLE=1 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  build-db \
  --motion_dirs "$CHANGE_DIR" \
  --manifest "$SPLIT_DIR/all_manifest.csv" \
  --audio_dirs "${AUDIO_DIRS[@]}" \
  --out_db "$RUN_ROOT/train_all_db"

echo "[3/9] Relabel Chang-E semantic event metadata"
python tools/v46_33_relabel_change_event_db.py \
  --db "$RUN_ROOT/train_all_db/events.npz" \
  --json "$RUN_ROOT/train_all_db_v46_33_relabel_report.json"

python tools/v46_quick_db_audit.py \
  --db "$RUN_ROOT/train_all_db/events.npz" \
  --json "$RUN_ROOT/train_all_db_audit_after_v46_33_relabel.json" \
  --strict

TRAIN_DB="$RUN_ROOT/train_all_db/events.npz"

echo "[4/9] Train V44 contrastive alignment"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
V46_DEVICE="${V46_DEVICE:-cuda}" \
V46_UNPAIRED_AUDIO_ENABLE=1 \
V46_UNPAIRED_DISABLE_MOTION_PROXY=1 \
V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1 \
V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=0 \
V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE=0 \
V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY=0 \
V46_CLASSIFICATION_SEMANTIC_ENABLE=1 \
V46_CHANG_E_EVENT_SEMANTIC_ENABLE=1 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-contrastive \
  --db "$TRAIN_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --epochs "$CONTRASTIVE_EPOCHS" \
  --out "$RUN_ROOT/v44_contrastive_v46_33.pt"

echo "[5/9] Train V45 reference-conditioned transition refiner"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
V46_DEVICE="${V46_DEVICE:-cuda}" \
V46_TRANSITION_TRAIN_MIN_FRAMES="${V46_TRANSITION_TRAIN_MIN_FRAMES:-10}" \
V46_TRANSITION_TRAIN_MAX_FRAMES="${V46_TRANSITION_TRAIN_MAX_FRAMES:-28}" \
V46_TRANSITION_MASK_HALO="${V46_TRANSITION_MASK_HALO:-6}" \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-refiner \
  --db "$TRAIN_DB" \
  --steps "$REFINER_STEPS" \
  --out "$RUN_ROOT/v45_refiner_v46_33.pt"

echo "[6/9] Train V46 reference-conditioned transition diffusion"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
PYTHONUNBUFFERED=1 \
V46_DEVICE="${V46_DEVICE:-cuda}" \
V46_DIFFUSION_STEPS="${V46_DIFFUSION_STEPS:-50}" \
V46_TRANSITION_TRAIN_MIN_FRAMES="${V46_TRANSITION_TRAIN_MIN_FRAMES:-10}" \
V46_TRANSITION_TRAIN_MAX_FRAMES="${V46_TRANSITION_TRAIN_MAX_FRAMES:-28}" \
V46_TRANSITION_MASK_HALO="${V46_TRANSITION_MASK_HALO:-6}" \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-diffusion \
  --db "$TRAIN_DB" \
  --steps "$DIFFUSION_STEPS_TRAIN" \
  --out "$RUN_ROOT/v46_diffusion_v46_33.pt"

COMMON_ENV=(
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  PYTHONUNBUFFERED=1
  V46_DEVICE="${V46_DEVICE:-cuda}"
  V46_ENABLE_TRUE_IK=1
  V46_ENABLE_REFINER=1
  V46_TRANSITION_BUDGET_ENABLE="${V46_TRANSITION_BUDGET_ENABLE:-1}"
  V46_TRANSITION_INBETWEEN_ENABLE="${V46_TRANSITION_INBETWEEN_ENABLE:-1}"
  V46_TRANSITION_MIN_FRAMES="${V46_TRANSITION_MIN_FRAMES:-10}"
  V46_TRANSITION_MAX_FRAMES="${V46_TRANSITION_MAX_FRAMES:-28}"
  V46_TRANSITION_RATIO="${V46_TRANSITION_RATIO:-0.18}"
  V46_TRANSITION_MASK_HALO="${V46_TRANSITION_MASK_HALO:-6}"
  V46_TRANSITION_MIN_CORE_FRAMES="${V46_TRANSITION_MIN_CORE_FRAMES:-30}"
  V46_CORE_WARP_MIN="${V46_CORE_WARP_MIN:-0.72}"
  V46_CORE_WARP_MAX="${V46_CORE_WARP_MAX:-1.38}"
  V46_REFINER_CORE_STRENGTH="${V46_REFINER_CORE_STRENGTH:-0.02}"
  V46_REFINER_TRANSITION_STRENGTH="${V46_REFINER_TRANSITION_STRENGTH:-1.00}"
  V46_DIFFUSION_CORE_STRENGTH="${V46_DIFFUSION_CORE_STRENGTH:-0.00}"
  V46_DIFFUSION_TRANSITION_STRENGTH="${V46_DIFFUSION_TRANSITION_STRENGTH:-0.72}"
  V46_DIFFUSION_REFERENCE_NOISE_SCALE="${V46_DIFFUSION_REFERENCE_NOISE_SCALE:-0.03}"
  V46_SEMANTIC_ROUTING_WEIGHT="${V46_SEMANTIC_ROUTING_WEIGHT:-0.90}"
  V46_ROUTE_SEMANTIC_BONUS_SCALE="${V46_ROUTE_SEMANTIC_BONUS_SCALE:-1.15}"
  V46_EVENT_FAMILY_BONUS="${V46_EVENT_FAMILY_BONUS:-0.75}"
  V46_MOTION_STAGE_ROLE_BONUS="${V46_MOTION_STAGE_ROLE_BONUS:-0.50}"
  V46_EVENT_QUALITY_WEIGHT="${V46_EVENT_QUALITY_WEIGHT:-0.16}"
  V46_ROUTE_SUPPORT_BONUS="${V46_ROUTE_SUPPORT_BONUS:-0.22}"
  V46_ROUTE_LOCOMOTION_BONUS="${V46_ROUTE_LOCOMOTION_BONUS:-0.28}"
  V46_ROUTE_FAMILY_RECENT_WINDOW="${V46_ROUTE_FAMILY_RECENT_WINDOW:-5}"
  V46_ROUTE_FAMILY_PENALTY_CAP="${V46_ROUTE_FAMILY_PENALTY_CAP:-0.16}"
  V46_ROUTE_SOURCE_REPEAT_PENALTY="${V46_ROUTE_SOURCE_REPEAT_PENALTY:-0.04}"
  V46_ROUTE_SOURCE_RUN_HARD_PENALTY="${V46_ROUTE_SOURCE_RUN_HARD_PENALTY:-0.35}"
)

echo "[7/9] Generate reference-transition refiner+IK diagnostic"
env "${COMMON_ENV[@]}" V46_ENABLE_DIFFUSION=0 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  generate \
  --audio "$AUDIO" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --db "$TRAIN_DB" \
  --contrastive "$RUN_ROOT/v44_contrastive_v46_33.pt" \
  --refiner "$RUN_ROOT/v45_refiner_v46_33.pt" \
  --out "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_refiner_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_refiner_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_refiner_ik.mp4"

echo "[8/9] Generate final reference-transition diffusion+IK"
env "${COMMON_ENV[@]}" V46_ENABLE_DIFFUSION=1 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  generate \
  --audio "$AUDIO" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --db "$TRAIN_DB" \
  --contrastive "$RUN_ROOT/v44_contrastive_v46_33.pt" \
  --refiner "$RUN_ROOT/v45_refiner_v46_33.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion_v46_33.pt" \
  --out "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_diffusion_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_diffusion_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_33_ref_transition_diffusion_ik.mp4"

echo "[9/9] Summarize selected events and final audits"
python - <<PY
import json, os, numpy as np
run = "$RUN_ROOT"
db = np.load(os.path.join(run, "train_all_db/events.npz"), allow_pickle=True)
summary = {}
for name in [
    "dunhuangwu2_v46_33_ref_transition_refiner_ik.report.json",
    "dunhuangwu2_v46_33_ref_transition_diffusion_ik.report.json",
]:
    p = os.path.join(run, name)
    if not os.path.exists(p):
        summary[name] = {"missing": True}
        continue
    rep = json.load(open(p, encoding="utf-8"))
    idx = rep.get("selected_event_indices", [])
    item = {"selected_events": len(idx), "final_audit": rep.get("final_audit", {}), "seam_mask_stats": rep.get("stage_reports", {}).get("seam_mask_stats", {})}
    for k in ["dance_keys", "event_families", "music_alignment_labels", "motion_stage_roles", "locomotion_labels", "support_labels"]:
        if k in db.files and idx:
            arr = db[k].astype(str)
            vals, cnt = np.unique(arr[idx], return_counts=True)
            item["selected_" + k] = [[str(v), int(c)] for v, c in zip(vals, cnt)]
    summary[name] = item
out = os.path.join(run, "V46_33_FINAL_SUMMARY.json")
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("[SUMMARY]", out)
PY

echo "[DONE] V46.33 reference-conditioned transition-masked MotionRAG-Diff finished."
echo "[RUN_ROOT] $RUN_ROOT"
echo "[REFINER_MP4] $RUN_ROOT/dunhuangwu2_v46_33_ref_transition_refiner_ik.mp4"
echo "[DIFFUSION_MP4] $RUN_ROOT/dunhuangwu2_v46_33_ref_transition_diffusion_ik.mp4"
' > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
disown

echo "[RUNNING] PID=$(cat "$PIDFILE")"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[LOG] $LOG"
echo "[MONITOR] tail -f $LOG"
