#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE

RUN_TS=${RUN_TS:-$(date +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-output/v46_32_chang_e_alltrain_transition_${RUN_TS}}
SPLIT_DIR=${SPLIT_DIR:-change/splits_v46_32_alltrain_${RUN_TS}}
AUDIO=${AUDIO:-test_music_bank/dunhuangwu2.wav}
mkdir -p "$RUN_ROOT/logs"

# Patch current source once.  This preserves the old V46.31 concat as fallback.
python tools/v46_research_contract_patch.py || true
python tools/v46_json_safety_hotfix.py || true
python tools/v46_32_transition_budget_patch.py
python -m py_compile tools/v46_motionrag_diff.py tools/v46_32_transition_budget_patch.py tools/v46_32_relabel_change_event_db.py

# Source-level split is still produced for audit, but all_manifest is used for all-train demo DB.
python tools/v46_make_change_splits.py \
  --motion_dir change \
  --out_dir "$SPLIT_DIR" \
  --train_ratio 0.70 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --allow_small

# Some split tools do not emit all_manifest.csv.  Create it if missing.
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
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = r.fieldnames
        rows.extend(list(r))
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rows)
print('[OK] wrote', out, 'rows=', len(rows))
PY
fi

AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music change/music change/audio; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1
export V46_DEVICE=${V46_DEVICE:-cuda}
export V46_BVH_RESAMPLE_TO_CONFIG_FPS=1
export V46_MANIFEST_SECONDARY_EVENT_SPLIT=1
export V46_CHANG_E_BOUNDARY_EVENT_SPLIT=1
export V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS=${V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS:-96}
export V46_WINDOW_LEN=${V46_WINDOW_LEN:-120}
export V46_HOP_LEN=${V46_HOP_LEN:-45}
export V46_MIN_EVENT_FRAMES=${V46_MIN_EVENT_FRAMES:-45}
export V46_MAX_EVENT_FRAMES=${V46_MAX_EVENT_FRAMES:-120}
export V46_FILENAME_SEMANTIC_ENABLE=1
export V46_CLASSIFICATION_SEMANTIC_ENABLE=1
export V46_CHANG_E_EVENT_SEMANTIC_ENABLE=1

# V46.32 transition-budget switches.
export V46_TRANSITION_BUDGET_ENABLE=${V46_TRANSITION_BUDGET_ENABLE:-1}
export V46_TRANSITION_INBETWEEN_ENABLE=${V46_TRANSITION_INBETWEEN_ENABLE:-1}
export V46_TRANSITION_MIN_FRAMES=${V46_TRANSITION_MIN_FRAMES:-10}
export V46_TRANSITION_MAX_FRAMES=${V46_TRANSITION_MAX_FRAMES:-28}
export V46_TRANSITION_RATIO=${V46_TRANSITION_RATIO:-0.18}
export V46_TRANSITION_MASK_HALO=${V46_TRANSITION_MASK_HALO:-6}
export V46_TRANSITION_MIN_CORE_FRAMES=${V46_TRANSITION_MIN_CORE_FRAMES:-30}
export V46_CORE_WARP_MIN=${V46_CORE_WARP_MIN:-0.72}
export V46_CORE_WARP_MAX=${V46_CORE_WARP_MAX:-1.38}
export V46_CORE_WARP_CLAMP_ENABLE=${V46_CORE_WARP_CLAMP_ENABLE:-1}

# Semantic routing after relabel.
export V46_SEMANTIC_ROUTING_WEIGHT=${V46_SEMANTIC_ROUTING_WEIGHT:-0.90}
export V46_ROUTE_SEMANTIC_BONUS_SCALE=${V46_ROUTE_SEMANTIC_BONUS_SCALE:-1.15}
export V46_EVENT_FAMILY_BONUS=${V46_EVENT_FAMILY_BONUS:-0.75}
export V46_MOTION_STAGE_ROLE_BONUS=${V46_MOTION_STAGE_ROLE_BONUS:-0.50}
export V46_EVENT_QUALITY_WEIGHT=${V46_EVENT_QUALITY_WEIGHT:-0.16}
export V46_ROUTE_SUPPORT_BONUS=${V46_ROUTE_SUPPORT_BONUS:-0.22}
export V46_ROUTE_LOCOMOTION_BONUS=${V46_ROUTE_LOCOMOTION_BONUS:-0.28}
export V46_ROUTE_FAMILY_RECENT_WINDOW=${V46_ROUTE_FAMILY_RECENT_WINDOW:-5}
export V46_ROUTE_FAMILY_PENALTY_CAP=${V46_ROUTE_FAMILY_PENALTY_CAP:-0.16}
export V46_ROUTE_SOURCE_REPEAT_PENALTY=${V46_ROUTE_SOURCE_REPEAT_PENALTY:-0.04}
export V46_ROUTE_SOURCE_RUN_HARD_PENALTY=${V46_ROUTE_SOURCE_RUN_HARD_PENALTY:-0.35}

# Realistic low-resource music grounding.
export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=0
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE=0
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY=0

LOG="$RUN_ROOT/logs/v46_32_overnight_$(date +%Y%m%d_%H%M%S).log"
PID="$RUN_ROOT/logs/v46_32_overnight.pid"

nohup bash -c '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE
RUN_ROOT="'"$RUN_ROOT"'"
SPLIT_DIR="'"$SPLIT_DIR"'"
AUDIO="'"$AUDIO"'"
TRAIN_DB="$RUN_ROOT/train_all_db/events.npz"
AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music change/music change/audio; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done

echo "[1/8] build all-train DB"
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  build-db \
  --motion_dirs change \
  --manifest "$SPLIT_DIR/all_manifest.csv" \
  --audio_dirs "${AUDIO_DIRS[@]}" \
  --out_db "$RUN_ROOT/train_all_db"

echo "[2/8] relabel DB semantics"
python tools/v46_32_relabel_change_event_db.py --db "$TRAIN_DB"
python tools/v46_quick_db_audit.py \
  --db "$TRAIN_DB" \
  --json "$RUN_ROOT/train_all_db_audit_after_v46_32_relabel.json" \
  --strict

echo "[3/8] train V44 contrastive"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 V46_DEVICE=${V46_DEVICE:-cuda} \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-contrastive \
  --db "$TRAIN_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --epochs ${V46_CONTRASTIVE_EPOCHS:-160} \
  --out "$RUN_ROOT/v44_contrastive_v46_32.pt"

echo "[4/8] train V45 refiner"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 V46_DEVICE=${V46_DEVICE:-cuda} \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-refiner \
  --db "$TRAIN_DB" \
  --steps ${V46_REFINER_TRAIN_STEPS:-10000} \
  --out "$RUN_ROOT/v45_refiner_v46_32.pt"

echo "[5/8] train V46 diffusion"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 V46_DEVICE=${V46_DEVICE:-cuda} \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-diffusion \
  --db "$TRAIN_DB" \
  --steps ${V46_DIFFUSION_TRAIN_STEPS:-20000} \
  --diffusion_steps ${V46_DIFFUSION_STEPS:-50} \
  --out "$RUN_ROOT/v46_diffusion_v46_32.pt"

echo "[6/8] generate refiner + IK diagnostic"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 V46_DEVICE=${V46_DEVICE:-cuda} \
V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  generate \
  --audio "$AUDIO" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --db "$TRAIN_DB" \
  --contrastive "$RUN_ROOT/v44_contrastive_v46_32.pt" \
  --refiner "$RUN_ROOT/v45_refiner_v46_32.pt" \
  --out "$RUN_ROOT/dunhuangwu2_v46_32_transition_refiner_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_32_transition_refiner_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_32_transition_refiner_ik.mp4"

echo "[7/8] generate refiner + diffusion + IK final"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1 V46_DEVICE=${V46_DEVICE:-cuda} \
V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=1 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  generate \
  --audio "$AUDIO" \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --db "$TRAIN_DB" \
  --contrastive "$RUN_ROOT/v44_contrastive_v46_32.pt" \
  --refiner "$RUN_ROOT/v45_refiner_v46_32.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion_v46_32.pt" \
  --out "$RUN_ROOT/dunhuangwu2_v46_32_transition_diffusion_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_32_transition_diffusion_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_32_transition_diffusion_ik.mp4"

echo "[8/8] summarize selected event families and audits"
python - <<PY
import json, os, numpy as np
run = "$RUN_ROOT"
db = np.load(os.path.join(run, "train_all_db/events.npz"), allow_pickle=True)
for name in ["dunhuangwu2_v46_32_transition_refiner_ik.report.json", "dunhuangwu2_v46_32_transition_diffusion_ik.report.json"]:
    p = os.path.join(run, name)
    print("\n====", name, "====")
    if not os.path.exists(p):
        print("missing"); continue
    rep = json.load(open(p, encoding="utf-8"))
    idx = rep.get("selected_event_indices", [])
    print("selected events", len(idx))
    for k in ["dance_keys", "event_families", "music_alignment_labels", "motion_stage_roles", "locomotion_labels"]:
        if k in db.files and idx:
            arr = db[k].astype(str)
            vals, cnt = np.unique(arr[idx], return_counts=True)
            print("--", k)
            for v,c in zip(vals,cnt): print(f"{v:28s} {int(c)}")
    print(json.dumps(rep.get("final_audit", {}), indent=2, ensure_ascii=False))
PY

echo "[DONE] V46.32 overnight run complete: $RUN_ROOT"
' > "$LOG" 2>&1 &

echo $! | tee "$PID"
disown || true

echo "[RUNNING] PID=$(cat "$PID")"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[LOG] $LOG"
echo "tail -f $LOG"
