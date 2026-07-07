#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

CFG="${V46_CONFIG:-configs/v46_motionrag_diff_config.json}"
AUDIO="${V46_TARGET_AUDIO:-test_music_bank/dunhuangwu2.wav}"
[[ -f "$AUDIO" ]] || AUDIO="data/music/dunhuangwu2.wav"
[[ -f "$AUDIO" ]] || AUDIO="custom_music/dunhuangwu2.wav"
if [[ ! -f "$AUDIO" ]]; then
  echo "[V46.34 ERROR] target audio not found. Set V46_TARGET_AUDIO=/path/to/song.wav" >&2
  exit 2
fi

RUN_ROOT="${V46_RUN_ROOT:-output/v46_34_pretrained_router_reference_transition_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/logs"
LOG="$RUN_ROOT/logs/v46_34_pretrained_router_reference_transition_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "[V46.34 RUN_ROOT] $RUN_ROOT"
echo "[V46.34 LOG] $LOG"
echo "[V46.34 AUDIO] $AUDIO"

# -----------------------------------------------------------------------------
# 0. Pretrained music-structure weights.
# -----------------------------------------------------------------------------
export V26_ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
export V26_PLANNER_CKPT="${V26_PLANNER_CKPT:-output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt}"
export V27_HG_PLANNER_CKPT="${V27_HG_PLANNER_CKPT:-output/v27_hg_overnight_20260609_222844/planner_seed_20260615/checkpoints/best.pt}"
export V34_CONTACT_INR_CKPT="${V34_CONTACT_INR_CKPT:-output/v34_dense_boundary_train_overnight_20260621_201033/v34_contact_inr_training/checkpoints/contact_finetuned_best.pt}"

# V26 slot planner needs the old V26/V34 retrieval index and V23 duration ckpt.
export V26_INDEX_JSON="${V26_INDEX_JSON:-data/v35_source_aware/v21_shared_event_index_source_aware.json}"
export V26_DURATION_INDEX_NPZ="${V26_DURATION_INDEX_NPZ:-data/v35_source_aware/v26_music_dominant_duration_index_source_aware.npz}"
export V26_HIERARCHY_INDEX_NPZ="${V26_HIERARCHY_INDEX_NPZ:-data/v35_source_aware/v34_hierarchical_event_index_source_aware.npz}"
export V26_FEATURE_CACHE="${V26_FEATURE_CACHE:-$RUN_ROOT/v26_music_features}"

if [[ -z "${V26_V23_CKPT:-}" ]]; then
  V26_V23_CKPT="$(find output checkpoints runs . -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' \) 2>/dev/null | grep -Ei 'v23|duration|monotonic|timewarp|time_warp|warp' | sort | tail -1 || true)"
  export V26_V23_CKPT
fi

# Strict by default: unseen-song slots must be produced by trained V21/V26 weights.
export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS="${V46_REQUIRE_PRETRAINED_ROUTER_SLOTS:-1}"
export V46_34_ALLOW_SEMANTIC_FALLBACK="${V46_34_ALLOW_SEMANTIC_FALLBACK:-0}"

for f in "$V26_ROUTER_CKPT" "$V26_PLANNER_CKPT" "$V26_INDEX_JSON" "$V26_DURATION_INDEX_NPZ"; do
  [[ -f "$f" ]] || { echo "[V46.34 ERROR] missing required file: $f" >&2; exit 2; }
done
if [[ "$V46_34_ALLOW_SEMANTIC_FALLBACK" != "1" ]]; then
  [[ -f "${V26_V23_CKPT:-}" ]] || {
    echo "[V46.34 ERROR] missing V26_V23_CKPT. Set it to the V23 duration/time-warp checkpoint used by schedule_v26_whole_song.py." >&2
    echo "[V46.34 HINT] find . -type f \( -iname '*.pt' -o -iname '*.pth' -o -iname '*.ckpt' \) | grep -Ei 'v23|duration|monotonic|timewarp|warp'" >&2
    exit 2
  }
fi

# -----------------------------------------------------------------------------
# 1. Apply code patches. V46.33 handles transition-budget/reference masking;
#    V46.34 connects pretrained router slots to generate().
# -----------------------------------------------------------------------------
echo "[1/10] Apply V46.33 transition-reference patch and V46.34 router-slot patch"
if [[ -f tools/v46_33_reference_transition_patch.py ]]; then
  python tools/v46_33_reference_transition_patch.py
else
  echo "[V46.34 WARN] tools/v46_33_reference_transition_patch.py not found; continuing with existing V46 concat/refiner/diffusion code."
fi
python tools/v46_34_router_slot_patch.py
python -m py_compile tools/v46_motionrag_diff.py tools/v46_34_pretrained_music_slot_plan.py tools/v46_34_router_slot_patch.py

# -----------------------------------------------------------------------------
# 2. Build all-train qualitative DB from the new change dataset.
# -----------------------------------------------------------------------------
CHANGE_DIR="${V46_CHANGE_DIR:-change}"
[[ -e "$CHANGE_DIR" ]] || { echo "[V46.34 ERROR] missing change dataset directory: $CHANGE_DIR" >&2; exit 2; }
TRAIN_DB_DIR="$RUN_ROOT/train_all_db"
TRAIN_DB="$TRAIN_DB_DIR/events.npz"

AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music change/music change/audio data/v21_router_music_999/splits/train data/v21_router_music_999/splits/val data/v21_router_music_valid25 data/v21_router_music_train_pcm16; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done

SEMANTIC_DIRS=()
IFS=':' read -ra _sem_env <<< "${V46_EXTERNAL_MUSIC_SEMANTIC_DIRS:-music_semantics:external_music_semantics:output/music_semantics}"
for d in "${_sem_env[@]}"; do
  [[ -e "$d" ]] && SEMANTIC_DIRS+=("$d")
done
MUSIC_SEMANTIC_ARGS=()
if [[ ${#SEMANTIC_DIRS[@]} -gt 0 ]]; then
  MUSIC_SEMANTIC_ARGS=(--music_semantic_dirs "${SEMANTIC_DIRS[@]}")
fi

MANIFEST_ARGS=()
if [[ -n "${V46_MANIFEST:-}" ]]; then
  MANIFEST_ARGS=(--manifest "$V46_MANIFEST")
elif [[ -f "${V46_SPLIT_DIR:-}/all_manifest.csv" ]]; then
  MANIFEST_ARGS=(--manifest "${V46_SPLIT_DIR}/all_manifest.csv")
elif [[ -f "change/all_manifest.csv" ]]; then
  MANIFEST_ARGS=(--manifest "change/all_manifest.csv")
fi

echo "[2/10] Build all-train Event-RAG DB from $CHANGE_DIR"
python tools/v46_motionrag_diff.py --config "$CFG" build-db \
  --motion_dirs "$CHANGE_DIR" \
  "${MANIFEST_ARGS[@]}" \
  --audio_dirs "${AUDIO_DIRS[@]}" \
  --out_db "$TRAIN_DB_DIR"

if [[ -f tools/v46_33_relabel_change_event_db.py ]]; then
  echo "[3/10] Relabel Chang-E/Dunhuang semantic event metadata"
  python tools/v46_33_relabel_change_event_db.py --db "$TRAIN_DB"
else
  echo "[3/10] Relabel script not found; skip semantic relabel."
fi

if [[ -f tools/v46_quick_db_audit.py ]]; then
  python tools/v46_quick_db_audit.py --db "$TRAIN_DB" --json "$RUN_ROOT/train_all_db_audit.json" --strict
fi

# -----------------------------------------------------------------------------
# 3. Generate pretrained router slot plan for unseen whole-song music.
# -----------------------------------------------------------------------------
SLOT_JSON="$RUN_ROOT/$(basename "${AUDIO%.*}")_v46_34_pretrained_router_slots.json"
echo "[4/10] Build V46.34 pretrained V21/V26 router slot plan"
python tools/v46_34_pretrained_music_slot_plan.py \
  --audio "$AUDIO" \
  --out_json "$SLOT_JSON" \
  --router_ckpt "$V26_ROUTER_CKPT" \
  --planner_ckpt "$V26_PLANNER_CKPT" \
  --v23_ckpt "${V26_V23_CKPT:-}" \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --hierarchy_index_npz "${V26_HIERARCHY_INDEX_NPZ:-}" \
  --feature_dir "$V26_FEATURE_CACHE" \
  --music_semantic_dirs "${SEMANTIC_DIRS[@]}"

# -----------------------------------------------------------------------------
# 4. Train V44/V45/V46 on the new DB. Use env switches for cost control.
# -----------------------------------------------------------------------------
export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_UNPAIRED_AUDIO_ENABLE="${V46_UNPAIRED_AUDIO_ENABLE:-1}"
export V46_UNPAIRED_DISABLE_MOTION_PROXY="${V46_UNPAIRED_DISABLE_MOTION_PROXY:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED="${V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY="${V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY:-0}"

export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-1}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-1}"
export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-160}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-10000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-20000}"
export V46_DIFFUSION_STEPS="${V46_DIFFUSION_STEPS:-50}"

# V46.33/V46.34 reference-transition controls.
export V46_TRANSITION_BUDGET_ENABLE="${V46_TRANSITION_BUDGET_ENABLE:-1}"
export V46_TRANSITION_INBETWEEN_ENABLE="${V46_TRANSITION_INBETWEEN_ENABLE:-1}"
export V46_TRANSITION_MIN_FRAMES="${V46_TRANSITION_MIN_FRAMES:-10}"
export V46_TRANSITION_MAX_FRAMES="${V46_TRANSITION_MAX_FRAMES:-28}"
export V46_TRANSITION_RATIO="${V46_TRANSITION_RATIO:-0.18}"
export V46_TRANSITION_MASK_HALO="${V46_TRANSITION_MASK_HALO:-6}"
export V46_TRANSITION_MIN_CORE_FRAMES="${V46_TRANSITION_MIN_CORE_FRAMES:-30}"
export V46_CORE_WARP_MIN="${V46_CORE_WARP_MIN:-0.72}"
export V46_CORE_WARP_MAX="${V46_CORE_WARP_MAX:-1.38}"
export V46_REFINER_CORE_STRENGTH="${V46_REFINER_CORE_STRENGTH:-0.02}"
export V46_REFINER_TRANSITION_STRENGTH="${V46_REFINER_TRANSITION_STRENGTH:-1.00}"
export V46_DIFFUSION_CORE_STRENGTH="${V46_DIFFUSION_CORE_STRENGTH:-0.00}"
export V46_DIFFUSION_TRANSITION_STRENGTH="${V46_DIFFUSION_TRANSITION_STRENGTH:-0.72}"
export V46_DIFFUSION_REFERENCE_NOISE_SCALE="${V46_DIFFUSION_REFERENCE_NOISE_SCALE:-0.03}"

CONTRASTIVE="$RUN_ROOT/v44_contrastive_v46_34.pt"
REFINER="$RUN_ROOT/v45_refiner_v46_34.pt"
DIFFUSION="$RUN_ROOT/v46_diffusion_v46_34.pt"

echo "[5/10] Train V44 contrastive alignment"
python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$TRAIN_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --epochs "$V46_CONTRASTIVE_EPOCHS" \
  --out "$CONTRASTIVE"

REFINER_ARG=()
if [[ "$V46_ENABLE_REFINER" == "1" ]]; then
  echo "[6/10] Train V45 reference-conditioned transition refiner"
  python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
    --db "$TRAIN_DB" \
    --steps "$V46_REFINER_TRAIN_STEPS" \
    --out "$REFINER"
  REFINER_ARG=(--refiner "$REFINER")
fi

DIFFUSION_ARG=()
if [[ "$V46_ENABLE_DIFFUSION" == "1" ]]; then
  echo "[7/10] Train V46 reference-conditioned masked diffusion"
  python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
    --db "$TRAIN_DB" \
    --steps "$V46_DIFFUSION_TRAIN_STEPS" \
    --out "$DIFFUSION"
  DIFFUSION_ARG=(--diffusion "$DIFFUSION")
fi

# -----------------------------------------------------------------------------
# 5. Generate explicit stages for ablation.
# -----------------------------------------------------------------------------
MG_OUT="$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.npy"
MG_JSON="$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.report.json"
MG_MP4="$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.mp4"
echo "[8/10] Generate Stage-1 router-slot MotionRAG/motion_mg baseline"
V46_ENABLE_REFINER=0 V46_ENABLE_DIFFUSION=0 python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --slots_json "$SLOT_JSON" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  --contrastive "$CONTRASTIVE" \
  --out "$MG_OUT" \
  --json "$MG_JSON" \
  --render_output "$MG_MP4"
python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$MG_OUT" --json "$RUN_ROOT/motion_mg_audit.json" || true

REF_OUT="$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.npy"
REF_JSON="$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.report.json"
REF_MP4="$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.mp4"
echo "[9/10] Generate V45 refiner + IK with pretrained router slots"
V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=0 python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --slots_json "$SLOT_JSON" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  --contrastive "$CONTRASTIVE" \
  "${REFINER_ARG[@]}" \
  --out "$REF_OUT" \
  --json "$REF_JSON" \
  --render_output "$REF_MP4"
python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$REF_OUT" --json "$RUN_ROOT/refiner_ik_audit.json" || true

DIFF_OUT="$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.npy"
DIFF_JSON="$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.report.json"
DIFF_MP4="$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.mp4"
echo "[10/10] Generate V46 diffusion + IK with pretrained router slots"
if [[ ${#DIFFUSION_ARG[@]} -gt 0 ]]; then
  V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=1 python tools/v46_motionrag_diff.py --config "$CFG" generate \
    --audio "$AUDIO" \
    --slots_json "$SLOT_JSON" \
    "${MUSIC_SEMANTIC_ARGS[@]}" \
    --db "$TRAIN_DB" \
    --contrastive "$CONTRASTIVE" \
    "${REFINER_ARG[@]}" \
    "${DIFFUSION_ARG[@]}" \
    --out "$DIFF_OUT" \
    --json "$DIFF_JSON" \
    --render_output "$DIFF_MP4"
  python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$DIFF_OUT" --json "$RUN_ROOT/diffusion_ik_audit.json" || true
else
  echo "[V46.34 INFO] diffusion disabled; skip diffusion_ik generation."
fi

cat > "$RUN_ROOT/V46_34_FINAL_SUMMARY.json" <<EOF
{
  "version": "V46.34_pretrained_router_reference_transition",
  "run_root": "$RUN_ROOT",
  "audio": "$AUDIO",
  "train_db": "$TRAIN_DB",
  "slot_plan": "$SLOT_JSON",
  "router_ckpt": "$V26_ROUTER_CKPT",
  "planner_ckpt": "$V26_PLANNER_CKPT",
  "v23_ckpt": "${V26_V23_CKPT:-}",
  "contrastive_ckpt": "$CONTRASTIVE",
  "refiner_ckpt": "$REFINER",
  "diffusion_ckpt": "$DIFFUSION",
  "stage1_motion_mg": "$MG_OUT",
  "refiner_ik": "$REF_OUT",
  "diffusion_ik": "$DIFF_OUT",
  "log": "$LOG"
}
EOF

echo "[V46.34 DONE]"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[SLOT_PLAN] $SLOT_JSON"
echo "[MOTION_MG] $MG_OUT"
echo "[REFINER_IK] $REF_OUT"
echo "[DIFFUSION_IK] $DIFF_OUT"
echo "[SUMMARY] $RUN_ROOT/V46_34_FINAL_SUMMARY.json"
