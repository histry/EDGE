#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# V46.31 official Chang-E MotionRAG-Diff experiment
# -----------------------------------------------------------------------------
# Scientific policy:
#   * Source-level train/val/test split is created before event slicing.
#   * Long Chang-E BVH sources are then sliced only inside each split to build
#     split-specific Event-RAG databases.
#   * Stage 1: contrastive retrieval + motion graph gives motion_mg.
#   * Stage 2: residual refiner/diffusion + IK gives motion_diff.
#   * train_db is the only Event-RAG memory used for training and generation.
#   * val_db/test_db are built and audited for evaluation/analysis only.
#   * all-change DB can be built only as qualitative_demo/upper_bound.
# -----------------------------------------------------------------------------

python tools/v46_research_contract_patch.py

export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_BVH_RESAMPLE_TO_CONFIG_FPS="${V46_BVH_RESAMPLE_TO_CONFIG_FPS:-1}"
export V46_SOURCE_GROUP_MODE="${V46_SOURCE_GROUP_MODE:-filename}"
export V46_FILENAME_SEMANTIC_ENABLE="${V46_FILENAME_SEMANTIC_ENABLE:-1}"
export V46_CLASSIFICATION_SEMANTIC_ENABLE="${V46_CLASSIFICATION_SEMANTIC_ENABLE:-1}"

# Enriched Chang-E cultural/action semantics for Stage-1 retrieval and Stage-2 conditioning.
export V46_SEMANTIC_ROUTING_WEIGHT="${V46_SEMANTIC_ROUTING_WEIGHT:-0.76}"
export V46_EVENT_FAMILY_BONUS="${V46_EVENT_FAMILY_BONUS:-0.62}"
export V46_MOTION_STAGE_ROLE_BONUS="${V46_MOTION_STAGE_ROLE_BONUS:-0.40}"
export V46_CHANG_E_EVENT_SEMANTIC_ENABLE="${V46_CHANG_E_EVENT_SEMANTIC_ENABLE:-1}"
export V46_PREFERRED_DANCE_KEY_BONUS="${V46_PREFERRED_DANCE_KEY_BONUS:-0.28}"
export V46_ROUTE_NATURAL_DURATION_WEIGHT="${V46_ROUTE_NATURAL_DURATION_WEIGHT:-0.22}"
export V46_ROUTE_FAMILY_BALANCE_PENALTY="${V46_ROUTE_FAMILY_BALANCE_PENALTY:-0.20}"
export V46_ROUTE_FAMILY_RECENT_WINDOW="${V46_ROUTE_FAMILY_RECENT_WINDOW:-8}"
export V46_ROUTE_FAMILY_PENALTY_CAP="${V46_ROUTE_FAMILY_PENALTY_CAP:-0.25}"
export V46_ROUTE_DANCE_KEY_REPEAT_PENALTY="${V46_ROUTE_DANCE_KEY_REPEAT_PENALTY:-0.18}"
export V46_ROUTE_FAMILY_REPEAT_PENALTY="${V46_ROUTE_FAMILY_REPEAT_PENALTY:-0.12}"
export V46_ROUTE_SOURCE_REPEAT_PENALTY="${V46_ROUTE_SOURCE_REPEAT_PENALTY:-0.12}"
export V46_ROUTE_MOTIF_RECALL_BONUS="${V46_ROUTE_MOTIF_RECALL_BONUS:-0.14}"
export V46_ROUTE_DEBUG_TOPK="${V46_ROUTE_DEBUG_TOPK:-12}"
export V46_CHANG_E_BOUNDARY_EVENT_SPLIT="${V46_CHANG_E_BOUNDARY_EVENT_SPLIT:-1}"
export V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS="${V46_CHANG_E_BOUNDARY_MAX_EXTRA_STARTS:-96}"
export V46_CHANG_E_MIN_EVENT_QUALITY="${V46_CHANG_E_MIN_EVENT_QUALITY:-0.22}"
export V46_CHANG_E_KEEP_POSE_ANCHOR_QUALITY="${V46_CHANG_E_KEEP_POSE_ANCHOR_QUALITY:-0.16}"
export V46_EVENT_QUALITY_WEIGHT="${V46_EVENT_QUALITY_WEIGHT:-0.22}"
export V46_ROUTE_SUPPORT_BONUS="${V46_ROUTE_SUPPORT_BONUS:-0.12}"
export V46_ROUTE_LOCOMOTION_BONUS="${V46_ROUTE_LOCOMOTION_BONUS:-0.14}"
export V46_ROUTE_STAGE_SEQUENCE_WEIGHT="${V46_ROUTE_STAGE_SEQUENCE_WEIGHT:-0.16}"
export V46_ROUTE_SOURCE_RUN_HARD_PENALTY="${V46_ROUTE_SOURCE_RUN_HARD_PENALTY:-0.30}"
export V46_ROUTE_SEMANTIC_BONUS_SCALE="${V46_ROUTE_SEMANTIC_BONUS_SCALE:-1.50}"
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED="${V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_DIRS="${V46_EXTERNAL_MUSIC_SEMANTIC_DIRS:-music_semantics:external_music_semantics:output/music_semantics}"
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY="${V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY:-0}"
export V46_UNPAIRED_AUDIO_ENABLE="${V46_UNPAIRED_AUDIO_ENABLE:-1}"
export V46_UNPAIRED_DISABLE_MOTION_PROXY="${V46_UNPAIRED_DISABLE_MOTION_PROXY:-1}"

# Chang-E has long 1--6 minute BVH sequences.  Official experiments keep
# source-level split, then slice within each split so the RAG memory contains
# short, retrievable motifs rather than one huge event per source.
export V46_MANIFEST_SECONDARY_EVENT_SPLIT="${V46_MANIFEST_SECONDARY_EVENT_SPLIT:-1}"
export V46_WINDOW_LEN="${V46_WINDOW_LEN:-120}"          # 4.0 s at 30 fps
export V46_HOP_LEN="${V46_HOP_LEN:-45}"                # 1.5 s stride
export V46_MIN_EVENT_FRAMES="${V46_MIN_EVENT_FRAMES:-45}"
export V46_MAX_EVENT_FRAMES="${V46_MAX_EVENT_FRAMES:-120}"
export V46_OVERLAP="${V46_OVERLAP:-12}"
export V46_MIN_TRAIN_EVENTS="${V46_MIN_TRAIN_EVENTS:-24}"

export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_ENABLE_ROOT_Y_PHYSICS="${V46_ENABLE_ROOT_Y_PHYSICS:-1}"
export V46_ROOT_Y_MAX_FLIGHT_SECONDS="${V46_ROOT_Y_MAX_FLIGHT_SECONDS:-1.2}"
export V46_ROOT_Y_DAMPING_MAX_SECONDS="${V46_ROOT_Y_DAMPING_MAX_SECONDS:-0.28}"

# MotionRAG-Diff two-stage default: Stage 1 retrieval/motion graph + Stage 2 diffusion refinement.
export V46_ENABLE_CONTRASTIVE="${V46_ENABLE_CONTRASTIVE:-1}"
export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-1}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-1}"
export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-120}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-8000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-15000}"

CFG="${V46_CONFIG:-configs/v46_motionrag_diff_config.json}"
CHANGE_DIR="${V46_CHANGE_DIR:-change}"
SPLIT_DIR="${V46_SPLIT_DIR:-$CHANGE_DIR/splits_official}"
RUN_ROOT="${V46_RUN_ROOT:-output/v46_31_chang_e_motionrag_diff_$(date +%Y%m%d_%H%M%S)}"
export RUN_ROOT
mkdir -p "$RUN_ROOT" output
printf '%s\n' "$RUN_ROOT" > output/LATEST_V46_CHANG_E_OFFICIAL.txt

if [[ ! -d "$CHANGE_DIR" ]]; then
  echo "[V46.31 ERROR] $CHANGE_DIR not found. Put Chang-E BVH/manifest under EDGE/$CHANGE_DIR." >&2
  exit 2
fi

INPUT_MANIFEST="${V46_CHANGE_MANIFEST:-}"
if [[ -z "$INPUT_MANIFEST" ]]; then
  for m in "$CHANGE_DIR/manifest.csv" "$CHANGE_DIR/manifest.tsv" manifest.csv data/manifest.csv; do
    if [[ -f "$m" ]]; then INPUT_MANIFEST="$m"; break; fi
  done
fi

SPLIT_ARGS=(--motion_dir "$CHANGE_DIR" --out_dir "$SPLIT_DIR" --train_ratio "${V46_TRAIN_RATIO:-0.70}" --val_ratio "${V46_VAL_RATIO:-0.15}" --test_ratio "${V46_TEST_RATIO:-0.15}" --seed "${V46_SPLIT_SEED:-42}")
if [[ -n "$INPUT_MANIFEST" ]]; then
  SPLIT_ARGS+=(--manifest "$INPUT_MANIFEST")
fi
if [[ "${V46_ALLOW_SMALL_SPLIT:-0}" == "1" ]]; then
  SPLIT_ARGS+=(--allow_small)
fi

python tools/v46_make_change_splits.py "${SPLIT_ARGS[@]}"
python tools/v46_audit_change_splits.py --split_dir "$SPLIT_DIR" --json "$RUN_ROOT/split_audit.json" --strict
cp "$SPLIT_DIR/split_report.json" "$RUN_ROOT/split_report.json"
cp "$SPLIT_DIR/chang_e_label_ontology.json" "$RUN_ROOT/chang_e_label_ontology.json"

AUDIO_DIRS=()
for d in test_music_bank custom_music data/music proxy_music "$CHANGE_DIR/music" "$CHANGE_DIR/audio"; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done
if [[ ${#AUDIO_DIRS[@]} -eq 0 ]]; then
  echo "[V46.31 ERROR] no audio dirs found and motion-proxy fallback is disabled." >&2
  exit 2
fi

SEMANTIC_DIRS=()
for d in music_semantics external_music_semantics output/music_semantics; do
  [[ -e "$d" ]] && SEMANTIC_DIRS+=("$d")
done
MUSIC_SEMANTIC_ARGS=()
if [[ ${#SEMANTIC_DIRS[@]} -gt 0 ]]; then
  MUSIC_SEMANTIC_ARGS=(--music_semantic_dirs "${SEMANTIC_DIRS[@]}")
fi

AUDIO="${V46_TARGET_AUDIO:-}"
if [[ -z "$AUDIO" ]]; then
  AUDIO="test_music_bank/dunhuangwu2.wav"
  [[ -f "$AUDIO" ]] || AUDIO="data/music/dunhuangwu2.wav"
  [[ -f "$AUDIO" ]] || AUDIO="custom_music/dunhuangwu2.wav"
  [[ -f "$AUDIO" ]] || AUDIO="$CHANGE_DIR/music/dunhuangwu2.wav"
  [[ -f "$AUDIO" ]] || AUDIO="$CHANGE_DIR/audio/dunhuangwu2.wav"
fi
if [[ ! -f "$AUDIO" ]]; then
  echo "[V46.31 ERROR] target audio not found: $AUDIO" >&2
  exit 2
fi

if [[ ! -f "music_semantics/$(basename "${AUDIO%.*}").music_semantic.json" && "${V46_GENERATE_PROXY_SEMANTIC:-0}" == "1" ]]; then
  mkdir -p music_semantics
  python tools/v46_classical_music_semantic_proxy.py \
    --audio "$AUDIO" \
    --out_json "music_semantics/$(basename "${AUDIO%.*}").music_semantic.json"
fi

build_one_db() {
  local split="$1"
  local manifest="$SPLIT_DIR/${split}_manifest.csv"
  local out_db="$RUN_ROOT/${split}_db"
  echo "[V46.31] build ${split}_db from $manifest secondary_split=$V46_MANIFEST_SECONDARY_EVENT_SPLIT win=$V46_WINDOW_LEN hop=$V46_HOP_LEN"
  python tools/v46_motionrag_diff.py --config "$CFG" build-db \
    --motion_dirs "$CHANGE_DIR" \
    --manifest "$manifest" \
    --audio_dirs "${AUDIO_DIRS[@]}" \
    --out_db "$out_db"
  python tools/v46_quick_db_audit.py --db "$out_db/events.npz" --json "$RUN_ROOT/${split}_db_audit.json" --strict
}

build_one_db train
build_one_db val
build_one_db test

python - <<'PY'
import json, os, sys
import numpy as np
run=os.environ['RUN_ROOT'] if 'RUN_ROOT' in os.environ else ''
train_db=os.path.join(run,'train_db','events.npz')
min_events=int(os.environ.get('V46_MIN_TRAIN_EVENTS','24'))
db=np.load(train_db, allow_pickle=True)
n=len(db['paths']) if 'paths' in db.files else 0
print(json.dumps({'train_db_events':n,'min_required':min_events}, indent=2))
if n < min_events:
    raise SystemExit(f'[V46.31 ERROR] train_db has only {n} events; enable secondary slicing or lower V46_MIN_TRAIN_EVENTS only for smoke tests.')
PY

if [[ "${V46_BUILD_ALL_CHANGE_DEMO_DB:-0}" == "1" ]]; then
  echo "[V46.31 WARN] Building all-change DB for qualitative demo / upper-bound only. Do NOT use in main table."
  python tools/v46_motionrag_diff.py --config "$CFG" build-db \
    --motion_dirs "$CHANGE_DIR" \
    --manifest "$SPLIT_DIR/all_manifest.csv" \
    --audio_dirs "${AUDIO_DIRS[@]}" \
    --out_db "$RUN_ROOT/all_change_demo_db"
  python tools/v46_quick_db_audit.py --db "$RUN_ROOT/all_change_demo_db/events.npz" --json "$RUN_ROOT/all_change_demo_db_audit.json" --strict
fi

TRAIN_DB="$RUN_ROOT/train_db/events.npz"
CONTRASTIVE_ARG=()
REFINER_ARG=()
DIFFUSION_ARG=()

if [[ "$V46_ENABLE_CONTRASTIVE" == "1" ]]; then
  python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
    --db "$TRAIN_DB" \
    --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
    "${MUSIC_SEMANTIC_ARGS[@]}" \
    --epochs "$V46_CONTRASTIVE_EPOCHS" \
    --out "$RUN_ROOT/v44_contrastive_train_only.pt"
  CONTRASTIVE_ARG=(--contrastive "$RUN_ROOT/v44_contrastive_train_only.pt")
else
  echo "[V46.31 INFO] V44 contrastive disabled. Stage-1 motion_mg uses descriptor retrieval fallback."
fi

# Stage 1: retrieval + motion graph baseline, no refiner/diffusion.
MG_OUT="$RUN_ROOT/dunhuangwu2_v46_31_motion_mg.npy"
MG_JSON="$RUN_ROOT/dunhuangwu2_v46_31_motion_mg.report.json"
MG_MP4="$RUN_ROOT/dunhuangwu2_v46_31_motion_mg.mp4"
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  "${CONTRASTIVE_ARG[@]}" \
  --out "$MG_OUT" \
  --json "$MG_JSON" \
  --render_output "$MG_MP4"
python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$MG_OUT" --json "$RUN_ROOT/motion_mg_audit.json"

if [[ "$V46_ENABLE_REFINER" == "1" ]]; then
  python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
    --db "$TRAIN_DB" \
    --steps "$V46_REFINER_TRAIN_STEPS" \
    --out "$RUN_ROOT/v45_refiner_train_only.pt"
  REFINER_ARG=(--refiner "$RUN_ROOT/v45_refiner_train_only.pt")
else
  echo "[V46.31 INFO] V45 refiner disabled by V46_ENABLE_REFINER=0"
fi

if [[ "$V46_ENABLE_DIFFUSION" == "1" ]]; then
  python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
    --db "$TRAIN_DB" \
    --steps "$V46_DIFFUSION_TRAIN_STEPS" \
    --out "$RUN_ROOT/v46_diffusion_train_only.pt"
  DIFFUSION_ARG=(--diffusion "$RUN_ROOT/v46_diffusion_train_only.pt")
else
  echo "[V46.31 INFO] V46 diffusion disabled by V46_ENABLE_DIFFUSION=0"
fi

# Stage 2: diffusion/refiner/IK optimized final motion_diff.
DIFF_OUT="$RUN_ROOT/dunhuangwu2_v46_31_motion_diff.npy"
DIFF_JSON="$RUN_ROOT/dunhuangwu2_v46_31_motion_diff.report.json"
DIFF_MP4="$RUN_ROOT/dunhuangwu2_v46_31_motion_diff.mp4"
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  "${CONTRASTIVE_ARG[@]}" \
  "${REFINER_ARG[@]}" \
  "${DIFFUSION_ARG[@]}" \
  --out "$DIFF_OUT" \
  --json "$DIFF_JSON" \
  --render_output "$DIFF_MP4"
python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$DIFF_OUT" --json "$RUN_ROOT/motion_diff_audit.json"

cat > "$RUN_ROOT/OFFICIAL_SPLIT_POLICY.txt" <<EOF2
Official Chang-E MotionRAG-Diff policy for paper experiments
===========================================================
1. Source-level train/val/test split is applied before event slicing.
2. Long Chang-E BVH sequences are sliced only inside their own split. No event from val/test enters train_db.
3. train_db, val_db, and test_db are built separately.
4. Stage 1 motion_mg uses contrastive retrieval + Chang-E event-family/stage/music-role routing + motion graph over train_db only.
5. Stage 2 motion_diff uses train-only refiner/diffusion/IK to optimize motion_mg.
6. val_db/test_db are evaluation-only and never passed to --db for official generation.
7. all_change_demo_db, if built, is qualitative_demo/upper_bound only and must not appear in the main quantitative table.
8. Internal Dunhuang data should be used only as supplement/cross-domain validation unless explicitly reported as a separate setting.
EOF2

cat > "$RUN_ROOT/MOTIONRAG_DIFF_STAGE_SUMMARY.json" <<EOF2
{
  "version": "V46.31",
  "stage1_motion_mg": "$MG_OUT",
  "stage1_report": "$MG_JSON",
  "stage2_motion_diff": "$DIFF_OUT",
  "stage2_report": "$DIFF_JSON",
  "train_db": "$TRAIN_DB",
  "secondary_event_split": "$V46_MANIFEST_SECONDARY_EVENT_SPLIT",
  "window_len": "$V46_WINDOW_LEN",
  "hop_len": "$V46_HOP_LEN",
  "overlap": "$V46_OVERLAP"
}
EOF2

echo "[V46.31 OFFICIAL DONE]"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[TRAIN_DB_AUDIT] $RUN_ROOT/train_db_audit.json"
echo "[MOTION_MG] $MG_OUT"
echo "[MOTION_MG_REPORT] $MG_JSON"
echo "[MOTION_DIFF] $DIFF_OUT"
echo "[MOTION_DIFF_REPORT] $DIFF_JSON"
echo "[MOTION_MG_AUDIT] $RUN_ROOT/motion_mg_audit.json"
echo "[MOTION_DIFF_AUDIT] $RUN_ROOT/motion_diff_audit.json"
[[ -f "$MG_MP4" ]] && echo "[MOTION_MG_MP4] $MG_MP4"
[[ -f "$DIFF_MP4" ]] && echo "[MOTION_DIFF_MP4] $DIFF_MP4"
