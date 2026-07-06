#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

# -----------------------------------------------------------------------------
# V46 official Chang-E split experiment
# -----------------------------------------------------------------------------
# Scientific policy:
#   * Source-level train/val/test split is created before event slicing.
#   * train_db is the only Event-RAG memory used for training and generation.
#   * val_db/test_db are built and audited for evaluation/analysis only.
#   * all-change DB can be built only as qualitative_demo/upper_bound.
# -----------------------------------------------------------------------------

# Patch V46.21 contract guards into tools/v46_motionrag_diff.py.
python tools/v46_research_contract_patch.py

export V46_DEVICE="${V46_DEVICE:-cuda}"
export V46_BVH_RESAMPLE_TO_CONFIG_FPS="${V46_BVH_RESAMPLE_TO_CONFIG_FPS:-1}"
export V46_SOURCE_GROUP_MODE="${V46_SOURCE_GROUP_MODE:-filename}"
export V46_FILENAME_SEMANTIC_ENABLE="${V46_FILENAME_SEMANTIC_ENABLE:-1}"
export V46_CLASSIFICATION_SEMANTIC_ENABLE="${V46_CLASSIFICATION_SEMANTIC_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE:-1}"
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED="${V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_DIRS="${V46_EXTERNAL_MUSIC_SEMANTIC_DIRS:-music_semantics:external_music_semantics:output/music_semantics}"
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE="${V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE:-0}"
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY="${V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY:-0}"
export V46_UNPAIRED_AUDIO_ENABLE="${V46_UNPAIRED_AUDIO_ENABLE:-1}"
export V46_UNPAIRED_DISABLE_MOTION_PROXY="${V46_UNPAIRED_DISABLE_MOTION_PROXY:-1}"
export V46_OVERLAP="${V46_OVERLAP:-6}"
export V46_WINDOW_LEN="${V46_WINDOW_LEN:-120}"
export V46_HOP_LEN="${V46_HOP_LEN:-60}"
export V46_MIN_EVENT_FRAMES="${V46_MIN_EVENT_FRAMES:-36}"
export V46_MAX_EVENT_FRAMES="${V46_MAX_EVENT_FRAMES:-180}"
# For official manifest records, default to preserving upstream semantic clips.
export V46_MANIFEST_SECONDARY_EVENT_SPLIT="${V46_MANIFEST_SECONDARY_EVENT_SPLIT:-0}"
export V46_ENABLE_TRUE_IK="${V46_ENABLE_TRUE_IK:-1}"
export V46_ENABLE_ROOT_Y_PHYSICS="${V46_ENABLE_ROOT_Y_PHYSICS:-1}"
export V46_ROOT_Y_MAX_FLIGHT_SECONDS="${V46_ROOT_Y_MAX_FLIGHT_SECONDS:-1.2}"
export V46_ROOT_Y_DAMPING_MAX_SECONDS="${V46_ROOT_Y_DAMPING_MAX_SECONDS:-0.28}"

# Training gates.  Contrastive is useful for the official method; refiner/diffusion
# stay opt-in because small source splits can overfit.
export V46_ENABLE_CONTRASTIVE="${V46_ENABLE_CONTRASTIVE:-1}"
export V46_ENABLE_REFINER="${V46_ENABLE_REFINER:-0}"
export V46_ENABLE_DIFFUSION="${V46_ENABLE_DIFFUSION:-0}"
export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-120}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-2000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-3000}"

CFG="${V46_CONFIG:-configs/v46_motionrag_diff_config.json}"
CHANGE_DIR="${V46_CHANGE_DIR:-change}"
SPLIT_DIR="${V46_SPLIT_DIR:-$CHANGE_DIR/splits_official}"
RUN_ROOT="${V46_RUN_ROOT:-output/v46_21_chang_e_official_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT" output
printf '%s\n' "$RUN_ROOT" > output/LATEST_V46_CHANG_E_OFFICIAL.txt

if [[ ! -d "$CHANGE_DIR" ]]; then
  echo "[V46.21 ERROR] $CHANGE_DIR not found. Put Chang-E BVH/manifest under EDGE/$CHANGE_DIR." >&2
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
  echo "[V46.21 ERROR] no audio dirs found and motion-proxy fallback is disabled." >&2
  echo "Put music under test_music_bank/, custom_music/, data/music/, proxy_music/, $CHANGE_DIR/music/, or $CHANGE_DIR/audio/." >&2
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
  echo "[V46.21 ERROR] target audio not found: $AUDIO" >&2
  exit 2
fi

# Debug-only fallback sidecar; default OFF for paper experiments.
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
  echo "[V46.21] build ${split}_db from $manifest"
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

if [[ "${V46_BUILD_ALL_CHANGE_DEMO_DB:-0}" == "1" ]]; then
  echo "[V46.21 WARN] Building all-change DB for qualitative demo / upper-bound only. Do NOT use in main table."
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
  echo "[V46.21 INFO] V44 contrastive disabled. Generation will use descriptor retrieval fallback."
fi

if [[ "$V46_ENABLE_REFINER" == "1" ]]; then
  python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
    --db "$TRAIN_DB" \
    --steps "$V46_REFINER_TRAIN_STEPS" \
    --out "$RUN_ROOT/v45_refiner_train_only.pt"
  REFINER_ARG=(--refiner "$RUN_ROOT/v45_refiner_train_only.pt")
else
  echo "[V46.21 INFO] V45 refiner disabled by V46_ENABLE_REFINER=0"
fi

if [[ "$V46_ENABLE_DIFFUSION" == "1" ]]; then
  python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
    --db "$TRAIN_DB" \
    --steps "$V46_DIFFUSION_TRAIN_STEPS" \
    --out "$RUN_ROOT/v46_diffusion_train_only.pt"
  DIFFUSION_ARG=(--diffusion "$RUN_ROOT/v46_diffusion_train_only.pt")
else
  echo "[V46.21 INFO] V46 diffusion disabled by V46_ENABLE_DIFFUSION=0"
fi

OUT="$RUN_ROOT/dunhuangwu2_v46_21_official_train_db_only.npy"
JSON="$RUN_ROOT/dunhuangwu2_v46_21_official_train_db_only.report.json"
MP4="$RUN_ROOT/dunhuangwu2_v46_21_official_train_db_only.mp4"

# Official rule: generation/RAG memory uses train_db only.
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  "${MUSIC_SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  "${CONTRASTIVE_ARG[@]}" \
  "${REFINER_ARG[@]}" \
  "${DIFFUSION_ARG[@]}" \
  --out "$OUT" \
  --json "$JSON" \
  --render_output "$MP4"

python tools/v46_motionrag_diff.py --config "$CFG" audit --input "$OUT" --json "$RUN_ROOT/final_audit.json"

cat > "$RUN_ROOT/OFFICIAL_SPLIT_POLICY.txt" <<EOF
Official Chang-E Event-RAG policy for paper experiments
======================================================
1. Source-level train/val/test split is applied before event slicing.
2. train_db, val_db, and test_db are built separately.
3. Training and generation use train_db only as the RAG memory.
4. val_db/test_db are evaluation-only and never passed to --db for official generation.
5. all_change_demo_db, if built, is qualitative_demo/upper_bound only and must not appear in the main quantitative table.
6. Internal Dunhuang data should be used only as supplement/cross-domain validation unless explicitly reported as a separate setting.
7. If FK is unavailable in a metadata-polluted clip, contact fallback intentionally stays below ik_contact_high so strong IK foot-lock is softly suspended rather than forcing incorrect double-foot anchoring.
EOF

echo "[V46.21 OFFICIAL DONE]"
echo "[RUN_ROOT] $RUN_ROOT"
echo "[SPLIT_REPORT] $RUN_ROOT/split_report.json"
echo "[TRAIN_DB_AUDIT] $RUN_ROOT/train_db_audit.json"
echo "[VAL_DB_AUDIT] $RUN_ROOT/val_db_audit.json"
echo "[TEST_DB_AUDIT] $RUN_ROOT/test_db_audit.json"
echo "[MOTION] $OUT"
echo "[REPORT] $JSON"
echo "[FINAL_AUDIT] $RUN_ROOT/final_audit.json"
[[ -f "$MP4" ]] && echo "[MP4] $MP4"
