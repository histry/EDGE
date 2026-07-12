#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT_DIR"
source configs/v46_51_fresh_wav_schedule.env

PY="$V46_51_PYTHON"
[[ -x "$PY" ]] || {
  echo "[FATAL] Python not executable: $PY" >&2
  exit 2
}

# Required reusable products from a completed full rebuild.
DB_AESD="${DB_AESD:-}"
V44_CKPT="${V44_CKPT:-}"
V45_CKPT="${V45_CKPT:-}"
V46_CKPT="${V46_CKPT:-}"

require_file() {
  local p="$1"
  local label="$2"
  [[ -s "$p" ]] || {
    echo "[FATAL] Missing $label: '$p'" >&2
    exit 2
  }
}

require_file "$AUDIO" "current WAV"
require_file "$DB_AESD" "heading-aware AESD database"
require_file "$V44_CKPT" "V44 contrastive checkpoint"
require_file "$V45_CKPT" "V45 refiner checkpoint"
require_file "$V46_CKPT" "V46 diffusion checkpoint"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
AUDIO_STEM="$(basename "${AUDIO%.*}")"
GEN_ROOT="${GEN_ROOT:-output/v46_51_generate_${AUDIO_STEM}_${RUN_TAG}}"
mkdir -p "$GEN_ROOT"

ASSET_JSON="$GEN_ROOT/scheduler_assets.json"
ASSET_ENV="$GEN_ROOT/scheduler_assets.env"
"$PY" tools/v46_51_resolve_scheduler_assets.py \
  --out_json "$ASSET_JSON" \
  --out_env "$ASSET_ENV"
# shellcheck disable=SC1090
source "$ASSET_ENV"

AUDIO_SHA="$(sha256sum "$AUDIO" | awk '{print $1}')"
export V46_51_SCHEDULE_RUN_ID="${RUN_TAG}_${AUDIO_SHA:0:12}"
SCHEDULE_RUN_DIR="$GEN_ROOT/fresh_schedule/$V46_51_SCHEDULE_RUN_ID"
FRESH_MSSD="$GEN_ROOT/${AUDIO_STEM}.fresh.final.mssd.json"
FINAL_NPY="$GEN_ROOT/${AUDIO_STEM}.v46_51.npy"
FINAL_REPORT="$GEN_ROOT/${AUDIO_STEM}.v46_51.report.json"
FINAL_MP4="$GEN_ROOT/${AUDIO_STEM}.v46_51.scientific_fixed.mp4"

FRESH_ARGS=(
  --audio "$AUDIO"
  --out_json "$FRESH_MSSD"
  --run_dir "$SCHEDULE_RUN_DIR"
  --run_id "$V46_51_SCHEDULE_RUN_ID"
  --router_ckpt "$V46_51_RESOLVED_ROUTER_CKPT"
  --planner_ckpt "$V46_51_RESOLVED_PLANNER_CKPT"
  --v23_ckpt "$V46_51_RESOLVED_V23_CKPT"
  --index_json "$V46_51_RESOLVED_INDEX_JSON"
  --duration_index_npz "$V46_51_RESOLVED_DURATION_INDEX_NPZ"
  --fps "$V46_51_FPS"
  --min_phrase_seconds "$V46_51_MIN_PHRASE_SECONDS"
  --max_phrase_seconds "$V46_51_MAX_PHRASE_SECONDS"
  --max_phrases "$V46_51_MAX_PHRASES"
  --boundary_quantile "$V46_51_BOUNDARY_QUANTILE"
  --beat_snap_seconds "$V46_51_BEAT_SNAP_SECONDS"
  --max_single_event_seconds "$V46_51_MAX_SINGLE_EVENT_SECONDS"
  --calm_max_single_event_seconds "$V46_51_CALM_MAX_SINGLE_EVENT_SECONDS"
  --min_subphrase_seconds "$V46_51_MIN_SUBPHRASE_SECONDS"
  --max_events_per_phrase "$V46_51_MAX_EVENTS_PER_PHRASE"
  --slot_beat_snap_seconds "$V46_51_SLOT_BEAT_SNAP_SECONDS"
  --beam_size "$V46_51_BEAM_SIZE"
  --candidate_top_k "$V46_51_CANDIDATE_TOP_K"
  --graph_node_top_k "$V46_51_GRAPH_NODE_TOP_K"
  --max_frame_error "$V46_51_MAX_FRAME_ERROR"
  --max_seconds_error "$V46_51_MAX_SECONDS_ERROR"
)

[[ -n "$V46_51_RESOLVED_HIERARCHY_INDEX_NPZ" ]] && \
  FRESH_ARGS+=(--hierarchy_index_npz "$V46_51_RESOLVED_HIERARCHY_INDEX_NPZ")
[[ -n "$V46_51_RESOLVED_START_POSE" ]] && \
  FRESH_ARGS+=(--start_pose "$V46_51_RESOLVED_START_POSE")
[[ "$V46_51_DEEP_MUSIC_FEATURES" == "1" ]] && \
  FRESH_ARGS+=(--deep_music_features)
[[ "$V46_51_REQUIRE_DEEP_MUSIC" == "1" ]] && \
  FRESH_ARGS+=(--require_deep_music)
FRESH_ARGS+=(--deep_music_model "$V46_51_DEEP_MUSIC_MODEL")
FRESH_ARGS+=(--deep_music_min_success "$V46_51_DEEP_MUSIC_MIN_SUCCESS")

if [[ "$V46_51_TRANSITION_DIFFUSION" == "1" ]]; then
  require_file "$V46_51_TRANSITION_DIFFUSION_CKPT" \
    "V26 transition diffusion checkpoint"
  FRESH_ARGS+=(
    --transition_diffusion
    --transition_diffusion_ckpt "$V46_51_TRANSITION_DIFFUSION_CKPT"
    --transition_diffusion_blend "$V46_51_TRANSITION_DIFFUSION_BLEND"
    --transition_diffusion_steps "$V46_51_TRANSITION_DIFFUSION_STEPS"
  )
fi

echo "========== V46.51 CURRENT-WAV TRANSACTION =========="
printf "AUDIO=%s\nAUDIO_SHA=%s\nRUN_ID=%s\nDB_AESD=%s\nGEN_ROOT=%s\n" \
  "$AUDIO" "$AUDIO_SHA" "$V46_51_SCHEDULE_RUN_ID" "$DB_AESD" "$GEN_ROOT"

"$PY" tools/v46_51_build_fresh_mssd.py "${FRESH_ARGS[@]}"

"$PY" tools/v46_51_heading_closed_loop.py \
  generate \
  --config "$CONFIG" \
  --audio "$AUDIO" \
  --slots_json "$FRESH_MSSD" \
  --db "$DB_AESD" \
  --contrastive "$V44_CKPT" \
  --refiner "$V45_CKPT" \
  --diffusion "$V46_CKPT" \
  --out "$FINAL_NPY" \
  --json "$FINAL_REPORT"

"$PY" tools/v46_49_audit_gravity_contract.py \
  --input "$FINAL_NPY" \
  --out "$GEN_ROOT/final.gravity.json" \
  --csv "$GEN_ROOT/final.gravity.csv"

"$PY" tools/v46_50_audit_generated_heading.py \
  --motion "$FINAL_NPY" \
  --report "$FINAL_REPORT" \
  --db "$DB_AESD" \
  --out "$GEN_ROOT/final.heading.json" \
  --csv "$GEN_ROOT/final.heading.csv"

"$PY" - "$FINAL_NPY" "$FRESH_MSSD.contract.json" <<'PY'
import json
import sys
from pathlib import Path
import numpy as np

motion = np.load(sys.argv[1], allow_pickle=True)
frames = int(motion.shape[-2])
contract = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
scheduled = int(contract["total_target_frames"])
audio_expected = int(contract["expected_audio_target_frames"])
if frames != scheduled:
    raise SystemExit(
        f"[FATAL] final motion frames={frames}, scheduled={scheduled}"
    )
print(
    f"[PASS] current-WAV frame conservation: motion={frames}, "
    f"audio_expected={audio_expected}, error={scheduled-audio_expected}"
)
PY

"$PY" render_from_npy.py \
  --motion "$FINAL_NPY" \
  --audio "$AUDIO" \
  --output "$FINAL_MP4" \
  --camera_mode fixed \
  --render_smooth_window 1 \
  --gravity_audit_json "$GEN_ROOT/final.render_gravity.json"

echo "========== V46.51 GENERATION COMPLETE =========="
printf "FRESH_MSSD=%s\nFINAL_NPY=%s\nFINAL_REPORT=%s\nFINAL_MP4=%s\n" \
  "$FRESH_MSSD" "$FINAL_NPY" "$FINAL_REPORT" "$FINAL_MP4"
