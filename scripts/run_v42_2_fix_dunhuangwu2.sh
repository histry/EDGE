#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
EDGE_ROOT="$(pwd)"
export PYTHONPATH="$EDGE_ROOT:${PYTHONPATH:-}"

if [[ ! -f output/LATEST_FINAL_DUNHUANG_ACCEPTED.txt ]]; then
  echo "[ERROR] output/LATEST_FINAL_DUNHUANG_ACCEPTED.txt not found" >&2
  exit 2
fi
FINAL_DIR="$(cat output/LATEST_FINAL_DUNHUANG_ACCEPTED.txt)"
mkdir -p "$FINAL_DIR"

INPUT="$FINAL_DIR/dunhuangwu2_v40_ACCEPTED_ref.npy"
if [[ ! -f "$INPUT" ]]; then
  INPUT="output/v38_source_aware_full_train_20260625_212818/v34_contact_inr/dunhuangwu2_v40_ACCEPTED_ref.npy"
fi
if [[ ! -f "$INPUT" ]]; then
  echo "[ERROR] input motion not found: $INPUT" >&2
  exit 2
fi

AUDIO="test_music_bank/dunhuangwu2.wav"
if [[ ! -f "$AUDIO" ]]; then
  AUDIO="data/music/dunhuangwu2.wav"
fi

OUT="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.npy"
JSON_OUT="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_physics.json"
TARGETS="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.v42_2_targets.npz"
MP4="$FINAL_DIR/dunhuangwu2_v42_2_ROOT_FOOTPLANT_ref.mp4"

CFG="configs/v42_2_physics_config.json"

echo "[V42.2 FINAL] input:   $INPUT"
echo "[V42.2 FINAL] output:  $OUT"
echo "[V42.2 FINAL] json:    $JSON_OUT"
echo "[V42.2 FINAL] targets: $TARGETS"
echo "[V42.2 FINAL] audio:   $AUDIO"

CMD=(python tools/v42_root_footplant_physics_optimizer.py
  --input "$INPUT"
  --output "$OUT"
  --json "$JSON_OUT"
  --targets "$TARGETS"
  --config "$CFG")

if [[ -f "$AUDIO" && -f render_from_npy.py ]]; then
  CMD+=(--audio "$AUDIO" --render_output "$MP4" --render_script render_from_npy.py --camera_mode follow --render_smooth_window 5)
else
  echo "[V42.2 WARN] audio or render_from_npy.py missing; will only write npy/json/npz"
fi

"${CMD[@]}"

echo "[V42.2 FINAL DONE]"
echo "[MOTION]  $OUT"
echo "[JSON]    $JSON_OUT"
echo "[TARGETS] $TARGETS"
if [[ -f "$MP4" ]]; then
  echo "[MP4]     $MP4"
fi
