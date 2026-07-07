#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export V46_DEVICE="${V46_DEVICE:-cuda}"

export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0  # training stages use unpaired audio corpus; generate stages re-enable strict slots
export V46_34_ALLOW_SEMANTIC_FALLBACK=0

export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=0
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE=0
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY=0

export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-160}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-10000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-20000}"
export V46_DIFFUSION_STEPS="${V46_DIFFUSION_STEPS:-50}"

export V46_TRANSITION_BUDGET_ENABLE=1
export V46_TRANSITION_INBETWEEN_ENABLE=1
export V46_TRANSITION_MIN_FRAMES=10
export V46_TRANSITION_MAX_FRAMES=28
export V46_TRANSITION_RATIO=0.18
export V46_TRANSITION_MASK_HALO=6
export V46_TRANSITION_MIN_CORE_FRAMES=30

export V46_CORE_WARP_MIN=0.72
export V46_CORE_WARP_MAX=1.38
export V46_REFINER_CORE_STRENGTH=0.02
export V46_REFINER_TRANSITION_STRENGTH=1.00
export V46_DIFFUSION_CORE_STRENGTH=0.00
export V46_DIFFUSION_TRANSITION_STRENGTH=0.72
export V46_DIFFUSION_REFERENCE_NOISE_SCALE=0.03

RUN_ROOT="output/v46_34_pretrained_router_reference_transition_20260706_223408"
TRAIN_DB="$RUN_ROOT/train_all_db/events.npz"
SLOT_JSON="$RUN_ROOT/dunhuangwu2_v46_34_pretrained_router_slots.json"
AUDIO="test_music_bank/dunhuangwu2.wav"
CFG="configs/v46_motionrag_diff_config.json"

CONTRASTIVE="$RUN_ROOT/v44_contrastive_v46_34.pt"
REFINER="$RUN_ROOT/v45_refiner_v46_34.pt"
DIFFUSION="$RUN_ROOT/v46_diffusion_v46_34.pt"

mkdir -p "$RUN_ROOT/logs"

[[ -f "$TRAIN_DB" ]] || { echo "[ERROR] missing TRAIN_DB: $TRAIN_DB"; exit 2; }
[[ -f "$SLOT_JSON" ]] || { echo "[ERROR] missing SLOT_JSON: $SLOT_JSON"; exit 2; }

AUDIO_DIRS=()
for d in test_music_bank custom_music proxy_music data/v21_router_music_999/splits/train data/v21_router_music_999/splits/val data/v21_router_music_valid25 data/v21_router_music_train_pcm16; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done

SEMANTIC_ARGS=()
for d in music_semantics external_music_semantics output/music_semantics; do
  [[ -e "$d" ]] && SEMANTIC_ARGS+=("$d")
done

echo "[CHECK] RUN_ROOT=$RUN_ROOT"
echo "[CHECK] TRAIN_DB=$TRAIN_DB"
echo "[CHECK] SLOT_JSON=$SLOT_JSON"
echo "[CHECK] strict pretrained slot source:"
python - <<PY
import json
p="$SLOT_JSON"
d=json.load(open(p, encoding="utf-8"))
print("slot_source:", d.get("slot_source"))
print("num_slots:", d.get("num_slots"))
print("total_target_frames:", d.get("total_target_frames"))
print("router_ckpt:", d.get("router_ckpt"))
print("planner_ckpt:", d.get("planner_ckpt"))
print("v23_ckpt:", d.get("v23_ckpt"))
assert d.get("slot_source") == "v21_router_v26_planner"
assert int(d.get("num_slots", 0)) > 0
PY

echo "[5/10 RESUME] Train V44 contrastive alignment"
python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$TRAIN_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  --music_semantic_dirs "${SEMANTIC_ARGS[@]}" \
  --epochs "$V46_CONTRASTIVE_EPOCHS" \
  --out "$CONTRASTIVE"

echo "[6/10 RESUME] Train V45 reference-conditioned transition refiner"
python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
  --db "$TRAIN_DB" \
  --steps "$V46_REFINER_TRAIN_STEPS" \
  --out "$REFINER"

echo "[7/10 RESUME] Train V46 reference-conditioned masked diffusion"
python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
  --db "$TRAIN_DB" \
  --steps "$V46_DIFFUSION_TRAIN_STEPS" \
  --diffusion_steps "$V46_DIFFUSION_STEPS" \
  --out "$DIFFUSION"

echo "[8/10 RESUME] Generate Stage-1 router-slot MotionRAG baseline"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_34_ALLOW_SEMANTIC_FALLBACK=0 V46_ENABLE_REFINER=0 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --slots_json "$SLOT_JSON" \
  --music_semantic_dirs "${SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  --contrastive "$CONTRASTIVE" \
  --out "$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.mp4"

echo "[9/10 RESUME] Generate V45 refiner + IK"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_34_ALLOW_SEMANTIC_FALLBACK=0 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --slots_json "$SLOT_JSON" \
  --music_semantic_dirs "${SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  --contrastive "$CONTRASTIVE" \
  --refiner "$REFINER" \
  --out "$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.mp4"

echo "[10/10 RESUME] Generate V46 diffusion + IK"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_34_ALLOW_SEMANTIC_FALLBACK=0 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=1 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" \
  --slots_json "$SLOT_JSON" \
  --music_semantic_dirs "${SEMANTIC_ARGS[@]}" \
  --db "$TRAIN_DB" \
  --contrastive "$CONTRASTIVE" \
  --refiner "$REFINER" \
  --diffusion "$DIFFUSION" \
  --out "$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.mp4"

python - <<PY
import json, os
run = "$RUN_ROOT"
slot_json = "$SLOT_JSON"
slot_plan = json.load(open(slot_json, encoding="utf-8"))
summary = {
    "slot_plan": {
        "slot_source": slot_plan.get("slot_source"),
        "num_slots": slot_plan.get("num_slots"),
        "total_target_frames": slot_plan.get("total_target_frames"),
        "router_ckpt": slot_plan.get("router_ckpt"),
        "planner_ckpt": slot_plan.get("planner_ckpt"),
        "v23_ckpt": slot_plan.get("v23_ckpt"),
        "raw_schedule_json": slot_plan.get("raw_schedule_json"),
    },
    "outputs": {}
}
for name in [
  "dunhuangwu2_v46_34_router_motion_mg.report.json",
  "dunhuangwu2_v46_34_router_refiner_ik.report.json",
  "dunhuangwu2_v46_34_router_diffusion_ik.report.json",
]:
    p = os.path.join(run, name)
    if os.path.exists(p):
        rep = json.load(open(p, encoding="utf-8"))
        summary["outputs"][name] = {
          "selected_events": len(rep.get("selected_event_indices", [])),
          "slot_source": rep.get("slot_source", ""),
          "final_audit": rep.get("final_audit", {}),
          "stage_reports_keys": list(rep.get("stage_reports", {}).keys()),
        }
out = os.path.join(run, "V46_34_FINAL_SUMMARY.json")
json.dump(summary, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("[SUMMARY]", out)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "[DONE] V46.34 resumed run complete."
echo "[RUN_ROOT] $RUN_ROOT"
echo "[MOTION_MG] $RUN_ROOT/dunhuangwu2_v46_34_router_motion_mg.mp4"
echo "[REFINER_IK] $RUN_ROOT/dunhuangwu2_v46_34_router_refiner_ik.mp4"
echo "[DIFFUSION_IK] $RUN_ROOT/dunhuangwu2_v46_34_router_diffusion_ik.mp4"
