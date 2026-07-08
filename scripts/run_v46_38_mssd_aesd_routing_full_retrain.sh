#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export V46_DEVICE="${V46_DEVICE:-cuda}"

# MSSD + AESD + routing switches.
export V46_MSSD_ENABLE=1
export V46_AESD_ENABLE=1
export V46_ROUTING_AWARE_ENABLE=1
export V46_MSSD_REQUIRE_FINAL_SCHEDULE_FOR_GENERATE=1
# Training uses weak descriptors and unpaired audio; generation is strict.
export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0
export V46_34_ALLOW_SEMANTIC_FALLBACK=0

# V44 MSSD-AESD Semantic OT settings.
export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=0
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE=0
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY=0
export V46_38_OT_SEMANTIC_WEIGHT="${V46_38_OT_SEMANTIC_WEIGHT:-1.25}"

# Routing weights. Increase MSSD-AESD semantic score but keep contrastive active.
export V46_38_ROUTE_CONTRASTIVE_WEIGHT="${V46_38_ROUTE_CONTRASTIVE_WEIGHT:-1.00}"
export V46_38_ROUTE_MSSD_AESD_WEIGHT="${V46_38_ROUTE_MSSD_AESD_WEIGHT:-1.15}"
export V46_38_ROUTE_LEGACY_SEM_WEIGHT="${V46_38_ROUTE_LEGACY_SEM_WEIGHT:-0.55}"
export V46_38_ROUTE_DURATION_WEIGHT="${V46_38_ROUTE_DURATION_WEIGHT:-0.28}"
export V46_38_ROUTE_QUALITY_WEIGHT="${V46_38_ROUTE_QUALITY_WEIGHT:-0.26}"
export V46_38_ROUTE_STAGE_WEIGHT="${V46_38_ROUTE_STAGE_WEIGHT:-0.18}"
export V46_38_ROUTE_BOUNDARY_RISK_WEIGHT="${V46_38_ROUTE_BOUNDARY_RISK_WEIGHT:-0.35}"
export V46_38_ROUTE_CANDIDATE_TOPK="${V46_38_ROUTE_CANDIDATE_TOPK:-128}"

# Model settings.
export V46_ENABLE_REFINER=1
export V46_ENABLE_DIFFUSION=1
export V46_ENABLE_TRUE_IK=1
export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-180}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-10000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-20000}"
export V46_DIFFUSION_STEPS="${V46_DIFFUSION_STEPS:-50}"

# Transition-mask policy.
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

AUDIO="${AUDIO:-test_music_bank/dunhuangwu2.wav}"
AUDIO_STEM="$(basename "${AUDIO%.*}")"
CFG="${CFG:-configs/v46_motionrag_diff_config.json}"
RUN_ROOT="${RUN_ROOT:-output/v46_38_mssd_aesd_routing_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/logs"

echo "[V46.38 RUN_ROOT] $RUN_ROOT"
echo "[V46.38 AUDIO] $AUDIO"

ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
PLANNER_CKPT="${V26_PLANNER_CKPT:-output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt}"
V23_CKPT="${V26_V23_CKPT:-./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt}"
INDEX_JSON="${V26_INDEX_JSON:-data/v35_source_aware/v21_shared_event_index_source_aware.json}"
DURATION_NPZ="${V26_DURATION_INDEX_NPZ:-data/v35_source_aware/v26_music_dominant_duration_index_source_aware.npz}"
HIER_NPZ="${V34_HIERARCHY_INDEX_NPZ:-data/v35_source_aware/v34_hierarchical_event_index_source_aware.npz}"

for f in "$ROUTER_CKPT" "$PLANNER_CKPT" "$V23_CKPT" "$INDEX_JSON" "$DURATION_NPZ"; do
  [[ -e "$f" ]] || { echo "[ERROR] missing required file: $f"; exit 2; }
done

echo "[1/10] Apply V46.38 complete MSSD-AESD routing patch"
python tools/apply_v46_38_complete_routing_patch.py
python -m py_compile \
  tools/v46_38_music_action_descriptor.py \
  tools/v46_38_build_music_semantic_slot_descriptor.py \
  tools/v46_38_build_aesd_event_semantics.py \
  tools/apply_v46_38_complete_routing_patch.py \
  tools/v46_motionrag_diff.py

echo "[2/10] Rebuild all-train Event-RAG DB"
RAW_DB_DIR="$RUN_ROOT/train_all_db"
python tools/v46_motionrag_diff.py --config "$CFG" build-db \
  --motion_dirs change \
  --out_db "$RAW_DB_DIR"
RAW_DB="$RAW_DB_DIR/events.npz"

echo "[3/10] Build full AESD action-event semantic DB"
AESD_DB="$RAW_DB_DIR/events_aesd.npz"
python tools/v46_38_build_aesd_event_semantics.py \
  --db "$RAW_DB" \
  --out "$AESD_DB" \
  --json "$RUN_ROOT/V46_38_AESD_AUDIT.json"

echo "[4/10] Build strict final V21/V26/V23 MSSD"
MSSD_JSON="$RUN_ROOT/${AUDIO_STEM}_v46_38_mssd.json"
python tools/v46_38_build_music_semantic_slot_descriptor.py \
  --audio "$AUDIO" \
  --out_json "$MSSD_JSON" \
  --router_ckpt "$ROUTER_CKPT" \
  --planner_ckpt "$PLANNER_CKPT" \
  --v23_ckpt "$V23_CKPT" \
  --index_json "$INDEX_JSON" \
  --duration_index_npz "$DURATION_NPZ" \
  --hierarchy_index_npz "$HIER_NPZ" \
  --feature_dir "$RUN_ROOT/v26_music_features" \
  --schedule_dir "$RUN_ROOT/v21_v26_schedule_raw"

python - <<PY
import json
p="$MSSD_JSON"
d=json.load(open(p, encoding="utf-8"))
assert d.get("descriptor_type") == "music_semantic_slot_descriptor"
assert d.get("usage") == "generate_schedule"
assert d.get("is_final_schedule") is True
assert "v21" in (d.get("slot_source", "") + d.get("router_ckpt", "")).lower() or "router" in d.get("slot_source", "").lower()
assert d.get("raw_schedule_json")
print("[MSSD CHECK]", d.get("slot_source"), d.get("num_slots"), d.get("total_target_frames"), d.get("raw_schedule_json"))
PY

AUDIO_DIRS=()
for d in test_music_bank custom_music proxy_music data/v21_router_music_999/splits/train data/v21_router_music_999/splits/val data/v21_router_music_valid25 data/v21_router_music_train_pcm16; do
  [[ -e "$d" ]] && AUDIO_DIRS+=("$d")
done
SEMANTIC_DIRS=()
for d in music_semantics external_music_semantics output/music_semantics "$RUN_ROOT"; do
  [[ -e "$d" ]] && SEMANTIC_DIRS+=("$d")
done
export V46_MSSD_DESCRIPTOR_DIRS="$(IFS=:; echo "${SEMANTIC_DIRS[*]}")"

CONTRASTIVE="$RUN_ROOT/v44_mssd_aesd_routing.pt"
REFINER="$RUN_ROOT/v45_aesd_refiner.pt"
DIFFUSION="$RUN_ROOT/v46_aesd_diffusion.pt"

echo "[5/10] Train V44 with MSSD-AESD Semantic OT"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0 python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$AESD_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --epochs "$V46_CONTRASTIVE_EPOCHS" \
  --out "$CONTRASTIVE"

echo "[6/10] Train V45 transition refiner on AESD DB"
python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
  --db "$AESD_DB" \
  --steps "$V46_REFINER_TRAIN_STEPS" \
  --out "$REFINER"

echo "[7/10] Train V46 masked diffusion on AESD DB"
python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
  --db "$AESD_DB" \
  --steps "$V46_DIFFUSION_TRAIN_STEPS" \
  --diffusion_steps "$V46_DIFFUSION_STEPS" \
  --out "$DIFFUSION"

echo "[8/10] Generate ablations and final strict-MSSD V46.38 outputs"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=0 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_38_motionrag.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_38_motionrag.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_38_motionrag.mp4"

V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" --refiner "$REFINER" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_38_refiner_ik.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_38_refiner_ik.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_38_refiner_ik.mp4"

V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=1 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" --refiner "$REFINER" --diffusion "$DIFFUSION" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_38_diffusion_ik.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_38_diffusion_ik.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_38_diffusion_ik.mp4"

echo "[9/10] Verify final report actually used V46.38 routing"
python tools/v46_38_verify_routing_report.py \
  --report "$RUN_ROOT/${AUDIO_STEM}_v46_38_diffusion_ik.report.json" \
  --mssd "$MSSD_JSON" | tee "$RUN_ROOT/V46_38_ROUTING_VERIFY.json"

echo "[10/10] Summarize"
python - <<PY
import json, os
run="$RUN_ROOT"
stem="$AUDIO_STEM"
mssd=json.load(open("$MSSD_JSON", encoding="utf-8"))
aesd=json.load(open(os.path.join(run,"V46_38_AESD_AUDIT.json"), encoding="utf-8"))
summary={
 "version":"V46.38_complete_mssd_aesd_routing",
 "mssd":{k:mssd.get(k) for k in ["usage","is_final_schedule","slot_source","num_slots","total_target_frames","router_ckpt","planner_ckpt","v23_ckpt","raw_schedule_json"]},
 "aesd":{"schema":aesd.get("schema"),"num_events":aesd.get("num_events"),"event_semantic_histogram":aesd.get("event_semantic_histogram"),"boundary_risk_histogram":aesd.get("boundary_risk_histogram")},
 "outputs":{}
}
for suffix in ["motionrag","refiner_ik","diffusion_ik"]:
    p=os.path.join(run,f"{stem}_v46_38_{suffix}.report.json")
    if os.path.exists(p):
        r=json.load(open(p, encoding="utf-8"))
        ret=r.get("stage_reports",{}).get("retrieval",[])
        summary["outputs"][suffix]={"report":p,"selected_events":len(r.get("selected_event_indices",[])),"routing_policy":ret[0].get("routing_policy") if ret else "", "final_audit":r.get("final_audit",{})}
out=os.path.join(run,"V46_38_COMPLETE_FINAL_SUMMARY.json")
json.dump(summary, open(out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("[SUMMARY]", out)
PY

echo "[DONE] V46.38 complete MSSD-AESD routing run finished: $RUN_ROOT"
