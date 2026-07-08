#!/usr/bin/env bash
set -euo pipefail

cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export V46_DEVICE="${V46_DEVICE:-cuda}"

# ---------- V46.38 routing switches ----------
export V46_MSSD_ENABLE=1
export V46_AESD_ENABLE=1
export V46_ROUTING_AWARE_ENABLE=1
export V46_MSSD_REQUIRE_FINAL_SCHEDULE_FOR_GENERATE=1
export V46_34_ALLOW_SEMANTIC_FALLBACK=0
export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0
export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=0
export V46_EXTERNAL_MUSIC_SEMANTIC_PROXY_ENABLE=0
export V46_EXTERNAL_MUSIC_SEMANTIC_FILENAME_PROXY=0
export V46_38_OT_SEMANTIC_WEIGHT="${V46_38_OT_SEMANTIC_WEIGHT:-1.25}"
export V46_38_ROUTE_CONTRASTIVE_WEIGHT="${V46_38_ROUTE_CONTRASTIVE_WEIGHT:-1.00}"
export V46_38_ROUTE_MSSD_AESD_WEIGHT="${V46_38_ROUTE_MSSD_AESD_WEIGHT:-1.15}"
export V46_38_ROUTE_LEGACY_SEM_WEIGHT="${V46_38_ROUTE_LEGACY_SEM_WEIGHT:-0.55}"
export V46_38_ROUTE_DURATION_WEIGHT="${V46_38_ROUTE_DURATION_WEIGHT:-0.28}"
export V46_38_ROUTE_QUALITY_WEIGHT="${V46_38_ROUTE_QUALITY_WEIGHT:-0.26}"
export V46_38_ROUTE_STAGE_WEIGHT="${V46_38_ROUTE_STAGE_WEIGHT:-0.18}"
export V46_38_ROUTE_BOUNDARY_RISK_WEIGHT="${V46_38_ROUTE_BOUNDARY_RISK_WEIGHT:-0.35}"
export V46_38_ROUTE_CANDIDATE_TOPK="${V46_38_ROUTE_CANDIDATE_TOPK:-128}"

# ---------- V46.41 MSA + TGT/KBO switches ----------
export V46_41_MSA_ENABLE="${V46_41_MSA_ENABLE:-1}"
export V46_41_STAGE_RADIUS_M="${V46_41_STAGE_RADIUS_M:-1.80}"
export V46_41_MSA_REFERENCE_STRENGTH="${V46_41_MSA_REFERENCE_STRENGTH:-0.10}"
export V46_41_MSA_COMMIT_STRENGTH="${V46_41_MSA_COMMIT_STRENGTH:-0.16}"
export V46_41_MSA_TRANSACTION_STRENGTH="${V46_41_MSA_TRANSACTION_STRENGTH:-0.08}"
export V46_41_MSA_STAGE_FINAL_STRENGTH="${V46_41_MSA_STAGE_FINAL_STRENGTH:-0.08}"
export V46_41_MSA_MAX_DELTA_M="${V46_41_MSA_MAX_DELTA_M:-0.06}"
export V46_41_TGT_ENABLE="${V46_41_TGT_ENABLE:-1}"
export V46_41_IK_TGT_ENABLE="${V46_41_IK_TGT_ENABLE:-1}"
export V46_41_TGT_HALO="${V46_41_TGT_HALO:-12}"
export V46_41_TGT_MIN_FRAMES="${V46_41_TGT_MIN_FRAMES:-16}"
export V46_41_TGT_MAX_FRAMES="${V46_41_TGT_MAX_FRAMES:-96}"
export V46_41_TGT_ACTIVE_THRESHOLD="${V46_41_TGT_ACTIVE_THRESHOLD:-0.05}"
export V46_41_DIFFUSION_EARLY_ABORT_ENABLE="${V46_41_DIFFUSION_EARLY_ABORT_ENABLE:-1}"
export V46_41_DIFFUSION_EARLY_ABORT_FRACTION="${V46_41_DIFFUSION_EARLY_ABORT_FRACTION:-0.50}"
export V46_41_HN_DPO_SAVE_PAIRS="${V46_41_HN_DPO_SAVE_PAIRS:-1}"

# V46.42 loophole fixes.
export V46_42_EARLY_ABORT_KBO_SMOOTH_SIGMA="${V46_42_EARLY_ABORT_KBO_SMOOTH_SIGMA:-1.35}"
export V46_42_EARLY_ABORT_KBO_RELAX="${V46_42_EARLY_ABORT_KBO_RELAX:-3.0}"
export V46_42_MSA_HIGH_ENERGY_SCALE="${V46_42_MSA_HIGH_ENERGY_SCALE:-0.22}"
export V46_42_MSA_DYNAMIC_ATTENUATION="${V46_42_MSA_DYNAMIC_ATTENUATION:-0.75}"
export V46_42_MSA_ROOT_SPEED_RELAX_THRESH="${V46_42_MSA_ROOT_SPEED_RELAX_THRESH:-0.045}"
export V46_42_MSA_MIN_WEIGHT="${V46_42_MSA_MIN_WEIGHT:-0.05}"
export V46_42_ENABLE_HN_DPO_FINETUNE="${V46_42_ENABLE_HN_DPO_FINETUNE:-${V46_41_ENABLE_HN_DPO_FINETUNE:-0}}"
export V46_42_HN_DPO_STEPS="${V46_42_HN_DPO_STEPS:-1500}"
export V46_42_HN_DPO_KINETIC_WEIGHT="${V46_42_HN_DPO_KINETIC_WEIGHT:-0.18}"
export V46_42_HN_DPO_KINETIC_FLOOR_RATIO="${V46_42_HN_DPO_KINETIC_FLOOR_RATIO:-0.80}"

# Conservative neural commit.
export V46_REFINER_CORE_STRENGTH="${V46_REFINER_CORE_STRENGTH:-0.00}"
export V46_REFINER_TRANSITION_STRENGTH="${V46_REFINER_TRANSITION_STRENGTH:-0.35}"
export V46_DIFFUSION_CORE_STRENGTH="${V46_DIFFUSION_CORE_STRENGTH:-0.00}"
export V46_DIFFUSION_TRANSITION_STRENGTH="${V46_DIFFUSION_TRANSITION_STRENGTH:-0.25}"
export V46_DIFFUSION_REFERENCE_NOISE_SCALE="${V46_DIFFUSION_REFERENCE_NOISE_SCALE:-0.01}"
export V46_41_REFINER_CORE_COMMIT="${V46_41_REFINER_CORE_COMMIT:-0.00}"
export V46_41_REFINER_TRANSITION_COMMIT="${V46_41_REFINER_TRANSITION_COMMIT:-0.18}"
export V46_41_DIFFUSION_CORE_COMMIT="${V46_41_DIFFUSION_CORE_COMMIT:-0.00}"
export V46_41_DIFFUSION_TRANSITION_COMMIT="${V46_41_DIFFUSION_TRANSITION_COMMIT:-0.12}"
export V46_41_ROOT_XZ_DELTA_MAX_M="${V46_41_ROOT_XZ_DELTA_MAX_M:-0.05}"
export V46_41_ROOT_Y_DELTA_MAX_M="${V46_41_ROOT_Y_DELTA_MAX_M:-0.02}"
export V46_41_ROT6D_DELTA_MAX="${V46_41_ROT6D_DELTA_MAX:-0.12}"

# KBO tripwires.
export V46_41_KBO_ROOT_RANGE_ABS_MAX_M="${V46_41_KBO_ROOT_RANGE_ABS_MAX_M:-2.50}"
export V46_41_KBO_FLOOR_SHIFT_MAX_M="${V46_41_KBO_FLOOR_SHIFT_MAX_M:-1.50}"
export V46_41_KBO_BONE_LENGTH_EPS_M="${V46_41_KBO_BONE_LENGTH_EPS_M:-0.02}"
export V46_41_KBO_ACC_MAX="${V46_41_KBO_ACC_MAX:-3.0}"
export V46_41_KBO_JERK_MAX="${V46_41_KBO_JERK_MAX:-3.0}"
export V46_41_KBO_JERK_RATIO="${V46_41_KBO_JERK_RATIO:-2.5}"
export V46_41_KBO_JERK_MARGIN="${V46_41_KBO_JERK_MARGIN:-0.15}"
export V46_41_KBO_SKATE_RATIO="${V46_41_KBO_SKATE_RATIO:-2.5}"
export V46_41_KBO_SKATE_MARGIN="${V46_41_KBO_SKATE_MARGIN:-0.06}"
export V46_41_KBO_PENETRATION_MARGIN_M="${V46_41_KBO_PENETRATION_MARGIN_M:-0.20}"
export V46_41_KBO_STAGE_ANCHOR_ENABLE="${V46_41_KBO_STAGE_ANCHOR_ENABLE:-1}"
export V46_41_KBO_ANCHOR_P95_MAX_M="${V46_41_KBO_ANCHOR_P95_MAX_M:-0.85}"

# Training settings.
export V46_ENABLE_REFINER=1
export V46_ENABLE_DIFFUSION=1
export V46_ENABLE_TRUE_IK=1
export V46_CONTRASTIVE_EPOCHS="${V46_CONTRASTIVE_EPOCHS:-180}"
export V46_REFINER_TRAIN_STEPS="${V46_REFINER_TRAIN_STEPS:-10000}"
export V46_DIFFUSION_TRAIN_STEPS="${V46_DIFFUSION_TRAIN_STEPS:-20000}"
export V46_DIFFUSION_STEPS="${V46_DIFFUSION_STEPS:-50}"

# Transition-budget policy.
export V46_TRANSITION_BUDGET_ENABLE=1
export V46_TRANSITION_INBETWEEN_ENABLE=1
export V46_TRANSITION_MIN_FRAMES=10
export V46_TRANSITION_MAX_FRAMES=28
export V46_TRANSITION_RATIO=0.18
export V46_TRANSITION_MASK_HALO=6
export V46_TRANSITION_MIN_CORE_FRAMES=30
export V46_CORE_WARP_MIN=0.72
export V46_CORE_WARP_MAX=1.38

AUDIO="${AUDIO:-test_music_bank/dunhuangwu2.wav}"
AUDIO_STEM="$(basename "${AUDIO%.*}")"
CFG="${CFG:-configs/v46_motionrag_diff_config.json}"
RUN_ROOT="${RUN_ROOT:-output/v46_42_stage_anchor_guided_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT/logs"
export V46_41_HN_DPO_DIR="${V46_41_HN_DPO_DIR:-$RUN_ROOT/v46_42_hn_pairs}"

ROUTER_CKPT="${V26_ROUTER_CKPT:-output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt}"
PLANNER_CKPT="${V26_PLANNER_CKPT:-output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt}"
V23_CKPT="${V26_V23_CKPT:-./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt}"
INDEX_JSON="${V26_INDEX_JSON:-data/v35_source_aware/v21_shared_event_index_source_aware.json}"
DURATION_NPZ="${V26_DURATION_INDEX_NPZ:-data/v35_source_aware/v26_music_dominant_duration_index_source_aware.npz}"
HIER_NPZ="${V34_HIERARCHY_INDEX_NPZ:-data/v35_source_aware/v34_hierarchical_event_index_source_aware.npz}"

for f in "$ROUTER_CKPT" "$PLANNER_CKPT" "$V23_CKPT" "$INDEX_JSON" "$DURATION_NPZ"; do
  [[ -e "$f" ]] || { echo "[ERROR] missing required file: $f"; exit 2; }
done

echo "[V46.42 RUN_ROOT] $RUN_ROOT"
echo "[V46.42 AUDIO] $AUDIO"

echo "[1/14] Canonicalize Chang-E BVH: 6DoF/cm -> rot-only/meter"
python tools/canonicalize_chang_e_bvh_rot_only_meter.py \
  --in_dir change \
  --out_dir output/change_rot_only_meter_bvh \
  --root_scale 0.01 \
  --max_nonroot_pos_range 0.001

CHANGE_CANON_DIR="${CHANGE_CANON_DIR:-output/change_rot_only_meter_bvh}"

echo "[2/14] Apply V46.38 routing patch + V46.41 TGT + V46.42 stability fixes"
python tools/apply_v46_38_complete_routing_patch.py
python tools/apply_v46_41_stage_anchor_guided_tgt_patch.py
python tools/apply_v46_42_stability_alignment_patch.py
python -m py_compile \
  tools/v46_38_music_action_descriptor.py \
  tools/v46_38_build_music_semantic_slot_descriptor.py \
  tools/v46_38_build_aesd_event_semantics.py \
  tools/apply_v46_38_complete_routing_patch.py \
  tools/apply_v46_41_stage_anchor_guided_tgt_patch.py \
  tools/apply_v46_42_stability_alignment_patch.py \
  tools/v46_42_verify_tgt_kbo_report.py \
  tools/v46_42_extract_hn_dpo_pairs.py \
  tools/v46_42_train_hn_dpo_diffusion.py \
  tools/v46_motionrag_diff.py

echo "[3/14] Rebuild Event-RAG DB from canonicalized Chang-E BVH"
RAW_DB_DIR="$RUN_ROOT/train_all_db"
python tools/v46_motionrag_diff.py --config "$CFG" build-db \
  --motion_dirs "$CHANGE_CANON_DIR" \
  --out_db "$RAW_DB_DIR"
RAW_DB="$RAW_DB_DIR/events.npz"

echo "[4/14] Build AESD action-event semantic DB"
AESD_DB="$RAW_DB_DIR/events_aesd.npz"
python tools/v46_38_build_aesd_event_semantics.py \
  --db "$RAW_DB" \
  --out "$AESD_DB" \
  --json "$RUN_ROOT/V46_42_AESD_AUDIT.json"

echo "[5/14] Build strict final MSSD"
MSSD_JSON="$RUN_ROOT/${AUDIO_STEM}_v46_42_mssd.json"
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

CONTRASTIVE="$RUN_ROOT/v44_v46_42_mssd_aesd.pt"
REFINER="$RUN_ROOT/v45_v46_42_refiner.pt"
DIFFUSION="$RUN_ROOT/v46_v46_42_diffusion.pt"


echo "[6/14] Train V44 with MSSD-AESD Semantic OT"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0 python tools/v46_motionrag_diff.py --config "$CFG" train-contrastive \
  --db "$AESD_DB" \
  --unpaired_audio_dirs "${AUDIO_DIRS[@]}" \
  --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --epochs "$V46_CONTRASTIVE_EPOCHS" \
  --out "$CONTRASTIVE"

echo "[7/14] Train V45 transition refiner"
python tools/v46_motionrag_diff.py --config "$CFG" train-refiner \
  --db "$AESD_DB" \
  --steps "$V46_REFINER_TRAIN_STEPS" \
  --out "$REFINER"

echo "[8/14] Train V46 masked diffusion"
python tools/v46_motionrag_diff.py --config "$CFG" train-diffusion \
  --db "$AESD_DB" \
  --steps "$V46_DIFFUSION_TRAIN_STEPS" \
  --diffusion_steps "$V46_DIFFUSION_STEPS" \
  --out "$DIFFUSION"

echo "[9/14] Generate MotionRAG baseline"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=0 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_42_motionrag.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_42_motionrag.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_42_motionrag.mp4"

echo "[10/14] Generate Refiner + IK with V46.41 TGT/KBO"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=0 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" --refiner "$REFINER" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_42_refiner_ik.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_42_refiner_ik.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_42_refiner_ik.mp4"

echo "[11/14] Generate Diffusion + IK with V46.41 early-abort guided TGT/KBO"
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 V46_ENABLE_REFINER=1 V46_ENABLE_DIFFUSION=1 V46_ENABLE_TRUE_IK=1 \
python tools/v46_motionrag_diff.py --config "$CFG" generate \
  --audio "$AUDIO" --slots_json "$MSSD_JSON" --music_semantic_dirs "${SEMANTIC_DIRS[@]}" \
  --db "$AESD_DB" --contrastive "$CONTRASTIVE" --refiner "$REFINER" --diffusion "$DIFFUSION" \
  --out "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.npy" \
  --json "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.report.json" \
  --render_output "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.mp4"

echo "[12/14] Verify routing + V46.41 TGT/KBO safety"
python tools/v46_38_verify_routing_report.py \
  --report "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.report.json" \
  --mssd "$MSSD_JSON"
python tools/v46_42_verify_tgt_kbo_report.py \
  --report "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.report.json" \
  --require_v46_38_routing \
  --require_v46_41_tokens \
  --require_v46_42_metadata \
  --out "$RUN_ROOT/V46_42_TGT_KBO_VERIFY.json"

echo "[13/14] Extract hard-negative DPO-style preference pairs"
python tools/v46_42_extract_hn_dpo_pairs.py \
  --report "$RUN_ROOT/${AUDIO_STEM}_v46_42_diffusion_ik.report.json" \
  --pair_dir "$V46_41_HN_DPO_DIR" \
  --out_jsonl "$RUN_ROOT/V46_42_HN_DPO_PAIRS.jsonl" || true

if [[ "${V46_42_ENABLE_HN_DPO_FINETUNE:-0}" == "1" && -s "$RUN_ROOT/V46_42_HN_DPO_PAIRS.jsonl" ]]; then
  echo "[13b] Optional HN-DPO-style diffusion fine-tune"
  python tools/v46_42_train_hn_dpo_diffusion.py \
    --config "$CFG" \
    --base_diffusion "$DIFFUSION" \
    --pairs_jsonl "$RUN_ROOT/V46_42_HN_DPO_PAIRS.jsonl" \
    --steps "$V46_42_HN_DPO_STEPS" \
    --kinetic_weight "$V46_42_HN_DPO_KINETIC_WEIGHT" \
    --kinetic_floor_ratio "$V46_42_HN_DPO_KINETIC_FLOOR_RATIO" \
    --out "$RUN_ROOT/v46_v46_42_diffusion_hn_dpo_kinetic.pt"
fi

echo "[14/14] Summarize"
python - <<PY
import json, os
run="$RUN_ROOT"
stem="$AUDIO_STEM"
summary={"run_root": run, "audio": "$AUDIO", "canonical_bvh": "$CHANGE_CANON_DIR", "mssd": "$MSSD_JSON", "outputs": {}}
for suffix in ["motionrag", "refiner_ik", "diffusion_ik"]:
    p=os.path.join(run, f"{stem}_v46_42_{suffix}.report.json")
    if os.path.exists(p):
        r=json.load(open(p, encoding="utf-8"))
        summary["outputs"][suffix]={
            "selected_events": len(r.get("selected_event_indices", [])),
            "final_audit": r.get("final_audit", {}),
            "v46_41_tgt_kbo_summary": r.get("v46_41_tgt_kbo_summary", {}),
            "v46_42_stability_alignment": r.get("v46_42_stability_alignment", {}),
            "routing_policy_sample": (r.get("stage_reports", {}).get("retrieval", [{}]) or [{}])[0].get("routing_policy", ""),
        }
out=os.path.join(run, "V46_42_FINAL_SUMMARY.json")
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("[SUMMARY]", out)
PY

echo "[DONE] V46.42 Stage-Anchored Guided TGT finished: $RUN_ROOT"
