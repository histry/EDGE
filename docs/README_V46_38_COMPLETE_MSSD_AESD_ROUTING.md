# V46.38 Complete MSSD + AESD + Routing-aware Event-RAG

This package implements the full chain:

```text
Music semantic labels -> Action semantic labels -> Routing cost -> Global path search
```

It is designed for the current `histry/EDGE` V46 codebase and keeps the existing
V44/V45/V46 training modules while replacing the semantic/retrieval interface.

## Files

```text
tools/v46_38_music_action_descriptor.py
  Unified MSSD parser + AESD ontology/routing helpers.

tools/v46_38_build_music_semantic_slot_descriptor.py
  Builds strict final MSSD from V21 router + V26 planner + V23 schedule report.

tools/v46_38_build_aesd_event_semantics.py
  Rebuilds action-side AESD metadata from Event-RAG events.npz.

tools/apply_v46_38_complete_routing_patch.py
  Patches tools/v46_motionrag_diff.py by injecting overrides before main().
  It actually overrides audio_slots, external sidecar parsing, V44 unpaired OT,
  and retrieve_schedule.

tools/v46_38_verify_routing_report.py
  Verifies that the final report used strict final MSSD and V46.38 MSSD-AESD routing.

scripts/run_v46_38_mssd_aesd_routing_full_retrain.sh
  Full rebuild/retrain/generate script.
```

## What is implemented

1. MSSD solves the old duplication between external music semantic sidecar and slot semantic array.
2. V44 training no longer inherits `V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1`; unpaired audio may use weak `usage=train_semantic` descriptors.
3. V46 generation requires a strict final MSSD produced by V21/V26/V23:
   `usage=generate_schedule`, `is_final_schedule=true`, `slot_source=v21_router_v26_planner`.
4. MSSD records `router_ckpt`, `planner_ckpt`, `v23_ckpt`, and `raw_schedule_json`.
5. AESD adds action-side soft music-response probabilities, natural duration, stage/energy/rhythm/locomotion/support profiles, entry/exit/contact state summaries, quality/confidence, and boundary risk.
6. V44 Semantic OT uses MSSD slot probability vectors and AESD music-response probabilities when constructing pseudo positives.
7. Event-RAG retrieval uses a global beam-search objective with:
   contrastive similarity, MSSD-AESD semantic score, legacy semantic bonus,
   natural duration score, stage score, event quality, boundary-risk penalty,
   transition cost, source/dance/family diversity penalties.

## Install from repo root

```bash
cd /home/disk/lsm/storage/EDGE

cp /mnt/data/v46_38_complete_solution/tools/*.py tools/
cp /mnt/data/v46_38_complete_solution/scripts/*.sh scripts/
chmod +x tools/v46_38_*.py tools/apply_v46_38_complete_routing_patch.py scripts/run_v46_38_mssd_aesd_routing_full_retrain.sh

export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
python tools/apply_v46_38_complete_routing_patch.py
python -m py_compile tools/v46_38_music_action_descriptor.py tools/v46_38_build_music_semantic_slot_descriptor.py tools/v46_38_build_aesd_event_semantics.py tools/apply_v46_38_complete_routing_patch.py tools/v46_motionrag_diff.py
```

## Full retrain/run

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH="/home/disk/lsm/storage/EDGE${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V26_V23_CKPT="./checkpoints/v23_release/v23_v2_5/v23_v2_5_seed20260610_best.pt"

nohup bash scripts/run_v46_38_mssd_aesd_routing_full_retrain.sh \
  > output/v46_38_mssd_aesd_routing_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Key switches

Training:

```bash
export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=0
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_38_OT_SEMANTIC_WEIGHT=1.25
```

Generation:

```bash
export V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1
export V46_MSSD_REQUIRE_FINAL_SCHEDULE_FOR_GENERATE=1
```

Routing:

```bash
export V46_38_ROUTE_CONTRASTIVE_WEIGHT=1.00
export V46_38_ROUTE_MSSD_AESD_WEIGHT=1.15
export V46_38_ROUTE_LEGACY_SEM_WEIGHT=0.55
export V46_38_ROUTE_DURATION_WEIGHT=0.28
export V46_38_ROUTE_QUALITY_WEIGHT=0.26
export V46_38_ROUTE_STAGE_WEIGHT=0.18
export V46_38_ROUTE_BOUNDARY_RISK_WEIGHT=0.35
export V46_38_ROUTE_CANDIDATE_TOPK=128
```

## Verification

After generation, run:

```bash
python tools/v46_38_verify_routing_report.py \
  --report "$RUN_ROOT/dunhuangwu2_v46_38_diffusion_ik.report.json" \
  --mssd "$RUN_ROOT/dunhuangwu2_v46_38_mssd.json"
```

A successful report means the final generation used strict MSSD and V46.38
MSSD-AESD routing in candidate previews.
