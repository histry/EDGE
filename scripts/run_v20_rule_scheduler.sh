#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/disk/lsm/storage/EDGE
source /home/disk/lsm/conda_envs/edge/bin/activate 2>/dev/null || true
export PYTHONPATH=$PWD:${PYTHONPATH:-}
RUN_ROOT=${RUN_ROOT:-output/v20_event_graph_scheduler_$(date +%Y%m%d_%H%M%S)}
EVENT_DB=${EVENT_DB:-data/dunhuang_dynamic_event_rag/index_dynamic_event.json}
AUDIO_LIST=${AUDIO_LIST:-"dunhuangwu2 dunhuangwu3 dunhuangwu4"}
mkdir -p "$RUN_ROOT"
for MUSIC in $AUDIO_LIST; do
  python tools/extract_music_event_stream.py \
    --audio test_music_bank/${MUSIC}.wav \
    --out_npy "$RUN_ROOT/music_${MUSIC}_events.npy" \
    --out_json "$RUN_ROOT/music_${MUSIC}_events.json" \
    --num_frames ${V20_NUM_FRAMES:-150}
  python tools/schedule_event_graph_phrase.py \
    --event_db "$EVENT_DB" \
    --music_events "$RUN_ROOT/music_${MUSIC}_events.npy" \
    --out "$RUN_ROOT/${MUSIC}_event_graph.npy" \
    --num_frames ${V20_NUM_FRAMES:-150} \
    --beam_size ${V20_BEAM_SIZE:-16} \
    --target_event_count ${V20_TARGET_EVENT_COUNT:-4} \
    --candidate_top_k ${V20_CANDIDATE_TOP_K:-360} \
    --min_source_gap ${V20_MIN_SOURCE_GAP:-120} \
    --visual_weight ${V20_VISUAL_WEIGHT:-0.75} \
    --quality_weight ${V20_QUALITY_WEIGHT:-0.80} \
    --safety_weight ${V20_SAFETY_WEIGHT:-0.30} \
    --activity_weight ${V20_ACTIVITY_WEIGHT:-0.12} \
    --event_weight ${V20_EVENT_WEIGHT:-0.60} \
    --emotion_weight ${V20_EMOTION_WEIGHT:-1.00} \
    --transition_weight ${V20_TRANSITION_WEIGHT:-0.50} \
    --diversity_weight ${V20_DIVERSITY_WEIGHT:-0.30} \
    2>&1 | tee "$RUN_ROOT/schedule_${MUSIC}.log"
  python render_from_npy.py --motion "$RUN_ROOT/${MUSIC}_event_graph.npy" --audio test_music_bank/${MUSIC}.wav --output "$RUN_ROOT/${MUSIC}_event_graph_fixed.mp4" --camera_mode fixed || true
  python render_from_npy.py --motion "$RUN_ROOT/${MUSIC}_event_graph.npy" --audio test_music_bank/${MUSIC}.wav --output "$RUN_ROOT/${MUSIC}_event_graph_follow.mp4" --camera_mode follow || true
  python tools/evaluate_dunhuang_motion.py --motion "$RUN_ROOT/${MUSIC}_event_graph.npy" --out_json "$RUN_ROOT/${MUSIC}_eval.json" || true
done
echo "DONE: $RUN_ROOT"
