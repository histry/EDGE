#!/usr/bin/env bash
set -euo pipefail

ROOT="${EDGE_ROOT:-/home/disk/lsm/storage/EDGE}"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

: "${V26_INDEX_JSON:?Set V26_INDEX_JSON}"
: "${V26_DURATION_INDEX_NPZ:?Set V26_DURATION_INDEX_NPZ}"
: "${V26_ROUTER_CKPT:?Set V26_ROUTER_CKPT}"
: "${V26_V23_CKPT:?Set V26_V23_CKPT}"

RUN_ID="${V28_RUN_ID:-v28_ragdiff_hyper_clap_public_$(date +%Y%m%d_%H%M%S)}"
V28_ROOT="${V28_ROOT:-output/${RUN_ID}}"
mkdir -p "$V28_ROOT"

python -m py_compile \
  tools/v26_hierarchical_graph_scheduler.py \
  tools/build_v26_hierarchical_event_index.py \
  tools/v27_deep_music_features.py \
  tools/check_v27_clap.py \
  tools/build_v27_transition_diffusion_dataset.py \
  tools/v27_transition_diffusion.py \
  train_v27_hyperbolic_hierarchy.py \
  train_v27_transition_diffusion.py \
  tools/schedule_v26_whole_song.py \
  tools/evaluate_v27_public_metrics.py \
  tools/evaluate_v28_experiment_table.py

python train_v27_hyperbolic_hierarchy.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --out_dir "$V28_ROOT/v27_hyperbolic_hierarchy" \
  --epochs "${V28_HYPER_EPOCHS:-320}" \
  --batch_size "${V28_HYPER_BATCH_SIZE:-256}" \
  --hidden_dim "${V28_HYPER_HIDDEN_DIM:-128}" \
  --lr "${V28_HYPER_LR:-2e-4}" \
  --weight_decay "${V28_HYPER_WEIGHT_DECAY:-1e-4}" \
  --temperature "${V28_HYPER_TEMPERATURE:-0.12}" \
  --radius_weight "${V28_HYPER_RADIUS_WEIGHT:-0.20}" \
  --center_weight "${V28_HYPER_CENTER_WEIGHT:-0.08}" \
  --seed "${V28_SEED:-20260610}"

export V27_HYPERBOLIC_CKPT
V27_HYPERBOLIC_CKPT="$(cat "$V28_ROOT/v27_hyperbolic_hierarchy/BEST_V27_HYPERBOLIC_CKPT.txt")"

python tools/build_v26_hierarchical_event_index.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --hyperbolic_ckpt "$V27_HYPERBOLIC_CKPT" \
  --out_npz "$V28_ROOT/v28_hyperbolic_hierarchical_event_index.npz" \
  --out_json "$V28_ROOT/v28_hyperbolic_hierarchical_event_index.json"

export V26_HIERARCHY_INDEX_NPZ="$V28_ROOT/v28_hyperbolic_hierarchical_event_index.npz"

python tools/build_v27_transition_diffusion_dataset.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --out_npz "$V28_ROOT/v28_transition_diffusion_dataset.npz" \
  --max_len "${V28_TRANSITION_DATA_MAX_LEN:-96}" \
  --min_len "${V28_TRANSITION_DATA_MIN_LEN:-8}" \
  --samples_per_event "${V28_TRANSITION_SAMPLES_PER_EVENT:-4}" \
  --source_pairs_per_event "${V28_TRANSITION_SOURCE_PAIRS_PER_EVENT:-0.45}" \
  --pseudo_pairs_per_event "${V28_TRANSITION_PSEUDO_PAIRS_PER_EVENT:-0.75}" \
  --condition_dropout "${V28_TRANSITION_CONDITION_DROPOUT:-0.05}" \
  --seed "${V28_SEED:-20260610}"

python train_v27_transition_diffusion.py \
  --data "$V28_ROOT/v28_transition_diffusion_dataset.npz" \
  --out_dir "$V28_ROOT/v27_transition_diffusion" \
  --epochs "${V28_DIFFUSION_EPOCHS:-260}" \
  --batch_size "${V28_DIFFUSION_BATCH_SIZE:-96}" \
  --hidden_dim "${V28_DIFFUSION_HIDDEN_DIM:-384}" \
  --diffusion_steps "${V28_DIFFUSION_STEPS_TRAIN:-64}" \
  --lr "${V28_DIFFUSION_LR:-2e-4}" \
  --weight_decay "${V28_DIFFUSION_WEIGHT_DECAY:-1e-4}" \
  --patience "${V28_DIFFUSION_PATIENCE:-45}" \
  --num_workers "${V28_NUM_WORKERS:-2}" \
  --seed "${V28_SEED:-20260610}"

export V27_TRANSITION_DIFFUSION_CKPT
V27_TRANSITION_DIFFUSION_CKPT="$(cat "$V28_ROOT/v27_transition_diffusion/BEST_V27_TRANSITION_DIFFUSION_CKPT.txt")"

export V26_OUT_DIR="${V28_GENERATE_OUT_DIR:-$V28_ROOT/three_music_generation}"
export V26_MUSIC="${V26_MUSIC:-test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav}"
unset V26_MUSIC_GLOB

export V26_HIERARCHICAL_RETRIEVAL="${V26_HIERARCHICAL_RETRIEVAL:-1}"
export V26_GRAPH_SCHEDULER="${V26_GRAPH_SCHEDULER:-1}"
export V26_HIERARCHY_WEIGHT="${V26_HIERARCHY_WEIGHT:-0.60}"
export V26_GRAPH_EDGE_WEIGHT="${V26_GRAPH_EDGE_WEIGHT:-0.55}"
export V26_GRAPH_NODE_TOP_K="${V26_GRAPH_NODE_TOP_K:-128}"
export V26_CANDIDATE_TOP_K="${V26_CANDIDATE_TOP_K:-768}"
export V26_BEAM_SIZE="${V26_BEAM_SIZE:-48}"

export V27_DEEP_MUSIC_FEATURES="${V27_DEEP_MUSIC_FEATURES:-1}"
export V27_DEEP_MUSIC_MODEL="${V27_DEEP_MUSIC_MODEL:-clap}"
export V27_DEEP_MUSIC_WEIGHT="${V27_DEEP_MUSIC_WEIGHT:-0.30}"
export V27_REQUIRE_DEEP_MUSIC="${V27_REQUIRE_DEEP_MUSIC:-1}"
export V27_DEEP_MUSIC_MIN_SUCCESS="${V27_DEEP_MUSIC_MIN_SUCCESS:-0.60}"
export V27_DEEP_MUSIC_CACHE="${V27_DEEP_MUSIC_CACHE:-$V28_ROOT/deep_music_cache}"

if [[ "${V27_DEEP_MUSIC_FEATURES:-1}" == "1" || "${V27_DEEP_MUSIC_FEATURES:-1}" == "true" ]]; then
  python tools/check_v27_clap.py \
    --audio test_music_bank/dunhuangwu2.wav \
    --model "${V27_DEEP_MUSIC_MODEL:-clap}" \
    --cache_dir "$V28_ROOT/clap_check_cache" \
    --out_json "$V28_ROOT/clap_check.json" \
    --strict "${V27_REQUIRE_DEEP_MUSIC:-1}" \
    --min_success "${V27_DEEP_MUSIC_MIN_SUCCESS:-0.60}"
fi

export V27_TRANSITION_DIFFUSION="${V27_TRANSITION_DIFFUSION:-1}"
export V27_TRANSITION_DIFFUSION_BLEND="${V27_TRANSITION_DIFFUSION_BLEND:-0.35}"
export V27_TRANSITION_DIFFUSION_STEPS="${V27_TRANSITION_DIFFUSION_STEPS:-18}"

export V26_MUSIC_DOMINANT_TIMING="${V26_MUSIC_DOMINANT_TIMING:-1}"
export V26_LOCK_MUSIC_BOUNDARIES="${V26_LOCK_MUSIC_BOUNDARIES:-1}"
export V26_ALLOW_MUSIC_BOUND_OVERRIDE="${V26_ALLOW_MUSIC_BOUND_OVERRIDE:-1}"
export V26_MULTI_EVENT_PHRASES="${V26_MULTI_EVENT_PHRASES:-1}"
export V26_MAX_SINGLE_EVENT_SECONDS="${V26_MAX_SINGLE_EVENT_SECONDS:-3.00}"
export V26_CALM_MAX_SINGLE_EVENT_SECONDS="${V26_CALM_MAX_SINGLE_EVENT_SECONDS:-2.80}"
export V26_MIN_SUBPHRASE_SECONDS="${V26_MIN_SUBPHRASE_SECONDS:-1.45}"
export V26_MAX_EVENTS_PER_PHRASE="${V26_MAX_EVENTS_PER_PHRASE:-4}"

export V26_TRANSITION_MIN_FRAMES="${V26_TRANSITION_MIN_FRAMES:-24}"
export V26_TRANSITION_MAX_FRAMES="${V26_TRANSITION_MAX_FRAMES:-120}"
export V26_TRANSITION_YAW_LIMIT_DPS="${V26_TRANSITION_YAW_LIMIT_DPS:-180}"
export V26_YAW_TRANSITION_SAFETY_FACTOR="${V26_YAW_TRANSITION_SAFETY_FACTOR:-2.10}"
export V26_MIN_TIME_WARP="${V26_MIN_TIME_WARP:-0.82}"
export V26_MAX_TIME_WARP="${V26_MAX_TIME_WARP:-1.50}"
export V26_BOUNDARY_VELOCITY_PENALTY_WEIGHT="${V26_BOUNDARY_VELOCITY_PENALTY_WEIGHT:-0.65}"
export V26_BOUNDARY_ACCELERATION_PENALTY_WEIGHT="${V26_BOUNDARY_ACCELERATION_PENALTY_WEIGHT:-0.65}"
export V26_TURN_PEAK_PENALTY_WEIGHT="${V26_TURN_PEAK_PENALTY_WEIGHT:-1.10}"
export V26_EDGE_DAMPING_FRAMES="${V26_EDGE_DAMPING_FRAMES:-12}"
export V26_EDGE_DAMPING_STRENGTH="${V26_EDGE_DAMPING_STRENGTH:-0.75}"

bash scripts/run_v26_whole_song.sh

KEYS="${V28_EVAL_KEYS:-dunhuangwu2;dunhuangwu3;dunhuangwu4}"
IFS=';' read -ra KEY_ITEMS <<< "$KEYS"
for key in "${KEY_ITEMS[@]}"; do
  if [[ -f "tools/evaluate_v26_long_dance.py" ]]; then
    python tools/evaluate_v26_long_dance.py \
      --motion "$V26_OUT_DIR/${key}_v26.npy" \
      --schedule_report "$V26_OUT_DIR/${key}_v26.schedule_report.json" \
      --out_json "$V26_OUT_DIR/${key}_v26.evaluation.json"
  fi
  if [[ -f "tools/evaluate_v26_hg_report.py" ]]; then
    python tools/evaluate_v26_hg_report.py \
      --motion "$V26_OUT_DIR/${key}_v26.npy" \
      --schedule_report "$V26_OUT_DIR/${key}_v26.schedule_report.json" \
      --out_json "$V26_OUT_DIR/${key}_v26.hg_evaluation.json"
  fi
  python tools/evaluate_v27_public_metrics.py \
    --motion "$V26_OUT_DIR/${key}_v26.npy" \
    --audio "test_music_bank/${key}.wav" \
    --index_json "$V26_INDEX_JSON" \
    --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
    --out_json "$V26_OUT_DIR/${key}_v26.public_metrics.json"
done

RUN_ARGS=(--run "proposed=$V26_OUT_DIR")
if [[ -n "${V28_BASELINE_RUNS:-}" ]]; then
  IFS=';' read -ra BASELINE_ITEMS <<< "$V28_BASELINE_RUNS"
  for item in "${BASELINE_ITEMS[@]}"; do
    [[ -n "$item" ]] && RUN_ARGS+=(--run "$item")
  done
fi

python tools/evaluate_v28_experiment_table.py \
  "${RUN_ARGS[@]}" \
  --keys "$KEYS" \
  --audio_dir "test_music_bank" \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --out_json "$V28_ROOT/v28_experiment_table.json" \
  --out_csv "$V28_ROOT/v28_experiment_table.csv" \
  --out_md "$V28_ROOT/v28_experiment_table.md"

echo "[DONE] V28 overnight run: $V28_ROOT"
echo "$V28_ROOT" > output/LATEST_V28_RAGDIFF_HYPER_CLAP_PUBLIC_RUN.txt
