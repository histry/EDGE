# ChoreoRAG Priority Fixes for EDGE

This package contains direct replacement/add-on files for the next V10 ChoreoRAG stage.

## Files

```text
v10_choreo_planner.py                 # Replace existing planner
choreorag_unit_prior.py               # Replace existing unit prior helper
build_choreorag_stats_cache.py        # New stats cache builder
scripts/run_energy_temporal_prior_ablation.sh  # New standard ablation script
```

## Install

From EDGE repository root:

```bash
cp v10_choreo_planner.py ./v10_choreo_planner.py
cp choreorag_unit_prior.py ./choreorag_unit_prior.py
cp build_choreorag_stats_cache.py ./build_choreorag_stats_cache.py
cp scripts/run_energy_temporal_prior_ablation.sh ./scripts/run_energy_temporal_prior_ablation.sh
chmod +x scripts/run_energy_temporal_prior_ablation.sh
```

## Build stats cache

```bash
python build_choreorag_stats_cache.py \
  --rag_db data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco.npz \
  --out data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_stats.npz \
  --copy_text_embeddings
```

Then use:

```bash
export EDGE_RAG_STATS_CACHE=data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_stats.npz
```

## Energy/Expressiveness-aware planner controls

```bash
export EDGE_UNIT_MIN_ENERGY=0.35
export EDGE_UNIT_MIN_EXPRESSIVENESS=0.40
export EDGE_UNIT_BAN_LOW_ENERGY=1
export EDGE_UNIT_LOW_ENERGY_THRESHOLD=0.25
export EDGE_UNIT_ENERGY_BONUS=0.15
export EDGE_UNIT_EXPRESSIVENESS_BONUS=0.25
export EDGE_V10_TEXT_SCORE_W=0.30
export EDGE_V10_MAX_RAG_UNITS=1000000
```

## Temporal unit prior controls

```bash
export EDGE_UNIT_SOFT_PRIOR=1
export EDGE_UNIT_PRIOR_TEMPORAL=1
export EDGE_UNIT_PRIOR_WINDOW=41
export EDGE_UNIT_PRIOR_FEATURES=upper+torso
export EDGE_UNIT_PRIOR_DCT=1
export EDGE_UNIT_PRIOR_LOW_FREQ_K=4
export EDGE_UNIT_PRIOR_STRENGTH=0.012
```

Important safety invariant: `choreorag_unit_prior.py` never constrains contacts or root X/Z. `all_no_root` includes rotations and optional root_y only.

## Standard ablation

```bash
export CHECKPOINT=/path/to/train-4.pt
export EDGE_V10_RAG_DB=/path/to/index_u45_s15_e10_expr_loco.npz
export EDGE_RAG_STATS_CACHE=/path/to/index_u45_s15_e10_expr_loco_stats.npz
export COMMON_ARGS="--music /path/to.wav --start_pose /path/start.npy --end_pose /path/end.npy --trajectory '0,0;0.5,0.7;-0.3,1.2;0,1.6' --feature_type hybrid --audio_dim 803"

bash scripts/run_energy_temporal_prior_ablation.sh
```

The script runs:

```text
energy_aware_off
energy_filter_only
energy_expr_filter
energy_expr_text
energy_expr_text_temporal_prior
```

Each case emits plan json, score_parts json, unit npy, final npy, and any metrics/json/mp4 that your generation pipeline creates.
