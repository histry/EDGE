# V10 Unified Choreo Planner v3

## Key switches

```bash
EDGE_V10_MODE=dual_auto_mid|manual_multiunit|upperdance_rag|auto_multiunit
EDGE_V10_MID_FRAMES=50,100
EDGE_V10_SEARCH_METHOD=greedy|beam
EDGE_V10_TOP_K=64
EDGE_V10_BEAM_WIDTH=8
EDGE_V10_MANUAL_UNITS="1042,56"
EDGE_RAG_SUMMARY_MODE=mean|temporal
```

## Validate

```bash
python -m py_compile \
  generate_controlled_v9.py \
  v9_rag_inference_patch.py \
  sitecustomize.py \
  generate_v10_choreo.py \
  v10_choreo_planner.py

bash scripts/run_v10_step1_dual_auto_mid.sh 2>&1 | tee logs/v10_step1_unified.log
bash scripts/run_v10_step4_auto_multiunit.sh 2>&1 | tee logs/v10_step4_beam.log
```

## Manual Unit mode

Preferred:

```bash
export EDGE_V10_MANUAL_UNITS="1042,56"
bash scripts/run_v10_step2_manual_multiunit.sh
```

Also supported:

```bash
export EDGE_V10_MANUAL_UNITS="unit_1042,unit_0056"
export EDGE_V10_MANUAL_UNITS="motions:1042,motions:56"
```

Legacy fallback, not recommended for fair Manual-vs-Auto comparison:

```bash
export EDGE_V10_MANUAL_MID_POSES="/path/mid1.npy,/path/mid2.npy"
bash scripts/run_v10_step2_manual_multiunit.sh
```

## Greedy vs Beam ablation

```bash
bash scripts/run_v10_ablation_beam_vs_greedy.sh
```
