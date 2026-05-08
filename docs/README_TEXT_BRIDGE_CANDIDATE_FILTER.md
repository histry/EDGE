# Text Bridge Candidate Filtering Patch

Replace:

```text
text_bridge_planner_patch.py
scripts/run_v10_textbridge_candidate_modes.sh
```

This patch upgrades TextBridge from a weak reranking term to candidate filtering.

## Modes

```bash
EDGE_TEXT_BRIDGE_MODE=rerank      # old compatible behavior
EDGE_TEXT_BRIDGE_MODE=hybrid      # union(original top-K, semantic top-K)
EDGE_TEXT_BRIDGE_MODE=filter      # restrict beam candidates to semantic top-K
EDGE_TEXT_BRIDGE_MODE=force_topk  # semantic top-K candidates, semantic score primary
```

Recommended tests:

```bash
python -m py_compile text_bridge_planner_patch.py sitecustomize.py

bash scripts/run_v10_textbridge_candidate_modes.sh 2>&1 | tee logs/textbridge_candidate_modes_all.log

grep -E "Text Bridge candidate filtering|top_semantic|top_final|V10 Unified planner selected units|Traceback|ERROR|RuntimeError" -n \
  logs/textbridge_candidate_modes_all.log \
  logs/textbridge_*_w*.log
```

Expected:
- rerank may keep the old units.
- filter / force_topk should make semantic candidates participate in Beam Search.
- If selected units still do not change under force_topk, set:

```bash
EDGE_TEXT_BRIDGE_MODE=force_topk \
EDGE_TEXT_BRIDGE_TOP_K=64 \
EDGE_TEXT_BRIDGE_WEIGHT=1.0 \
bash scripts/run_v10_text_context_step4.sh
```
