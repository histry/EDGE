# Fixed EDGE generate entrypoints

Replace these files in the repository root:

```text
generate_v10_choreo.py
generate_controlled_v9.py
```

They fix the `from __future__ import annotations` SyntaxError by keeping
`from __future__` at the top, while still force-loading `sitecustomize` before
V10 planner/model imports.

After replacement:

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  generate_v10_choreo.py \
  generate_controlled_v9.py \
  text_context_rag_utils.py \
  text_context_rag_model_patch.py \
  text_context_rag_io_patch.py \
  text_bridge_planner_patch.py \
  sitecustomize.py

bash scripts/run_v10_text_bridge_weight_sweep.sh 2>&1 | tee logs/textbridge_sweep_after_entrypoint_fix.log

grep -E "Text Bridge semantic|Text/Pose Context RAG attached|Text/Pose Context RAG enabled|V10 Unified planner selected units|Traceback|ERROR" -n \
  logs/textbridge_sweep_after_entrypoint_fix.log \
  logs/textbridge_w*.log
```
