# Text/Pose Context RAG model patch v2

This fixes:

```text
RuntimeError: The size of tensor a (217) must match the size of tensor b (150)
```

Cause:
The old patch appended context tokens inside `_encode_condition_tokens`, making
`cond_tokens` length 217 before trajectory modulation, while scale/shift stayed
length 150.

Fix:
Do not change `_encode_condition_tokens`.  Append context tokens at
`DecoderLayerStack.forward`, after all frame-wise modulation is complete and just
before decoder cross-attention memory is consumed.

Replace:

```text
text_context_rag_model_patch.py
```

Then run:

```bash
python -m py_compile text_context_rag_model_patch.py sitecustomize.py

bash scripts/run_v10_text_bridge_weight_sweep.sh 2>&1 | tee logs/textbridge_sweep_after_memory_fix.log

grep -E "Text Bridge semantic|Text/Pose Context RAG attached|appended to decoder memory|V10 Unified planner selected units|Traceback|ERROR|RuntimeError" -n \
  logs/textbridge_sweep_after_memory_fix.log \
  logs/textbridge_w*.log
```
