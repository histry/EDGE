# Text-Bridged Pose/Text Context RAG Full Loop

This patch implements the first trainable closed loop:

```text
Text query
→ semantic re-ranking in Unified Planner
→ retrieved motion clips
→ Pose/Text Context Encoder
→ cross-attention memory
→ diffusion generation
```

## Files

```text
text_context_rag_utils.py
text_context_rag_model_patch.py
text_context_rag_io_patch.py
text_bridge_planner_patch.py
sitecustomize.py
scripts/run_v10_text_context_step4.sh
scripts/run_v10_text_bridge_weight_sweep.sh
scripts/run_train_text_context_rag_adapter.sh
README_TEXT_CONTEXT_RAG_FULL_LOOP.md
```

## Stage status after this patch

| Stage | Status |
|---|---|
| Text Bridge encoder | Existing |
| motion unit caption + text embedding DB | Existing |
| Text-guided semantic re-ranking | Added |
| Pose/Text Context Encoder | Added |
| Cross-Attention RAG memory | Added by appending context tokens to decoder memory |
| Diffusion generation using context | Added |
| Training support | Added via self-context clips in p_losses |

## Switches

```bash
export EDGE_ENABLE_TEXT_CONTEXT_RAG=1
export EDGE_TEXT_BRIDGE_WEIGHT=0.30
export EDGE_TEXT_QUERY="敦煌舞，飞天风格，高能量，上肢大幅舒展，流动转身，空间展开"
export EDGE_TEXT_CONTEXT_TRAIN_SELF=1
```

## Quick test

```bash
python -m py_compile \
  text_context_rag_utils.py \
  text_context_rag_model_patch.py \
  text_context_rag_io_patch.py \
  text_bridge_planner_patch.py \
  sitecustomize.py

bash scripts/run_v10_text_bridge_weight_sweep.sh
```

Check:

```bash
grep -E "Text Bridge semantic|Text/Pose Context RAG attached|V10 Unified planner selected units|Traceback|ERROR" -n logs/textbridge_w*.log
```

## Training

```bash
EPOCHS=8 BATCH_SIZE=16 bash scripts/run_train_text_context_rag_adapter.sh
```

Then evaluate:

```bash
CHECKPOINT=runs/train_stage45/v10_text_context_rag_adapter/weights/train-8.pt \
bash scripts/run_v10_text_bridge_weight_sweep.sh
```

## Accurate claim

We implement a trainable Text/Pose Context RAG pathway by using text-guided
semantic re-ranking to retrieve motion units, encoding the retrieved pose clips
and text embeddings as context tokens, and appending them to the decoder
cross-attention memory.
