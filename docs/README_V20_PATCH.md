# V20 Rhythm-Event Graph ChoreoRAG Patch

直接复制本目录下的 `tools/`、`model/`、`scripts/`、`train_*.py`、`v20_event_adapter_patch.py` 到 `/home/disk/lsm/storage/EDGE/`。

## 推荐执行顺序

### 1. 重建 Dynamic Rhythm Event-RAG 数据库

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}

# 按你的实际 151D motion 数据目录改 INPUT_DIR
INPUT_DIR=data/dunhuang_bvh/processed \
OUT_DIR=data/dunhuang_dynamic_event_rag \
bash scripts/run_v20_build_dynamic_event_db.sh
```

输出：

```text
data/dunhuang_dynamic_event_rag/index_dynamic_event.json
data/dunhuang_dynamic_event_rag/events/*.pkl
```

### 2. 规则版 Event-Graph Scheduler

```bash
EVENT_DB=data/dunhuang_dynamic_event_rag/index_dynamic_event.json \
bash scripts/run_v20_rule_scheduler.sh
```

输出：三首音乐的 `*_event_graph.npy`、fixed/follow 视频、评估 JSON。

### 3. 训练 DPN + Endpoint Transition Refiner

```bash
EVENT_DB=data/dunhuang_dynamic_event_rag/index_dynamic_event.json \
bash scripts/run_v20_train_transition_models.sh
```

### 4. 评价

```bash
python tools/evaluate_dunhuang_motion.py \
  --motion_dir output/v20_event_graph_scheduler_xxx \
  --out_json output/v20_eval_report.json
```

## 可调环境变量

- `V20_MIN_LEN=24`
- `V20_IDEAL_LEN=48`
- `V20_MAX_LEN=72`
- `V20_BOUNDARY_MIN_GAP=18`
- `V20_BEAM_SIZE=16`
- `V20_EVENT_WEIGHT=0.60`
- `V20_EMOTION_WEIGHT=1.00`
- `V20_DIVERSITY_WEIGHT=0.30`
- `V20_DPN_EPOCHS=500`
- `V20_TRANS_EPOCHS=800`

## 可选 Event Adapter

`v20_event_adapter_patch.py` 是可选项。只有当你后续希望重新训练 EDGE 的 event-conditioned adapter 时才启用：

```bash
export EDGE_ENABLE_V20_EVENT_ADAPTER=1
```

它会把 `cond["event_token"]` 映射到现有 `rag_summary` token 分支，避免大改 `DanceDecoder`。
