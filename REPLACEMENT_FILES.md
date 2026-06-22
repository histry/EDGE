# EDGE V34 Source-aware RAG Replacement

## 解决的问题

你的 BVH 数据是 motion-only 动作库，不是音乐-动作配对数据。命名规则为：

```text
dyl_002_Take_003.bvh
│   │   │
│   │   └── category_id = Take_003
│   └────── repeat_id = 002
└────────── dancer_id = dyl
```

现有 Event-RAG 代码保留了这些信息，但主要把它们当成文件名字符串，并没有在构库和检索时正式使用。审计结果显示：

```text
dynamic event 原始库：min=119, max=687, max/min=5.8
V34 shared index：min=4, max=134, max/min=33.5
```

这说明不均衡从动态切分阶段已经存在，并在 shared index / V34 quality filtering 中被放大。该替换包的目标是：

1. 显式解析 `dancer_id / repeat_id / category_id / source_uid`；
2. 构建 JSON+NPZ 对齐的 source-aware balanced index；
3. 重建匹配的 hierarchy NPZ，避免数组长度错位；
4. 在 V34 检索打分中加入 source/category/dancer/repeat 约束；
5. 保留 Dense Boundary、Dynamic Relax、Rhythm Repair、Inpainting 和 Latent Blending。

## 需要替换/新增的文件

将本目录中的文件复制到 EDGE 根目录同名路径：

```text
tools/v34_source_aware_rag.py                 新增
tools/build_v34_source_aware_index.py         新增
tools/build_v21_shared_event_index.py         替换
tools/v34_warp_aware_retrieval.py             替换
scripts/launch_v34_source_aware_rag.sh        新增
scripts/resume_v34_inference_v33ckpt.sh       替换
```

## 主要机制

### 1. Source metadata 结构化

每个 event 会补充：

```json
{
  "dancer_id": "dyl",
  "repeat_id": "002",
  "category_id": "Take_003",
  "source_uid": "dyl_002_Take_003",
  "dancer_category_group": "dyl_Take_003",
  "category_repeat_group": "Take_003_002"
}
```

### 2. Source-aware balanced index

`tools/build_v34_source_aware_index.py` 会读取：

```text
data/v34_shared_event_index.json
data/v26_music_dominant_duration_index.npz
```

输出对齐的新文件：

```text
data/v34_source_aware/v34_shared_event_index_source_aware.json
data/v34_source_aware/v34_shared_event_index_source_aware.npz
data/v34_source_aware/v34_shared_event_index_source_aware.audit.json
```

它不是只改 JSON，而是同步过滤 NPZ 中所有第一维等于事件数的数组，避免 schedule 时 metadata 和 tensor 错位。

### 3. V34 检索约束

`tools/v34_warp_aware_retrieval.py` 新增：

```text
V34_SOURCE_AWARE_RAG
V34_SOURCE_AWARE_WEIGHT
V34_SOURCE_AWARE_WINDOW
V34_SOURCE_UID_REPEAT_WEIGHT
V34_DANCER_CATEGORY_REPEAT_WEIGHT
V34_CATEGORY_REPEAT_WEIGHT
V34_DANCER_REPEAT_WEIGHT
V34_REPEAT_ID_REPEAT_WEIGHT
V34_CATEGORY_PRIOR_BALANCE_WEIGHT
V34_DANCER_PRIOR_BALANCE_WEIGHT
V34_REPEAT_PRIOR_BALANCE_WEIGHT
```

每个 schedule part 会写入：

```json
{
  "source_uid": "...",
  "dancer_id": "...",
  "repeat_id": "...",
  "category_id": "...",
  "source_aware_transition_penalty": 0.0,
  "source_aware_meta": {}
}
```

## 推荐运行：纯推理重建库

替换文件后执行：

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  tools/v34_source_aware_rag.py \
  tools/build_v34_source_aware_index.py \
  tools/build_v21_shared_event_index.py \
  tools/v34_warp_aware_retrieval.py

bash -n \
  scripts/launch_v34_source_aware_rag.sh \
  scripts/launch_v34_rhythm_repair.sh \
  scripts/resume_v34_inference_v33ckpt.sh

tmux kill-session -t v34_source_aware_rag 2>/dev/null || true

tmux new-session -d -s v34_source_aware_rag "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V34_TRAIN=0
export V34_SOURCE_AWARE_REBUILD=1
export V34_SOURCE_AWARE_REBUILD_HIERARCHY=1

export V34_SOURCE_CAP_PER_SOURCE_UID=64
export V34_SOURCE_CATEGORY_CAP_FACTOR=1.35
export V34_SOURCE_REPEAT_CAP_FACTOR=1.60
export V34_SOURCE_DANCER_CAP_FACTOR=1.50

export V34_SOURCE_AWARE_RAG=1
export V34_SOURCE_AWARE_WEIGHT=1.0
export V34_SOURCE_AWARE_WINDOW=8

export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_DENSE_SCORE=1
export V34_RHYTHM_DEGRADATION_PENALTY=1
export V34_BOUNDARY_INPAINT=1
export V34_LATENT_SNIPPET_BLEND=1
export V34_USE_GPU_RETRIEVAL=1

export RUN_ID=v34_source_aware_rag_infer_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p \"\$RUN_ROOT\"

bash scripts/launch_v34_source_aware_rag.sh 2>&1 | tee -a \"\$RUN_ROOT/outer.log\"
'"

sleep 5
RUN_ROOT=$(cat output/LATEST_V34_SOURCE_AWARE_RAG.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -80 "$RUN_ROOT/outer.log"
```

## 完整训练

如果要完整训练 Contact-INR / diffusion transition：

```bash
export V34_TRAIN=1
bash scripts/launch_v34_source_aware_rag.sh
```

注意：完整训练会显著更慢。建议先跑纯推理验证 source-aware 检索和 schedule report。

## 结果检查

```bash
cd /home/disk/lsm/storage/EDGE

RUN_ROOT=$(cat output/LATEST_V34_SOURCE_AWARE_RAG.txt)

cat data/v34_source_aware/v34_shared_event_index_source_aware.audit.json | head -120

find "$RUN_ROOT" -type f \
  \( -name "*.schedule_report.json" \
     -o -name "*.boundary_v34.json" \
     -o -name "*.contact_metrics.json" \
     -o -name "*.jitter.json" \
     -o -name "*.mp4" \
     -o -name "*SUMMARY.json" \) | sort

grep -nE "source_aware|V34 SOURCE-AWARE|V34-RELAX|PASS|DONE|Traceback|RuntimeError|ERROR" \
  "$RUN_ROOT/run.log" "$RUN_ROOT/outer.log" 2>/dev/null | tail -200
```

## 论文表述

可以写成：

> 由于原始敦煌舞 BVH 数据是无音乐配对的 motion-only 低资源动作库，且源文件包含舞者、重复次数和动作类别结构，本文将 Event-RAG 数据库从 event-level flat pool 升级为 source-aware, category-balanced, quality-weighted memory bank。该机制保留舞者习惯作为风格多样性，同时通过质量加权和来源约束抑制动作失误、类别偏置和近重复检索。
