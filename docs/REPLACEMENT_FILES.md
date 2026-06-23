# V35 Source-aware RAG Segmentation Replacement

## 结论

当前 RAG 数据库切割算法需要改进。

现有代码里已经有动态切割，不是简单固定窗口：`tools/build_dynamic_rhythm_event_db.py` 会使用动作能量低谷、Root-Y 速度零交叉和接触切换来生成事件边界。这一点是合理的。但它仍然有四个科研级缺口：

1. 切割阶段不显式理解 `dyl_002_Take_003` 这类来源结构，导致舞者、重复次数、类别的分布偏置只能在后处理阶段补救。
2. `pose_hold / calm_flow / neutral_flow` 中“第一秒快速到位，后半段保持”的片段会被当作安全片段大量选入数据库。
3. 同一个舞者、同一类别、同一重复的近重复片段没有在建库时抑制，容易让检索命中看起来很好，但实际泛化不足。
4. 旧版 source-aware patch 主要是对已建好的 JSON/NPZ 做再平衡，无法修复一开始切割出来的低质量事件边界。

V35 的改法是把 source-aware、motion-density、near-duplicate suppression 和 boundary quality 前移到 RAG 数据库切割阶段。

## 一次性替换文件

把本目录下文件复制到 EDGE 仓库根目录的对应路径：

```text
tools/v34_source_aware_rag.py
tools/v35_source_aware_segmentation.py
tools/build_v21_shared_event_index.py
scripts/launch_v35_source_aware_segmentation.sh
```

## 关键机制

### 1. Source-aware 切割

从文件名解析：

```text
dyl_002_Take_003.bvh
```

得到：

```text
dancer_id = dyl
repeat_id = 002
category_id = Take_003
source_uid = dyl_002_Take_003
```

这些字段会写入事件 JSON、共享索引 JSON 和 NPZ。

### 2. 动态边界质量

边界候选仍然来自动作低能量点，但 V35 同时看：

```text
energy valley
jerk valley
root_y zero crossing
contact switch near valley
```

边界得分：

```math
Q_{boundary}=1-\left(0.68E_t+0.32J_t\right)
```

其中 `E_t` 是归一化动作能量，`J_t` 是归一化 jerk。

### 3. 抑制“第一秒动完后面保持”

对每个 segment 计算：

```text
mean_activity
tail_mean_activity
first1s_energy_ratio
static_hold_score
```

核心惩罚：

```math
C_{hold}
=
\max(0,r_{1s}-0.65)
\cdot
\max(0,\tau_e-\bar e_{tail})
```

它不会直接删除所有 pose_hold，但会降低“前面突变、后面摆住”的片段质量权重。

### 4. 近重复抑制

同一个 `source_uid` 内，若两个事件区间的 IoU 大于阈值，保留质量更高者：

```math
\mathrm{IoU}([a,b],[c,d])=
\frac{|[a,b]\cap[c,d]|}{|[a,b]\cup[c,d]|}
```

默认阈值：

```text
V35_NEAR_DUP_IOU=0.72
```

### 5. 分布审计

输出文件会包含：

```text
source_distribution_raw
source_distribution_selected
event_type_counts
near_duplicate_suppression
source_reports
```

可以直接检查舞者、类别、重复次数是否严重倾斜。

## 建库命令

如果 151D 动作文件目录不是默认路径，先指定：

```bash
export V35_MOTION_INPUT_DIR=/home/disk/lsm/storage/EDGE/data/dunhuang_dynamic_event_rag_physical
```

只重建数据库，不训练：

```bash
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=$EDGE_ENV/bin:$PATH
export PYTHONPATH=$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

bash scripts/launch_v35_source_aware_segmentation.sh
```

注意：V35 默认只接受原始整段动作文件，例如：

```text
dyl_002_Take_003.pkl
dyl_002_Take_003.npy
dyl_002_Take_003.npz
```

它会拒绝旧数据库里已经切好的：

```text
src0003_ev0371_dyl_001_Take_003_016756_016802.pkl
```

这是为了防止“把旧 event 再切一遍”的数据库污染。如果只是调试旧 event 文件，可临时设置：

```bash
export V35_ALLOW_EVENT_FILES=1
```

正式实验不要打开这个开关。

## 完整重建 + 重训

```bash
cd /home/disk/lsm/storage/EDGE

tmux kill-session -t v35_source_aware_full 2>/dev/null || true

tmux new-session -d -s v35_source_aware_full "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V35_MOTION_INPUT_DIR=\${V35_MOTION_INPUT_DIR:-data/dunhuang_dynamic_event_rag_physical}
export V35_REBUILD_EVENT_DB=1
export V35_REBUILD_SHARED_INDEX=1
export V35_BUILD_DURATION_INDEX=1
export V35_BUILD_HIERARCHY=1
export V35_BUILD_CONTACT_CACHE=1
export V35_BUILD_TRANSITION_DATASET=1

export V35_RUN_FULL_RESEARCH=1
export V34_TRAIN=1
export V34_BUILD_EVENT_LIBRARY=1
export V34_OVERWRITE_EVENT_LIBRARY=1

export V34_SOURCE_AWARE_RAG=1
export V34_RHYTHM_DEGRADATION_PENALTY=1
export V34_BOUNDARY_COMPAT=1
export V34_COMPAT_DENSE_SCORE=1
export V34_BOUNDARY_INPAINT=1
export V34_LATENT_SNIPPET_BLEND=1

export RUN_ID=v35_source_aware_full_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p \"\$RUN_ROOT\"

bash scripts/launch_v35_source_aware_segmentation.sh 2>&1 | tee -a \"\$RUN_ROOT/outer.log\"
'"
```

## 查看进度

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V35_SOURCE_AWARE_SEGMENTATION.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -f "$RUN_ROOT/outer.log"
```

## 建库质量审计

```bash
cd /home/disk/lsm/storage/EDGE

python - <<'PY'
import json, pathlib
p = pathlib.Path("data/v35_source_aware/dynamic_event_rag/index_dynamic_event_source_aware.json")
data = json.loads(p.read_text())
print("num_events_raw", data["num_events_raw"])
print("num_events_after_dedup", data["num_events_after_dedup"])
print("num_events", data["num_events"])
print("event_type_counts", data["event_type_counts"])
print("source_selected", data["source_distribution_selected"])
print("near_duplicate_suppression", data["near_duplicate_suppression"])
PY
```

共享索引审计：

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path("data/v35_source_aware/v21_shared_event_index_source_aware.json")
data = json.loads(p.read_text())
print("num_indexed", data["num_indexed"])
print("event_type_counts", data["event_type_counts"])
print("source_distribution_indexed", data["source_distribution_indexed"])
PY
```

## 是否需要重新切分 BVH

不需要手动重新切分原始 BVH。

需要做的是：用同一批已经转换为 151D 的整段动作文件重新运行 V35 自动切割建库。V35 会重新决定 event 边界、事件质量、source-aware 下采样和近重复抑制。

## 预期效果

V35 主要改善的是 RAG 候选库质量：

1. 检索候选不再被少数舞者、少数重复、少数类别主导。
2. “第一秒快速换姿势，后面保持很久”的片段会减少进入高优先级候选池。
3. 同源近重复片段减少，训练和评估更容易站得住脚。
4. 后续 Dense Boundary、Masked Inpainting、Rhythm Repair 的输入更干净。

它不会单独替代 boundary repair、contact repair 和 motion quality postprocess。它解决的是 RAG 数据库切割和候选池偏置问题。
