# EDGE V34 Motion Quality Repair Replacement

## 目标问题

本包针对 `v34_rhythm_repair_infer_20260622_154221` 仍然暴露出的两个问题：

1. 前 1 秒快速换到新姿势，后续 slot 动作密度不足；
2. 接触脚在支撑态仍有水平滑动，渲染固定相机不够稳定。

根因不是单纯训练 epoch 不够。代码层面主要有三点：

- 调度器虽然已有 hold/density/streak penalty，但 `mean_energy` 会被第一秒高能量抬高，导致“前动后停”的片段漏判；
- Dense Boundary 使搜索器偏好物理安全片段，低资源库中安全片段常常也是低动态片段；
- 接触标签只参与评分和训练，不等于输出世界坐标被零速度约束锁死。

## 替换/新增文件

将本目录中的文件复制到 EDGE 根目录同名路径：

```text
tools/v34_source_aware_rag.py                  新增/保留
tools/build_v34_source_aware_index.py          新增/保留
tools/build_v21_shared_event_index.py          替换
tools/v34_warp_aware_retrieval.py              替换
tools/v34_motion_quality_postprocess.py        新增
scripts/launch_v34_source_aware_rag.sh         替换
scripts/launch_v34_motion_quality_repair.sh    新增
scripts/resume_v34_inference_v33ckpt.sh        替换
scripts/run_v34_full_research.sh               替换
render_from_npy.py                             替换
vis.py                                         替换
```

## 新增机制

### 1. 持续运动密度约束

`tools/v34_warp_aware_retrieval.py` 新增/强化以下指标：

```text
first1s_energy_ratio
late_energy_ratio
tail_to_mean_energy
low_energy_fraction
```

新增惩罚项：

```text
V34_FRONTLOAD_PENALTY_WEIGHT
V34_TAIL_RATIO_PENALTY_WEIGHT
V34_COVERAGE_PENALTY_WEIGHT
V34_LOW_ENERGY_PENALTY_WEIGHT
```

它们解决旧版漏判：如果第一秒能量占比极高，即使均值能量不低，也会被 frontload penalty 和 late coverage penalty 压下去。

### 2. 静态片段连续惩罚

默认把：

```text
V34_STATIC_STREAK_ALLOW=1
V34_STREAK_PENALTY_WEIGHT=3.00
```

即 `pose_hold / calm_flow / neutral_flow` 可以偶尔出现，但不允许连续堆叠造成 8-15 秒观感静止。

### 3. 接触脚 root-lock 后处理

新增：

```text
tools/v34_motion_quality_postprocess.py
```

它读取 `*_v26.npy` 的 contact channels，找到连续接触脚段，把 root 的 X/Z 平移反向校正，使支撑脚尽量固定在该接触段的中位锚点上。

默认不会覆盖原始文件，而是生成：

```text
*_v26_motion_quality.npy
*.motion_quality_postprocess.json
```

`scripts/run_v34_full_research.sh` 会优先对修复后的 motion 做 metrics 和 scientific render。

### 4. 固定渲染坐标盒

`vis.py` 新增：

```text
EDGE_RENDER_FIXED_BOUNDS=1
EDGE_RENDER_XLIM=-1.8,1.8
EDGE_RENDER_YLIM=-1.8,1.8
EDGE_RENDER_ZLIM=-0.05,2.25
```

它用于消除 Matplotlib 3D box 的视觉缩放干扰。该改动只影响展示，不改变 motion 数据。

## 推荐纯推理命令

```bash
cd /home/disk/lsm/storage/EDGE

python -m py_compile \
  tools/v34_source_aware_rag.py \
  tools/build_v34_source_aware_index.py \
  tools/v34_warp_aware_retrieval.py \
  tools/v34_motion_quality_postprocess.py \
  render_from_npy.py \
  vis.py

bash -n \
  scripts/launch_v34_source_aware_rag.sh \
  scripts/launch_v34_motion_quality_repair.sh \
  scripts/resume_v34_inference_v33ckpt.sh \
  scripts/run_v34_full_research.sh

tmux kill-session -t v34_motion_quality_repair 2>/dev/null || true

tmux new-session -d -s v34_motion_quality_repair "bash -lc '
set -euo pipefail
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=\$EDGE_ENV/bin:\$PATH
export PYTHONPATH=\$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V34_TRAIN=0
export V34_SOURCE_AWARE_REBUILD=0
export V34_SOURCE_AWARE_REBUILD_HIERARCHY=0

export V34_RHYTHM_DEGRADATION_PENALTY=1
export V34_RHYTHM_WEIGHT=1.20
export V34_HOLD_FIRST1S_RATIO_LIMIT=0.55
export V34_HOLD_TAIL_ENERGY_LIMIT=0.024
export V34_MIN_SLOT_MEAN_ENERGY=0.018
export V34_LATE_ENERGY_RATIO_MIN=0.34
export V34_TAIL_MEAN_RATIO_MIN=0.55
export V34_LOW_ENERGY_FRACTION_MAX=0.62
export V34_FRONTLOAD_PENALTY_WEIGHT=4.50
export V34_TAIL_RATIO_PENALTY_WEIGHT=3.00
export V34_COVERAGE_PENALTY_WEIGHT=2.75
export V34_STATIC_STREAK_ALLOW=1
export V34_STREAK_PENALTY_WEIGHT=3.00

export V34_MOTION_QUALITY_POSTPROCESS=1
export V34_CONTACT_LOCK_POSTPROCESS=1
export V34_CONTACT_LOCK_STRENGTH=0.85
export V34_OUTPUT_SMOOTH=1

export RUN_ID=v34_motion_quality_repair_\$(date +%Y%m%d_%H%M%S)
export RUN_ROOT=output/\$RUN_ID
mkdir -p \"\$RUN_ROOT\"

bash scripts/launch_v34_motion_quality_repair.sh 2>&1 | tee -a \"\$RUN_ROOT/outer.log\"
'"

sleep 5
RUN_ROOT=$(cat output/LATEST_V34_MOTION_QUALITY_REPAIR.txt)
echo "RUN_ROOT=$RUN_ROOT"
tail -80 "$RUN_ROOT/outer.log"
```

## 完整长训

如果要在 source-aware transition dataset 上完整训练：

```bash
cd /home/disk/lsm/storage/EDGE

export EDGE_ROOT=/home/disk/lsm/storage/EDGE
export EDGE_ENV=/home/disk/lsm/conda_envs/edge
export PATH=$EDGE_ENV/bin:$PATH
export PYTHONPATH=$EDGE_ROOT
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

export V34_TRAIN=1
export V34_SOURCE_AWARE_REBUILD=0
export V34_SOURCE_AWARE_REBUILD_HIERARCHY=0

bash scripts/launch_v34_motion_quality_repair.sh
```

## 检查输出

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT=$(cat output/LATEST_V34_MOTION_QUALITY_REPAIR.txt)

find "$RUN_ROOT" -type f \
  \( -name "*motion_quality.npy" \
     -o -name "*.motion_quality_postprocess.json" \
     -o -name "*.schedule_report.json" \
     -o -name "*.contact_metrics.json" \
     -o -name "*.jitter.json" \
     -o -name "*.mp4" \) | sort

grep -nE "rhythm_degradation|frontload|tail_ratio|motion_quality|PASS|DONE|Traceback|RuntimeError|ERROR" \
  "$RUN_ROOT/run.log" "$RUN_ROOT/outer.log" 2>/dev/null | tail -200
```

## 科研表述

这次改动可以定义为第三篇长程调度问题的工程落地：

> Dense Boundary 解决了片段接缝的物理突变，但会暴露低资源 Event-RAG 的安全片段偏置。搜索器倾向于选择边界风险低、动作密度低、前置能量释放过快的片段。为此，本文引入 motion-density preserving retrieval field，通过 frontload risk、late-energy coverage、tail-to-mean ratio 与 static streak penalty，把全局搜索轨迹从安全静态流形推回持续运动流形；同时利用 contact-aware root locking 在输出层约束支撑脚零速度，降低 foot skating。
