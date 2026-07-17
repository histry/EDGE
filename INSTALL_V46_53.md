# V46.53 直接替换包安装说明

## 1. 适用基线

本补丁按公开仓库 `histry/EDGE` 的 V46.52 提交
`6acc666a013f77b381cc589970555df55bf40acc` 设计，并兼容本地
V46.52.3 的足部 joint-center proxy 门限调整。

安装程序不会替换 `tools/v46_motionrag_diff.py`、重定向基础实现或
V46.52 Anatomy Contract。它只新增 V46.53 模块，并替换三个小型入口文件：

```text
tools/v46_50_build_event_heading_db.py
tools/v46_51_heading_closed_loop.py
scripts/run_v46_50_full_rebuild_retrain.sh
```

## 2. 安装

```bash
cd /path/to/EDGE_V46_53_RESEARCH_REPLACEMENT
bash install_v46_53_patch.sh /home/disk/lsm/storage/EDGE
```

安装程序会：

1. 检查 V46.52 基础文件；
2. 将被替换入口备份到 `backup/v46_53_<timestamp>/`；
3. 复制新增/替换文件；
4. 执行 Python 语法检查；
5. 运行 `test_v46_53_*.py` 单元测试。

## 3. 正式全量重建、重训和生成

```bash
cd /home/disk/lsm/storage/EDGE

export ROOT_DIR=$PWD
export V46_53_FULL_REBUILD=1
export CHANGE_BVH_DIR=$PWD/change

bash scripts/run_v46_50_full_rebuild_retrain.sh
```

`configs/v46_53_research.env` 默认开启：

```text
重建 retarget cache
重建 source-disjoint Event-DB
训练双分支 Grounder
重训 V44
重训 V45
重训 V46
全局熵正则 Event 路径预排序
双向切空间边界风险
时间×关节局部掩码
Anatomy/KBO 回滚
```


## 3.1 视频时长合同

本项目**不使用固定 120 秒或其他固定时长**。输出动作帧数由当前输入音乐决定：

```text
T_audio = 当前 WAV 的真实时长
N_audio = round(T_audio × FPS)
N_output = Σ slot.target_frames ≈ N_audio
```

V46.51 Fresh-WAV Schedule Contract 先保证当前音乐与 slot 时间线一致，
V46.53 在生成结束后再次检查输出 NPY 帧数。默认允许 2 帧误差，超出即终止，
并生成：

```text
<output>.npy.v46_53_duration.json
```

可配置开关：

```bash
export V46_53_ENFORCE_DYNAMIC_DURATION=1
export V46_53_OUTPUT_FRAME_TOLERANCE=2
```

## 4. 仅验证数据库，不重训 V44/V45/V46

```bash
export V46_53_FULL_REBUILD=0
export V46_51_REBUILD_RETARGET_CACHE=1
export V46_51_REBUILD_EVENT_DB=1
export V46_51_RETRAIN_V44=0
export V46_51_RETRAIN_V45=0
export V46_51_RETRAIN_V46=0

bash scripts/run_v46_50_full_rebuild_retrain.sh
```

已有 V44/V45/V46 checkpoint 必须仍能由原 V46.51 launcher 找到，否则基础脚本会按合同终止。

## 5. 关键环境开关

| 开关 | 默认 | 含义 |
|---|---:|---|
| `V46_53_GROUNDER_ENABLE` | 1 | 开启双分支接地模型 |
| `V46_53_TRAIN_GROUNDER_ON_BUILD` | 1 | 在 train source DB 构建后训练 |
| `V46_53_GLOBAL_ROUTE_ENABLE` | 1 | 开启全曲熵正则路径预排序 |
| `V46_53_BODY_PART_MASK_ENABLE` | 1 | 开启时间×关节掩码 |
| `V46_53_FULL_ROLLBACK_ON_FAIL` | 1 | 局部投影失败后回退参考动作 |
| `V46_52_ALLOW_UNSAFE_RESCUE` | 0 | 禁止最低违规候选强行通过 |
| `V46_53_GROUNDER_STEPS` | 1600 | 双分支接地训练步数 |
| `V46_53_GLOBAL_ROUTE_BEAM` | 32 | 全曲路径 beam 宽度 |
| `V46_53_GLOBAL_ROUTE_TOPK` | 20 | 每个 slot 的全局候选数 |
| `V46_53_ENFORCE_DYNAMIC_DURATION` | 1 | 强制输出时长由当前音乐决定 |
| `V46_53_OUTPUT_FRAME_TOLERANCE` | 2 | 输出与音乐调度允许的帧误差 |

## 6. 输出

正式运行目录包含：

```text
v46_53_dual_branch_grounder.pt
retarget_cache/
event_db_split/train/events.npz
event_db_split/val/events.npz
event_db_split/test/events.npz
*.v46_53_geometry.audit.json
*.v46_53_grounding.json
v46_51_final.npy
final.v46_52_anatomy.json
final.v46_53_intrinsic.json
<output>.npy.v46_53_duration.json
```

Event-DB 新增字段以 `v46_53_` 为前缀，旧 V44/V45/V46 需要的字段保持不变。

## 7. 科研模式注意事项

- `V46_53` 的全局路径是“Schrödinger-inspired entropy-regularised discrete path prior”，不是连续 Schrödinger Bridge 的完整数值求解器，论文中不得夸大。
- 脚部 `-0.08 m` 是 joint-center proxy 的 source screening 门限，不是脚底网格允许穿地量。
- 双曲分支只表达 AESD/family/posture 层级；人体旋转仍由 `SO(3)^24` 表达。
- V45/V46 的输出在最终写回前会由时间×关节掩码限制，非风险 Event core 保持参考动作。
