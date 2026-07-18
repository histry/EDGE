# EDGE V46.53.1 科研合同修复方案

## 1. 适用基线

本替换包面向 `histry/EDGE` 最新 V46.53 代码基线：

```text
2873b96730814f2c3e7956b3e73d0bff0c43588b
```

它不修改 `change/` 原始 BVH，不复用旧的失败重定向缓存，允许重新构建
source-disjoint Event-DB，并重训 V44、V45、V46 和 V46.53 Grounder。

## 2. 代码层问题与修复

### 2.1 源级安全与事件级质量混淆

旧代码以整段 source 的普通局部角度越界比例执行一票否决，导致少量异常窗口
使整个 BVH 及其全部有效事件丢失。新方案将合同解耦为：

```text
Source Safety Gate
  只拒绝非有限值、旋转退化、严重角度超限、灾难性碰撞、骨长漂移、
  严重地面/重力错误和不可接受拟合误差

Posture-Aware Event Quality Gate
  在事件切片后按 standing / squat / kneeling / floor_pose / aerial
  分别评估局部角度、压缩、碰撞与连续质量
```

### 2.2 Root 全锁定与局部关节补偿冲突

新重定向器取消 Root orientation 完全锁定，改为 SO(3) 测地软锚定，使小范围
Root roll/pitch/body-frame 修正可以由 Root 承担，避免全部误差压入脊柱、髋和肩。

### 2.3 Rot6D 欧氏时间损失与 SO(3) 审计不一致

新优化器使用：

```text
omega_t = Log(R_t^T R_{t+1}) * FPS
alpha_t = (omega_t - omega_{t-1}) * FPS
```

替代 Rot6D 列向量直接差分。重叠窗口不再平均 Rot6D，而是加权平均旋转矩阵后
通过 SVD 投影回 SO(3)。

### 2.4 固定直立模板误伤敦煌舞低姿态

upright/head/feet 约束改为源结构引导：以当前 BVH 的头—骨盆和足—骨盆关系
作为目标下界，而不是把所有盘坐、击鼓、琵琶和大幅侧屈动作拉回统一站立模板。

### 2.5 小样本 source split 产生 empty_test

新 splitter 先计算全局精确容量，再进行标签感知分配。对 12 个 source 和
0.67/0.165/0.165 比例固定得到：

```text
train = 8
val   = 2
test  = 2
```

所有划分非空，且 source_uid 在三个集合间严格不重叠。

### 2.6 配置重复加载覆盖外部开关

`v46_52_anatomy_research.env` 和 `v46_53_research.env` 被替换为兼容桥接文件，
无论基础脚本重复 source 多少次，最终都会重新应用 `v46_53_1_research.env`。
科研开关统一使用 `V46_53_1_*` 前缀。

## 3. 与最新论文方法逻辑的对应

论文中的“异质信息解耦—结构先验协同—多维联合约束—动态正则”迁移为：

```text
退化噪声分支 / 结构语义分支
        ↓ 方法论迁移
灾难性 source safety / 姿态相关 event quality

结构引导协同注意力
        ↓
源 body-frame、姿态类别和解剖质量引导的约束调度

多维联合损失
        ↓
关键点 + Root + SO(3)速度/加速度 + Anatomy + Floor + Fit

动态 TV 权重衰减
        ↓
前期结构先验较强，粗对齐后 Anatomy 权重平滑升高，姿态先验逐步衰减
```

这里迁移的是优化逻辑，不把图像频域算子机械复制到骨架动作中。

## 4. 替换文件

```text
configs/v46_52_anatomy_research.env
configs/v46_53_research.env
configs/v46_53_1_research.env
scripts/run_v46_50_full_rebuild_retrain.sh
scripts/run_v46_53_1_research.sh
tools/v46_49_audit_gravity_contract.py
tools/v46_52_anatomy_contract.py
tools/v46_52_build_retarget_cache.py
tools/v46_51_split_retarget_cache.py
tools/v46_52_filter_event_db.py
tools/v46_53_1_anatomy_retarget.py
tools/v46_53_1_preflight.py
tests/test_v46_53_1_contract.py
tests/test_v46_53_1_rotation_fusion.py
tests/test_v46_53_1_split.py
```

现有 V46.53 几何、Grounder、全局路径、边界风险和动态时长代码保持不变。

## 5. 安装

```bash
cd /path/to/EDGE_V46_53_1_RESEARCH_CONTRACT_PATCH
bash install_v46_53_1_patch.sh /home/disk/lsm/storage/EDGE
```

安装程序会备份所有同名文件，然后执行 Python 编译、Shell 语法检查和单元测试。

## 6. 全量重建、重训和生成

```bash
cd /home/disk/lsm/storage/EDGE
bash scripts/run_v46_53_1_research.sh \
  "$PWD/test_music_bank/dunhuangwu2.wav"
```

默认设置：

```text
change/                                重新重定向
source split                           重新划分
Event-DB                               重新构建
V46.53 Grounder                        重新训练
V44 / V45 / V46                       重新训练
MUSIC_DIRS                             data/v21_router_music_999/splits/train
整曲测试音乐                          test_music_bank/dunhuangwu2.wav
输出帧数                              由当前音乐真实时长决定
```

## 7. 常用环境开关

完整重建：

```bash
export V46_53_1_REBUILD_RETARGET_CACHE=1
export V46_53_1_REBUILD_EVENT_DB=1
export V46_53_1_RETRAIN_V44=1
export V46_53_1_RETRAIN_V45=1
export V46_53_1_RETRAIN_V46=1
```

复用已经通过的新重定向缓存：

```bash
export V46_53_1_REBUILD_RETARGET_CACHE=0
```

改变最低合格 source 数量：

```bash
export V46_53_1_MIN_OK_SOURCES=8
```

所有外部覆盖应优先使用 `V46_53_1_*` 名称，避免旧 shell 中残留的 V46.52 参数
意外恢复旧合同。

## 8. 科研边界

- Source gate 的放宽不是无条件接收：严重旋转、骨架退化、非有限值、碰撞、
  重力和拟合错误仍然硬拒绝。
- Event rescue 只能恢复 hard-valid 且质量超过安全下限的事件，不能绕过硬合同。
- 本包通过静态检查和单元测试，但真实 12-source 全量训练需要在项目 RTX 4090
  环境中执行后，才能报告最终通过数、Event 数量、模型损失和整曲视频质量。
