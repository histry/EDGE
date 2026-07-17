# V46.53 科研重构说明

## 1. 设计取舍

本补丁没有把所有讨论过的概念机械堆入代码，而选择对当前低资源 EDGE 项目最有效、可复现且可消融的组合：

```text
双分支事件表示
+ 双曲语义层级
+ SO(3) 内蕴运动流
+ Anatomy-Gated Event-DB
+ Wasserstein-2 source 重心
+ 多层概率接地
+ 熵正则全曲 Event 路径
+ 双向切空间边界风险
+ 时间×关节局部修复
+ 事务化回滚
```

完整连续 Schrödinger Bridge、完整黎曼扩散、在线持续学习和超网络生成整套扩散参数没有加入正式替换包，因为 12-source 低资源条件下，它们会显著提高过拟合和不可复现风险。

## 2. 与最新论文逻辑的迁移关系

最新上传论文的核心逻辑是“不同性质信息不应在单一分支中混合处理，而应先解耦，再由结构先验协同融合，并采用多维联合约束”。在动作项目中对应为：

| 图像论文思想 | EDGE 中的重构 |
|---|---|
| 梯度—频域双分支 | 语义层级分支—SO(3) 内蕴动态分支 |
| 结构引导协同注意力 | Anatomy/structure quality 引导的双分支门控 |
| 多维联合损失 | 多正样本、双曲层级、source-invariance、Jigsaw、动态一致性 |
| 动态 TV | 随训练衰减的一致性正则和推理阶段高阶运动风险 |
| 保留线条核心、局部抑噪 | 冻结 Event core，仅修复风险帧×风险关节 |

该迁移保留了方法论，而没有将图像梯度或频率算子不加区分地复制到骨架动作中。

## 3. 数据合同

EDGE 151D 不变：

```text
[0:4] contacts
[4:7] root XYZ
[7:151] 24 × column-concatenated Rot6D
```

新字段均为 Event-DB 派生数据，不改变训练数据主维度。

## 4. V46.53 六个执行层


### 4.0 Audio-Derived Dynamic Duration Contract

整曲生成不预设 120 秒或其他固定视频长度。当前 WAV 经 Fresh-WAV 分析得到
连续、无重叠且守恒的 slot 时间线，最终动作帧数满足：

```text
N_output = Σ_i N_slot_i ≈ round(T_audio × FPS)
```

V46.53 在生成后执行第二次时长审计，检查输出帧数、调度目标帧数与音乐期望帧数。
因此，同一模型可以处理不同长度音乐，视频时长随输入音乐自动变化。

### 4.1 Geometry Contract

`geometry/v46_53_rotation_contract.py`

- 统一 column-concat Rot6D；
- SVD 投影到合法 `SO(3)`；
- Log/Exp、测地距离；
- 角速度、角加速度；
- 切空间局部融合。

### 4.2 Intrinsic Event Geometry

`tools/v46_53_event_geometry.py`

- Root、身体部位角速度和角加速度；
- 手腕、踝、手部长期轨迹；
- 支撑事件；
- structure quality；
- quality-weighted diagonal Gaussian W2 barycenter；
- shared/private source 表示。

### 4.3 Dual-Branch Grounding

`tools/v46_53_grounding.py`

- 语义分支；
- 内蕴几何分支；
- Poincaré ball 层级分支；
- structure-guided gate；
- 多正样本对比；
- source-invariance；
- shuffled-geometry Jigsaw；
- 动态一致性权重。

### 4.4 Global Event Path

`tools/v46_53_heading_closed_loop.py`

- 先用接地分数、质量和 Anatomy 构建 unary；
- 用角运动流、Posture、Root-Y 和 Contact 构建 transition energy；
- 通过全曲 beam path 预排序候选；
- 再交由 V46.52 的真实拼接模拟和硬合同重选。

### 4.5 Tangent Boundary Risk

`tools/v46_53_boundary_contract.py`

- forward/entry 动态方向对齐；
- SO(3) pose gap；
- angular velocity/acceleration；
- 身体部位切空间 Gaussian Bures-Wasserstein；
- Root-Y、Root velocity 和 Contact；
- 硬拒绝与风险原因。

### 4.6 Surgical Repair

现有 V45/V46/IK 生成 proposal 后：

```text
seam mask
→ 动态风险
→ frame×joint mask
→ tangent masked merge
→ EDGE contract
→ Anatomy contract
→ commit / rollback
```

## 5. 直接替换文件

```text
tools/v46_50_build_event_heading_db.py
tools/v46_51_heading_closed_loop.py
scripts/run_v46_50_full_rebuild_retrain.sh
```

## 6. 新增文件

```text
geometry/__init__.py
geometry/v46_53_rotation_contract.py
tools/v46_53_event_geometry.py
tools/v46_53_grounding.py
tools/v46_53_boundary_contract.py
tools/v46_53_duration_contract.py
tools/v46_53_build_event_db.py
tools/v46_53_heading_closed_loop.py
configs/v46_53_research.env
tests/test_v46_53_rotation_contract.py
tests/test_v46_53_event_geometry.py
tests/test_v46_53_boundary_contract.py
```

## 7. 不替换的关键文件

```text
tools/v46_motionrag_diff.py
tools/chang_e_edge_retarget.py
tools/v46_52_anatomy_contract.py
tools/v46_52_anatomy_retarget.py
tools/v46_52_build_retarget_cache.py
tools/v46_50_heading_closed_loop.py
```

这些文件已经承载成熟训练、重定向和回滚能力。直接重写大文件会增加回归风险，因此 V46.53 通过小型 wrapper 和可测试模块进行扩展。
