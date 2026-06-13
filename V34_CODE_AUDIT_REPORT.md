# EDGE V34 代码审计与架构决策报告

## 1. 结论

用户提出的四项思路可以采用，而且是当前系统最合理的下一步。但需要做四点科研化修正：

1. Contact 不能直接覆盖原始 V21 数据，应构建带指纹校验的镜像事件库，保留原库可追溯性。
2. 五次 Hermite 足以匹配姿态、速度和加速度；只有在进一步匹配 jerk 时才需要七次多项式。V34 使用带收缩和幅值上限的七次 SO(3) 切空间路径，避免把噪声三阶差分原样写入轨迹。
3. 绝对 jerk 阈值不能长期手填为 5000；应由未编辑的敦煌动作事件内部连续边界的高分位数校准。
4. Exit Handshake 必须同时检查入口和握手尾部，否则可能只是把突变从转场出口移动到 8～12 帧之后。

V34 因此定义为：

```text
Event-level Contact Back-injection
+ Warp-aware Event-RAG Hard Pruning
+ Regularised Septic SO(3)/Root Boundary Path
+ C3-zero Continuous INR Residual
+ Cross-boundary Absolute Safety Gate
+ Adaptive Length-preserving Exit Handshake
```

## 2. 当前 V33 剩余问题

V33 已经消除了持续性高频抖动，但评估仍出现少量极高脉冲：

```text
joint jerk P95 正常
joint jerk max 达到 70000+
最坏帧集中在 transition end
```

这说明问题从高频噪声转为边界冲击。

### 2.1 五次基线路径的二阶边界错位

当前基线路径匹配端点姿态与速度，并把端点二阶导数固定为零。后一真实事件经过 V23 time-warp 后通常具有非零初始加速度，因此：

```text
transition end acceleration ≈ 0
following event initial acceleration != 0
```

三阶差分会在拼接帧产生尖峰。

### 2.2 安全门只相对比较

旧安全门主要判断候选是否比 C2 基线更差。若基线本身因为极端 warp 或边界动力学不一致而不安全，相对更优仍可能绝对不安全。

### 2.3 转场内部指标遗漏出口跨界步

旧 `max_rotation_step` 和主要 jerk 统计位于 transition 内部，没有完整覆盖：

```text
transition[-2:]
following[:3]
```

因此“转场内部平滑、出口单帧突变”可以漏过门控。

### 2.4 Contact 训练—推理错位

V33 事件级 Contact 只进入训练 NPZ；整曲调度仍从原 V21 事件文件读取前四维全零 Contact。后果包括：

- start/end contact logit 约为 -9.21；
- Contact-INR 难以翻转为有效支撑概率；
- foot-slip 风险门控接近失明；
- 训练时的非零接触条件与推理时的零接触条件不一致。

### 2.5 Warp 约束出现得太晚

V26/V32 只把自然时长作为软分数，候选确定后才计算真实 transition length 和 locked-slot warp。极端候选可能进入最终 beam，然后在事后 strict gate 失败，或在诊断模式中产生 0.33 倍时长压缩。

## 3. V34 代码方案

### 3.1 镜像 Contact 事件库

新增：

```text
tools/build_v34_contact_event_library.py
```

流程：

```text
原始 V21 index
+ V33 event contact cache
→ event id/path/length/fingerprint 严格核验
→ 仅覆盖 [0:4] Contact
→ 每事件独立 NPY 镜像
→ 新 V34 index JSON
→ read-back 审计
```

原始 root、rotation 和原始事件库保持不变。

### 3.2 经验阈值校准

新增：

```text
tools/calibrate_v34_boundary_thresholds.py
```

从自然事件内部连续帧采样，估计：

- cross-boundary joint jerk；
- angular jerk；
- entry/exit rotation step；
- entry/exit FK jump；
- exit acceleration。

阈值由高分位数乘安全系数产生，而不是固定人工常数。

### 3.3 正则化七次边界路径

新增：

```text
tools/v34_boundary_dynamics.py
```

七次多项式具备 8 个系数，约束：

```text
p(0), p(1)
v(0), v(1)
a(0), a(1)
j(0), j(1)
```

SO(3) 在相对旋转切空间内求解；root 在欧氏空间求解。由于三阶差分噪声大，V34 默认：

```text
V34_JERK_MATCH_SHRINK=0.35
```

并对速度、加速度、jerk 切向量设置上限。因此应称为“regularised C3 boundary model”，不能夸大为对所有真实动力学的无条件精确恢复。

### 3.4 C3-zero INR 残差

学习残差改为：

```text
E(t) = 256 t^4 (1-t)^4
```

残差在两端的值、一阶、二阶和三阶导数为零，避免网络重新破坏基线路径的边界动力学。

### 3.5 跨界绝对安全门

重写：

```text
tools/v32_transition_quality.py
```

评价窗口：

```text
previous[-4:] + transition + following[:4]
```

候选必须同时满足：

```text
相对基线风险
AND
自然动作分布校准的绝对阈值
```

foot-slip contact gate 使用：

```text
max(predicted contact, kinematic contact proxy)
```

防止模型通过输出“无接触”规避滑步惩罚。

### 3.6 自适应 Exit Handshake

在不改变整曲帧数的情况下，用 8/10/12/16/20 帧候选桥接下一事件开头。每个候选同时评价：

1. transition → handshake 开头；
2. handshake 尾部 → 原事件未修改部分。

选择绝对安全且风险最低的长度。若严格论文模式下所有候选失败，系统直接中止，不输出伪安全结果。

### 3.7 Warp-aware 检索

候选的真实 warp 在候选特定 transition length 确定后计算：

```text
exact content length = slot length - transition length
warp = exact content length / natural duration
```

严格模式中，超出 `[0.82, 1.30]` 的候选不会进入 beam。

## 4. 是否需要重新训练

### 兼容诊断

现有 V33 checkpoint 参数形状兼容 V34，可直接运行：

```text
launch_v34_inference_from_v33.sh
```

用途是快速验证 Contact 回注、Warp prune、绝对门和 Handshake。

### 正式论文实验

建议完整重训，因为残差学习目标已经从 quintic/C2 基线变为 septic/C3 基线，并新增：

- endpoint acceleration loss；
- endpoint jerk loss；
- C3-zero residual envelope。

正式主实验应使用：

```text
launch_v34_full_overnight.sh
```

## 5. 当前仍不能夸大的内容

1. V33 Contact 是 event-level kinematic pseudo-contact，不是人工 Contact Ground Truth。
2. 当前 transition dataset 仍是 V34-Weak：真实跨事件 gap 数为零。
3. 七次插值和 Handshake 降低边界脉冲，需要真实视频与指标验证，不是理论上自动保证。
4. 绝对阈值是数据集经验阈值，不是人体运动的通用牛顿力学常数。
5. 旧 V33 checkpoint 的 V34 推理只属于兼容消融，不是最终 V34 full model。
