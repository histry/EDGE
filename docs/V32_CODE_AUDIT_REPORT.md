# EDGE V32 代码审计：为什么不能直接恢复原始 V30，以及本次彻底升级内容

## 1. 当前仓库的真实状态

当前 `main` 的主训练器是 V31 的低阶系数扩散，不再是 V30 连续 INR。
V30 原始连续 INR 文件仍在仓库，但它不是安全的主训练链：

- Fourier bands=10，对应时间频率最高 512；
- 多层高频 Sine；
- 单关节旋转残差上限 0.40 rad；
- contact 只是无物理耦合的通道残差；
- VAE 训练使用随机 posterior，latent diffusion 使用 posterior mean；
- 推理曾以高 blend 直接接管基础转场。

因此，“彻底升级到连续 INR”不能理解为恢复原始 V30 参数；那会重新引入
已经在视频与 jitter 指标中暴露的高频自由度。

## 2. V32 底层模型

V32 保留 V30 的核心研究思想：

```text
variable-length motion
→ deterministic transition encoder
→ latent transition representation
→ conditional latent diffusion
→ continuous INR at arbitrary t
→ native EDGE [T,151] motion
```

但重构为：

```text
C2 quintic SO(3) base
+ low-band continuous SiLU INR
+ C2-zero learned residual
+ contact-logit head
+ differentiable FK/contact losses
+ multi-candidate context gate
+ deterministic C2 fallback
```

### 连续性

所有学习残差乘以：

```text
E(t)=64 t^3 (1-t)^3
```

在 t=0/1 处，残差的值、一阶导数和二阶导数均为零，不破坏端点的姿态、
速度和加速度。

### 频率限制

默认 Fourier bands=5，仅保留：

```text
1, 2, 4, 8, 16
```

不再使用最高 512 的时间频率。

### 潜变量

使用确定性 encoder，加 VICReg 风格方差、协方差和均值约束。删除 V30
VAE posterior sampling 与 diffusion posterior-mean 之间的分布错位。

## 3. 可微接触损失

V32 contact head 输出四个 contact logits。训练包含：

```text
contact BCE
contact-weighted horizontal foot skating
contact foot-height consistency
foot-ground penetration
swing-foot clearance
contact temporal consistency
contact binary regularisation
```

foot skating、height 和 penetration 通过可微 FK 反向传播到：

```text
local SO(3) rotations
root height
continuous INR
```

不是只训练 contact 分类器。

## 4. 三阶段训练

### Stage A：连续 INR Autoencoder

混合使用：

- intra_event_real；
- source_gap_real；
- source_boundary_mask_real；
- 可选低权重 synthetic adjacent。

优化姿态、FK、速度、加速度、jerk、角速度、端点、频域和接触损失。

### Stage B：Differentiable Contact Fine-tuning

只使用 `real_target=True`，冻结 encoder、condition encoder 和 diffusion，
仅小学习率微调 INR，强化 foot skating、contact height 和 penetration。

### Stage C：Latent Diffusion

冻结 encoder 和 INR，训练 latent diffusion。除 latent noise/x0 loss 外，
对少量 decoded samples 再计算 SO(3)、FK 与 contact loss，使 checkpoint
选择不只依赖 latent loss。

## 5. 推理链修复

旧调度器只向 sampler 提供起止单帧，并在生成后再次覆盖 root rotation。
V32：

- 捕获前一事件末 4 帧和后一事件前 4 帧；
- 在真实上下文中计算 entry/exit 风险；
- 生成后不再后置覆盖 root rotation；
- 每个边界采样多个候选；
- 检查 entry、exit、jerk、foot slip、penetration 和 rotation step；
- 不优于 C2 baseline 时自动回退。

## 6. 当前数据局限

当前自动审计结果是：

```text
72 source groups
0 resolved full sequences
4225 cropped events
```

因此当前可以运行 `weak` 模式：

- 真实 intra-event masked windows 是主要监督；
- 少量 synthetic adjacent 仅作跨事件条件正则；
- contact fine-tuning 只使用真实事件内部运动。

但不能据此声称学习了真实 event-to-event transition。只有找回 72 个完整
长序列及真实边界后，才能用 `strict` 模式解决 Ouroboros 数据闭环。

## 7. 论文中必须区分

```text
V32-Weak:
cropped-event supervision + low-weight synthetic bridges

V32-Strict:
full-sequence real boundary masks, synthetic off
```

两者不能混写为同一个 full model。
