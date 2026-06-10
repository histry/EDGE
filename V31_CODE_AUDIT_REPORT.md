# EDGE 最新代码审计报告：V29/V30 抖动根因与 V31 修复

## 结论

上传的 V30 代码并没有解决全部列出的审稿风险，并且包含会直接影响结果的
确定性代码问题。上一版结果更加抖动具有明确的代码原因。

## 已确认的高风险代码

| 位置 | 风险 | 后果 |
|---|---|---|
| `tools/v30_continuous_inr.py` | Fourier bands=10，时间频率达到 2^9；多层 omega=24 Sine | 可表达远高于人体运动所需的时间频率 |
| `tools/v30_continuous_inr.py` | rotation residual cap=0.40 rad | 单关节单时刻残差过大 |
| `tools/v27_transition_diffusion.py` | INR blend 默认 0.85 | 未验证模型大幅接管稳定基线 |
| `tools/v27_transition_diffusion.py` | 单候选直接采用，无安全回退 | 随机潜变量的坏样本直接进入整曲 |
| `train_v27_transition_diffusion.py` | VAE sampled latent 重构、posterior mean 训练 diffusion | 潜变量训练/推理分布错位 |
| `train_v27_transition_diffusion.py` | diffusion checkpoint 以 latent loss 为主 | 低 latent loss 不等于低 FK jerk |
| `tools/build_v27_transition_diffusion_dataset.py` | synthetic adjacent 默认开启 | 合成桥接继续主导边界分布 |
| 同上 | external prior 标为 real_target | 真实监督比例统计失真 |
| 同上 | source minimum length 加 max_len | 有效序列尾部真实边界被误拒绝 |
| `tools/schedule_v30_whole_song.py` | 最近规则查询匹配几何条件 | 重复短语可能条件错位 |
| `tools/evaluate_v30_frequency_metrics.py` | `_spectrum_power()` 自递归 | DCT 绘图路径无限递归 |
| `tools/v26_global_duration_alignment.py` | lock boundaries 时只报告 warp 超限，不阻止 | 内容可被强行压缩/拉伸并产生抖动 |

## V31 修复原则

1. 确定性 C² SO(3) 基线永远存在；
2. 学习模块只生成低阶、端点二阶归零残差；
3. 不生成接触标签；
4. 多候选采样；
5. 使用真实前后动作上下文评价入口、内部和出口；
6. 候选不优于基线则自动回退；
7. 主模型禁用 synthetic adjacent 与 pseudo pair；
8. 真实边界按 unique pair 计数；
9. 锁定音乐边界但自然时长超限时直接失败；
10. 双曲检索默认关闭，必须通过留出集审计后才能开启。

## 不能由代码自动解决的问题

- 没有完整长动作序列时，无法消除 Ouroboros 数据闭环；
- LODGE++、OpenDanceNet、MotionRAG-Diff 需要独立公平复现；
- 双曲空间是否优于欧氏空间必须由留出集和多随机种子证明；
- “顶会/顶刊”取决于数据、实验、统计和写作，不由版本号保证。
