# EDGE Dunhuang Choreography Progress Notes

当前项目目标是把 EDGE 的音乐条件扩散框架改造成可控的敦煌舞数字编舞原型：输入起止/中间关键姿态、背景音乐和 2D 空间轨迹，输出 151 维 SMPL motion tensor，并渲染为 3D 骨架视频。

## 当前可汇报能力

1. 姿态控制：支持起始、终止和多中间关键帧软约束，用于引导敦煌舞标志性姿态之间的过渡。
2. 轨迹控制：支持 2D X/Z 轨迹条件，并在推理后处理阶段对 root 轨迹进行锚定或优化。
3. 音乐使用方式：不声称在敦煌数据上学习了真实音乐-动作配对；当前采用“继承预训练音乐先验 + 推理阶段节拍锚点引导”。
4. 稳定性处理：增加局部姿态尖峰滤波、旋转平滑、可选 root 轨迹微平滑和 follow camera 平滑，用于减少长序列拼接后的突变。
5. 定量评估：新增 `eval_quantitative.py`，支持关键帧误差、轨迹误差、脚滑率和 BeatAlign，指标定义见 `docs/evaluation_metrics.md`。
6. Proxy music 弱配对：新增 `build_proxy_weak_pairs.py`，按 BPM、beat density 和动作重音密度生成弱配对候选，操作步骤见 `docs/proxy_music_weak_pairing_steps.md`。

## 音乐相关 claim

由于当前没有真实的敦煌音乐-动作配对数据集，音乐部分的表述统一为：

模型保留原 EDGE/AIST 的扩散生成框架、动作生成先验和 audio-conditioned 接口结构。需要注意的是，当前敦煌版本使用 803 维 hybrid audio feature（Wav2Vec2 + Librosa）。当该特征维度与原 checkpoint 的音频投影层不一致时，相关音频投影层会重新初始化，因此当前不声称完整继承了原模型的音乐编码能力。

敦煌阶段主要训练姿态风格、关键帧过渡、轨迹控制和物理连续性；音乐部分采用 proxy music 弱节奏候选和推理阶段 onset/beat 锚点引导，不把 proxy music 当作真实监督配对。

因此当前可以说“生成结果对重音和节奏变化有响应趋势”，不说“在敦煌数据上学会了音乐-动作严格对齐”或“完美卡准节拍”。

## 当前运动表示

当前代码实际使用 151 维表示：

- 4 维脚部接触
- 3 维 root position
- 24 个关节的 6D rotation，共 144 维

总计 4 + 3 + 144 = 151 维。

For data and music-supervision limitations, see `docs/data_limitations.md`.

我们发现目前 trajectory control 主要依赖 TTO 和后处理 anchor，模型原生循迹能力不足。为此新增了 native trajectory control patch：

1. 在推理阶段加入 trajectory-specific classifier-free guidance。与普通 CFG 不同，它保留 audio 条件，只 drop trajectory 条件作为 baseline，因此能增强空间循迹而尽量不破坏音乐节奏。

2. 在训练阶段将 trajectory loss 改为 time-dependent 权重，高噪声阶段更强调 root X/Z 宏观路径，低噪声阶段保留姿态细节生成。

3. 将轨迹监督从单一绝对坐标 MSE 扩展为位置、相对速度、加速度和 endpoint 联合损失，降低长程漂移和终点误差。

这个版本不依赖 TTO，也不强制后处理替换 root，因此能作为下一阶段提升原生可控性的基础。

7. 后续如果这个补丁仍不够
如果 native CFG + 动态 loss 仍然不足，再进入更大改动：
阶段 2：把 trajectory 表征从绝对位置改成 ΔX/ΔZ 速度条件。
阶段 3：新增 trajectory encoder / memory tokens。
阶段 4：ControlNet-like 逐层 trajectory adapter。
阶段 5：root trajectory generator + local pose diffusion 的分层生成。