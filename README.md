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
