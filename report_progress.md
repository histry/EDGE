# 敦煌可控数字编舞系统进度汇报

## 1. 课题目标

本课题目标是把现有 EDGE 音乐条件舞蹈生成框架改造成一个可控的敦煌舞数字编舞原型。

输入包括：

- 起始敦煌舞姿关键帧
- 终止敦煌舞姿关键帧
- 可选中间关键帧
- 背景音乐
- 2D 空间移动轨迹

输出包括：

- 151 维 SMPL motion tensor
- 3D 骨架渲染视频
- 定量评估指标

当前重点不是声称已经解决完整的“真实敦煌音乐-动作配对生成”，而是验证可控编舞链路：关键帧控制、轨迹控制、音乐节拍锚点、长序列生成和定量评估。

## 2. 当前系统能力

### 2.1 关键帧控制

当前系统支持：

- 起始关键帧
- 终止关键帧
- 多个中间关键帧
- 中间关键帧软约束强度
- 关键帧约束宽度
- 首/中/尾姿态路径弱引导

关键帧主要约束局部身体姿态和 root 高度。默认不把 root X/Z 纳入关键帧姿态误差，因为 root X/Z 由轨迹模块负责。

### 2.2 轨迹控制

当前系统支持输入 2D 控制点，例如：

```text
0,0; 1,2; -1,4; 0,5
```

代码会生成连续 X/Z 轨迹，并在推理后处理阶段进行轨迹锚定或优化。

当前保留两类输出：

- `v5_strict_path`：严格轨迹版，X/Z 轨迹完全不改。
- `v6_display_smooth`：展示平滑版，对 root X/Z 做厘米级微平滑，用于降低 follow camera 下轨迹折点造成的视觉卡顿。

### 2.3 音乐使用方式

由于目前没有真实敦煌音乐-动作配对数据集，音乐相关 claim 统一采用保守口径：

> 当前音乐部分采用“继承预训练 + 推理节拍锚点引导”，而不是在敦煌数据上重新学习真实音乐-动作配对。

具体含义：

1. 当前模型保留原 EDGE/AIST 的扩散生成框架、动作生成先验和 audio-conditioned 接口结构。
2. 由于当前使用 803 维 hybrid audio feature，当音频投影层维度与原 checkpoint 不一致时，相关音频投影层会重新初始化；因此不 claim 完整继承原模型的音乐编码能力。
3. 敦煌阶段主要学习敦煌姿态风格、关键帧过渡、轨迹控制和物理连续性。
4. proxy music 不作为真实标签，只作为弱节奏候选和音频接口占位。
5. 推理阶段从测试音乐中提取 onset/beat，作为节拍锚点参与轨迹节奏和可选姿态锚点引导。

不 claim：

- 不说敦煌数据上完成了真实音乐-动作配对训练。
- 不说 proxy music 是真实标签。
- 不说模型已经完美卡准节拍。

### 2.4 稳定性处理

针对长序列生成中出现的动作突变，当前增加了：

- 局部姿态尖峰检测
- SO(3) 角速度/角加速度检测
- quaternion SLERP 局部平滑
- 全局弱旋转平滑
- 可选 root X/Z 轨迹微平滑
- follow camera 平滑

当前推荐：

- 严格轨迹汇报：使用 v5。
- 展示视频观感：使用 v6。

## 3. Proxy Music 弱配对实验

### 3.1 方法

在没有真实音乐-动作配对数据时，使用 BPM/节拍相近的 proxy music 构造弱配对候选。

流程：

1. 收集 proxy music。
2. 对 proxy music 提取 BPM、beat density、onset density。
3. 将敦煌动作切成 5 秒窗口。
4. 对每个动作窗口估计 motion BPM、动作重音密度和 root 运动强度。
5. 计算动作窗口与 proxy music 的节奏相似度。
6. 每个动作窗口保留 top-k 个 proxy music 候选。
7. 明确这些候选不是自然配对标签，只是弱节奏候选。

生成脚本：

```bash
/home/disk/lsm/conda_envs/edge/bin/python build_proxy_weak_pairs.py \
  --motion_dir data/dunhuang_bvh/processed \
  --proxy_dir proxy_music \
  --seq_len 150 \
  --stride 75 \
  --fps 30 \
  --top_k 3 \
  --out_dir data/proxy_weak_pairs
```

输出文件：

```text
data/proxy_weak_pairs/weak_pairs.csv
data/proxy_weak_pairs/weak_pairs.json
data/proxy_weak_pairs/proxy_music_features.csv
data/proxy_weak_pairs/motion_rhythm_features.csv
```

### 3.2 统计结果

当前统计：

- motion windows：2880
- top-k candidate rows：8640
- high-confidence threshold：score >= 0.8
- high-confidence candidate rows：1865

| Proxy music | BPM | Beat density | Onset density | All hits | Hits >= 0.8 | High-conf windows | High-conf share | Mean score | Best score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guzhen.wav | 112.50 | 1.883 | 2.574 | 2838 | 573 | 573 | 0.307 | 0.597 | 0.979 |
| gu.wav | 105.88 | 1.729 | 2.530 | 1614 | 525 | 525 | 0.282 | 0.628 | 0.966 |
| xiao.wav | 120.00 | 1.999 | 2.866 | 2815 | 517 | 517 | 0.277 | 0.578 | 0.972 |
| pipa.wav | 120.00 | 2.033 | 3.579 | 1373 | 250 | 250 | 0.134 | 0.609 | 0.907 |

解释：

- 当前 proxy music 覆盖了约 105-120 BPM 的节奏区间。
- 高置信候选主要集中在 105-120 BPM，说明当前动作窗口也大量集中在该节奏范围。
- 后续如果需要更完整覆盖，应补充 90、100、130、140 BPM 左右的 proxy music。

## 4. 定量评估

新增评估脚本：

```text
eval_quantitative.py
```

支持指标：

- 关键帧误差：MPJPE、旋转角误差、feature RMSE
- 轨迹误差：ADE、RMSE、max error、final error、DTW
- 脚滑率：contact frame 下脚底水平速度超阈值比例
- BeatAlign：motion beat 与 audio beat 的高斯匹配分数

### 4.1 v5/v6 对比

| Metric | v5_strict_path | v6_display_smooth |
| --- | --- | --- |
| Keyframe MPJPE mean (cm) | 8.49 | 8.49 |
| Keyframe MPJPE max (cm) | 20.23 | 20.23 |
| Keyframe rot err mean (deg) | 7.61 | 7.61 |
| Trajectory ADE (cm) | 0.00 | 0.35 |
| Trajectory RMSE (cm) | 0.00 | 0.43 |
| Trajectory max err (cm) | 0.00 | 1.18 |
| Trajectory final err (cm) | 0.00 | 0.00 |
| Foot slide rate (%) | 94.06 | 100.00 |
| Foot contact speed P95 (m/s) | 0.416 | 0.439 |
| BeatAlign symmetric | 0.560 | 0.516 |
| BeatAlign motion->audio | 0.458 | 0.441 |
| BeatAlign audio->motion | 0.662 | 0.592 |

### 4.2 结果解释

关键帧：

- v5 和 v6 关键帧误差完全一致。
- 说明 v6 的展示平滑没有破坏关键帧姿态控制。

轨迹：

- v5 严格保留原始轨迹，轨迹误差为 0。
- v6 对 root X/Z 做厘米级微平滑，Trajectory ADE 为 0.35 cm，最大偏差 1.18 cm。
- 该偏差可以解释为展示观感优化，而不是轨迹控制失败。

脚滑：

- foot slide rate 仍然较高。
- 这说明当前 foot contact/foot lock/physical loss 仍是主要短板。
- 后续应优先优化脚底锁定和接触一致性。

BeatAlign：

- 当前 BeatAlign 约 0.52-0.56。
- 说明音乐节拍响应存在，但还不稳定。
- 后续需要把 beat anchor 从推理后处理进一步接入 sampling guidance 或训练损失。

## 5. 当前可汇报结论

当前已经完成的工作：

1. 搭建了敦煌舞可控生成推理链路。
2. 支持起止关键帧和中间关键帧控制。
3. 支持 2D 空间轨迹输入与后处理锚定。
4. 支持背景音乐输入和 onset/beat 节拍锚点提取。
5. 明确了没有真实配对数据时的音乐使用口径。
6. 构建了 BPM/beat 相近的 proxy music 弱配对候选库。
7. 加入了关键帧误差、轨迹误差、脚滑率和 BeatAlign 的定量评估。
8. 修复并缓解了长序列视频中的部分动作突变和 follow camera 卡顿问题。

当前仍存在的问题：

1. 脚滑率较高，foot contact/foot lock 需要加强。
2. BeatAlign 不够稳定，音乐节拍目前主要是弱引导。
3. proxy music 数量较少，BPM 覆盖范围还不够宽。
4. 中间关键帧误差仍明显高于首尾关键帧。
5. RAG 尚未作为默认能力开启，目前只能作为可选姿态先验。

## 6. 老师可能追问的问题

### Q1：没有真实音乐-动作配对，为什么还能输入音乐？

答：音乐输入分两层作用。第一层是继承原 EDGE/AIST 预训练模型中的 audio-conditioned denoising prior；第二层是在推理阶段提取真实音乐的 onset/beat，作为节拍锚点进行弱引导。敦煌阶段不 claim 学到了真实音乐-动作配对关系。

### Q2：proxy music 是训练标签吗？

答：不是。proxy music 只根据 BPM、beat density 和动作重音密度形成弱节奏候选，用于保持音频条件接口和节奏分布，不作为真实监督标签。

### Q3：音乐相关 loss 怎么设置？

答：敦煌阶段主 loss 是 diffusion denoising、关键帧约束、轨迹约束和物理连续性。由于没有真实配对音乐，不把 MMR/cross-modal alignment 作为主 loss；如果使用，也只能极小权重并结合弱配对 confidence。

### Q4：v5 和 v6 应该展示哪个？

答：如果强调严格轨迹控制，展示 v5；如果强调视频观感和 follow camera 平滑，展示 v6。v6 的最大轨迹偏差约 1.18 cm，属于展示平滑带来的厘米级偏移。

### Q5：当前最大短板是什么？

答：最大短板是脚滑和节拍稳定性。定量评估中 foot slide rate 仍然偏高，BeatAlign 也只是中等水平，后续需要加强 foot lock 和 beat-guided sampling。

## 7. 下一步计划

优先级建议：

1. 优化 foot lock 和接触一致性，降低脚滑率。
2. 将 beat anchor 从后处理/弱引导进一步接入 sampling guidance。
3. 扩充 proxy music 库，覆盖 90、100、130、140 BPM。
4. 降低中间关键帧误差，调节 mid pose strength、width 和 pose path guidance。
5. 谨慎开启 RAG，只在强拍附近注入局部姿态，不覆盖 root X/Z。

## 8. 相关文件

核心脚本：

```text
inference_music.py
eval_quantitative.py
build_proxy_weak_pairs.py
make_proxy_weak_pair_table.py
make_eval_comparison_table.py
```

结果文件：

```text
data/proxy_weak_pairs/proxy_weak_pair_stats.md
data/proxy_weak_pairs/proxy_weak_pair_stats.csv
output/v5_v6_eval_comparison.md
output/v5_v6_eval_comparison.csv
output/stage2B_train5_4key_midmove_filter_20s_resmoothed_v5_followcam.mp4
output/stage2B_train5_4key_midmove_filter_20s_resmoothed_v6_followcam.mp4
```
