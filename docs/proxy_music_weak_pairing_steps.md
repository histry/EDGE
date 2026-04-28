# BPM/Beat Proxy Music Weak Pairing Steps

目标：在没有真实敦煌音乐-动作配对数据时，用 BPM 和节拍相近的 proxy music 构造弱配对候选。该结果只能作为节奏弱条件或实验记录，不作为真实音乐-动作标签。

## 1. 准备 proxy music

把候选音乐放入：

```bash
proxy_music/
```

当前已有：

```text
proxy_music/gu.wav
proxy_music/guzhen.wav
proxy_music/pipa.wav
proxy_music/xiao.wav
```

建议 proxy music 覆盖慢板、中板、鼓点明显、旋律性强几类。文件统一用 `.wav`。

## 2. 准备动作数据

脚本默认读取：

```bash
data/dunhuang_bvh/processed/*.pkl
```

每个 `.pkl` 需要包含：

```text
pos: [T, 3]
q:   [T, 72]
```

## 3. 提取音乐节拍特征

脚本会对每首 proxy music 自动提取：

```text
BPM
beat_count
beat_density
onset_count
onset_density
onset_energy
```

输出文件：

```bash
data/proxy_weak_pairs/proxy_music_features.csv
```

## 4. 提取动作节奏特征

脚本会把长动作切成 5 秒窗口：

```text
seq_len = 150 frames
fps = 30
duration = 5 sec
stride = 75 frames
```

每个动作窗口提取：

```text
motion_bpm
accent_count
accent_density
accent_energy
root_path_len
root_speed_mean
```

动作重音来自 root 速度、root 加速度和局部旋转变化的混合曲线。

输出文件：

```bash
data/proxy_weak_pairs/motion_rhythm_features.csv
```

## 5. 生成 top-k 弱配对候选

运行：

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

输出：

```bash
data/proxy_weak_pairs/weak_pairs.csv
data/proxy_weak_pairs/weak_pairs.json
```

`weak_pairs.csv` 每行是一个动作窗口和一首 proxy music 的候选关系。

## 6. 查看弱配对结果

查看 proxy music 的 BPM：

```bash
cat data/proxy_weak_pairs/proxy_music_features.csv
```

查看分数最高的候选：

```bash
head -20 data/proxy_weak_pairs/weak_pairs.csv
```

关键列含义：

```text
score: 总弱配对分数
bpm_score: BPM 相似度
density_score: 节拍密度相似度
onset_score: onset 密度相似度
window_id: 动作窗口 ID
motion_bpm: 动作估计 BPM
audio_bpm: proxy music BPM
weak_pair_claim: 明确说明不是真实配对标签
```

## 7. 如何筛选可用候选

建议先保留：

```text
score >= 0.70
```

更严格可以用：

```text
score >= 0.80
```

不要只看总分，也要检查：

```text
motion_bpm 和 audio_bpm 是否接近
motion_accent_density 和 audio_beat_density 是否接近
```

## 8. 如何用于训练或实验

推荐用法：

1. 每个动作窗口保留 top-3 proxy music。
2. 训练时随机采样其中一首，采样概率按 `score` 加权。
3. 不使用强 cross-modal label loss。
4. MMR/cross-modal loss 关闭或极小，并乘以弱配对 confidence。
5. 主训练目标仍然是 diffusion reconstruction、keyframe、trajectory 和 physical losses。

不要把 CSV 里的候选写成真实配对。

## 9. 如何用于推理

推理时不必使用 proxy music。推理音乐应该是用户输入的真实背景音乐。

推理阶段使用真实音乐提取：

```text
onset peaks
beat positions
BPM
beat strength
```

这些特征作为节拍锚点，用于轨迹速度分配、重音处姿态锚点或可选 RAG 片段。

## 10. 汇报口径

可以这样说：

> 由于没有真实敦煌音乐-动作配对数据，我们不把 proxy music 当作真实监督标签，而是根据 BPM、beat density 和动作重音密度构造 top-k 弱配对候选。该弱配对只提供节奏分布和接口占位，音乐生成能力主要继承预训练 audio prior，推理阶段再用真实音乐的 beat/onset 作为节拍锚点进行弱引导。
