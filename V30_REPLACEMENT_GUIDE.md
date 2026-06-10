# EDGE V30 顶会/顶刊科研版替换与训练说明

## 0. 先读：不要覆盖正在运行的 V29

如果当前 V29 整夜实验仍在训练，请等它完成生成、评价和渲染后再安装 V30。
原因是 V29 的后续阶段会重新启动 Python 进程；训练过程中覆盖
`tools/v27_transition_diffusion.py` 等文件，可能导致后续进程加载 V30
代码却读取 V29 checkpoint，从而中断当前实验。

## 1. V30 的主线

V30 保留已经稳定的整曲规划部分：

- 音乐短语边界锁定；
- 短语内多事件槽；
- V23 自然时长与单调 time-warp；
- 层级 Event-RAG；
- 图路径调度；
- 精确整曲长度对齐。

V30 重构两个薄弱环节：

1. **转场生成**  
   固定长度离散序列扩散升级为连续 SO(3) 残差 INR + 潜空间扩散。
   INR 在任意归一化时间坐标上解码，不要求训练和推理具有相同帧数；
   网络只预测端点安全 SO(3) Hermite 基路径上的有界残差。

2. **跨模态检索与生成条件**  
   删除“CLAP → 固定随机矩阵 → 12D”的核心依赖。
   训练音乐塔与多部件动作塔进入同一个 Poincaré 球，用测地对比、
   层级约束和正负偏好损失联合训练。相同几何音乐嵌入同时用于
   Event-RAG 检索和转场潜扩散条件。

## 2. 真实监督门控

V30 不再允许最终论文模型悄悄依赖合成桥接。

### 2.1 完整长序列 source manifest

先运行审计工具：

```bash
python tools/build_v30_source_manifest.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --full_motion_root /path/to/full_3d_sequences \
  --audio_root /path/to/source_audio \
  --out_json data/v30_source_manifest.json
```

打开 JSON，修复 `unresolved` 列表。最终格式：

```json
{
  "sources": {
    "source_001": {
      "motion": "/abs/path/source_001.pkl",
      "audio": "/abs/path/source_001.wav"
    }
  },
  "events": {
    "event_00001": {
      "source_id": "source_001",
      "source_motion": "/abs/path/source_001.pkl",
      "source_start": 120,
      "source_end": 173
    }
  }
}
```

即使两个事件裁剪在边界处无空白 gap，V30 也会在完整长序列上随机遮挡
一个跨越真实事件边界的区间，构造 `source_boundary_mask_real`。

### 2.2 音乐—动作显式配对

顶刊主模型应提供人工复核的 JSONL：

```json
{"audio":"/abs/a.wav","start_frame":0,"end_frame":90,"event_index":128,"group":"song_001","weight":1.0}
{"audio":"/abs/a.wav","start_frame":90,"end_frame":170,"event_index":913,"group":"song_001","weight":1.0}
```

也可先从已有 schedule 导出弱监督初稿：

```bash
python tools/build_v30_pair_manifest_from_schedules.py \
  --report_glob 'output/**/**.schedule_report.json' \
  --audio_dir test_music_bank \
  --out_jsonl data/v30_pairs_weak.jsonl
```

该文件必须人工复核。未复核的 schedule 派生配对只能用于预训练或消融，
不能作为“无配对学习”的真实证据。

## 3. 安装

```bash
python install_v30_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

旧文件会备份到：

```text
EDGE/backup_v30_时间戳/
```

## 4. 最严格的一键训练

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

export V26_INDEX_JSON=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json
export V26_DURATION_INDEX_NPZ=data/v26_music_dominant_duration_index.npz
export V26_ROUTER_CKPT=output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt
export V26_V23_CKPT=output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt
export V26_PLANNER_CKPT=$(cat output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt)
export V26_START_POSE=data/canonical_dunhuang_start_pose.npy
export V27_HYPERBOLIC_CKPT=/path/to/v27_hyperbolic/best.pt

export V27_CLAP_CKPT=$PWD/pretrained/laion_clap/music_audioset_epoch_15_esc_90.14.pt
export V27_CLAP_AMODEL=HTSAT-base
export V27_CLAP_DEVICE=cuda:0
export V27_CLAP_ENABLE_FUSION=0
export V27_CLAP_USE_FILELIST=0

export V30_SOURCE_MANIFEST=data/v30_source_manifest.json
export V30_PAIR_MANIFEST=data/v30_pairs_expert_checked.jsonl
export V30_AUDIO_ROOT=/path/to/source_audio

export V26_MUSIC='test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav'
export V30_KEYS='dunhuangwu2;dunhuangwu3;dunhuangwu4'

bash scripts/run_v30_full_research.sh
```

## 5. 严格门控默认值

```bash
export V30_ALLOW_WEAK_SUPERVISION=0
export V30_REQUIRE_EXPLICIT_PAIRS=1000
export V30_REQUIRE_CLAP_PAIRS=900
export V30_REQUIRE_REAL_BOUNDARIES=1000
export V30_REQUIRE_REAL_BOUNDARY_RATIO=0.10
export V30_REQUIRE_REAL_MUSIC_SAMPLES=1000
export V30_REQUIRE_REAL_MUSIC_RATIO=0.10
```

不足时脚本直接失败。应修复数据，不应为了跑通而修改论文主实验门槛。

仅用于代码冒烟测试：

```bash
export V30_ALLOW_WEAK_SUPERVISION=1
export V30_ROUTER_DATA=/path/to/v21_router_training_data.npz
```

## 6. 连续 INR 与潜扩散开关

```bash
export V30_AE_EPOCHS=220
export V30_DIFFUSION_EPOCHS=320
export V30_AE_BATCH_SIZE=48
export V30_LATENT_BATCH_SIZE=256
export V30_LATENT_DIM=128
export V30_INR_HIDDEN=320
export V30_INR_LAYERS=5
export V30_FOURIER_BANDS=10
export V30_DIFFUSION_HIDDEN=512
export V30_DIFFUSION_BLOCKS=6
export V30_DIFFUSION_STEPS=100

export V30_INFERENCE_STEPS=40
export V30_LATENT_GUIDANCE=1.20
export V30_INR_BLEND=0.85
export V30_TRANSITION_FILTER_WINDOW=3
export V30_TRANSITION_FILTER_STRENGTH=0.10
```

4090 显存不足时优先降低：

```bash
export V30_AE_BATCH_SIZE=24
export V30_LATENT_BATCH_SIZE=128
```

## 7. 跨模态几何检索开关

```bash
export V30_ALIGNMENT_EPOCHS=320
export V30_ALIGNMENT_BATCH_SIZE=192
export V30_ALIGNMENT_EMBED_DIM=32
export V30_CROSSMODAL_RETRIEVAL_WEIGHT=0.35
```

建议消融：

```text
crossmodal weight = 0.00 / 0.20 / 0.35 / 0.50
Euclidean router baseline
Poincaré without preference loss
Poincaré without hierarchy loss
V30 full
```

## 8. 外部 Wild Video 先验

主代码只提供低权重外部先验接口：

```bash
export V30_EXTERNAL_PRIOR_NPZ=/path/to/external_prior.npz
```

NPZ 至少包含：

```text
target [N,K,151]
start  [N,151]
end    [N,151]
length [N]
music  [N,D]  可选
```

外部数据默认权重低于真实敦煌长序列。它不能替代敦煌舞真实边界监督。
在没有可靠 2D→3D 重建、动作质量过滤、文化风格过滤和数据许可审计之前，
不要把互联网视频作为主训练数据。

## 9. 评价输出

每首音乐生成：

```text
*_v30.long_dance.json
*_v30.public_metrics.json
*_v30.frequency_foot.json
*_v30.dct_spectrum.png
*_v30_scientific_fixed.mp4
```

`frequency_foot.json` 包含：

- 全曲和转场单独的 DCT 低/中/高频能量；
- 高频/低频比；
- 频谱熵；
- contact-conditioned foot sliding；
- explicit kinematic multi-scale MMD；
- explicit foot-motion Fréchet distance。

这些透明描述符用于内部消融。标准论文表中仍应同时报告社区通用的
learned motion encoder 指标，并明确 encoder、训练数据和实现来源。

## 10. 必做消融

1. V26 rule baseline；
2. V29 discrete temporal diffusion；
3. V30 INR autoencoder without latent diffusion；
4. V30 latent diffusion without SO(3)/FK/spectral losses；
5. V30 without real boundary masks；
6. V30 Euclidean alignment；
7. V30 hyperbolic alignment without preference ranking；
8. V30 without cross-modal retrieval；
9. V30 without graph edge cost；
10. V30 full。

所有版本使用相同音乐、事件库、随机种子和渲染设置。

## 11. 不能夸大的表述

- 连续 INR 技术路线可作为重要升级，但最终效果必须由实验决定；
- 本实现中的 multi-scale MMD 和 foot-motion Fréchet 是透明内部指标，
  不能冒充社区统一标准缩写；
- Wild Video 模块是可选接口，不是当前完整实现的论文主贡献；
- 若 source manifest 或 expert pair 不足，不能声称解决了伪标签闭环；
- 顶会/顶刊结果还需要公平复现 LODGE++、MotionRAG-Diff 等基线、
  多随机种子统计、主观评价和显著性检验。
