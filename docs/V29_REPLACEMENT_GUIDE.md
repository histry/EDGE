# EDGE V29 科研版替换说明

## 1. 本补丁替换什么

直接替换的现有文件：

- `tools/v27_transition_diffusion.py`
- `tools/build_v27_transition_diffusion_dataset.py`
- `train_v27_transition_diffusion.py`
- `tools/evaluate_v26_long_dance.py`
- `tools/evaluate_v27_public_metrics.py`
- `render_from_npy.py`

新增文件：

- `tools/v29_motion_geometry.py`
- `tools/schedule_v29_whole_song.py`
- `tools/diagnose_v29_jitter.py`
- `scripts/run_v29_whole_song.sh`
- `scripts/run_v29_rebuild_retrain_generate.sh`

原有 `tools/schedule_v26_whole_song.py` 不删除。V29 入口在运行时替换其中的几何敏感函数，从而继续使用已经验证的音乐边界锁定、V23 时长、层级 RAG 和图调度逻辑。

## 2. 安装

在解压目录执行：

```bash
python install_v29_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

安装器会把被覆盖文件保存到：

```text
/home/disk/lsm/storage/EDGE/backup_v29_时间戳/
```

也可以将压缩包中的目录结构直接覆盖到 EDGE 根目录。

## 3. 必须重新训练

V29 模型结构与旧 V27/V28 逐帧 MLP checkpoint 不兼容，必须重建 transition dataset 并重新训练。旧 checkpoint 会被代码明确拒绝，避免误用。

## 4. 推荐的一键运行

先激活环境：

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0
```

配置已有模块：

```bash
export V26_INDEX_JSON=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json
export V26_DURATION_INDEX_NPZ=data/v26_music_dominant_duration_index.npz
export V26_ROUTER_CKPT=output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt
export V26_V23_CKPT=output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt
export V26_PLANNER_CKPT=$(cat output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt)
export V26_START_POSE=data/canonical_dunhuang_start_pose.npy
export V26_HIERARCHY_INDEX_NPZ=你的层级索引.npz
export V27_HYPERBOLIC_CKPT=你的双曲层级checkpoint.pt

export V27_CLAP_CKPT=$PWD/pretrained/laion_clap/music_audioset_epoch_15_esc_90.14.pt
export V27_CLAP_AMODEL=HTSAT-base
export V27_CLAP_DEVICE=cuda:0
export V27_CLAP_ENABLE_FUSION=0
export V27_CLAP_USE_FILELIST=0
export V27_DEEP_MUSIC_FEATURES=1
export V27_DEEP_MUSIC_MODEL=clap
export V27_REQUIRE_DEEP_MUSIC=1
export V27_DEEP_MUSIC_MIN_SUCCESS=0.80
```

运行：

```bash
bash scripts/run_v29_rebuild_retrain_generate.sh
```

## 5. 环境开关

### 数据集

```bash
export V29_REBUILD_TRANSITION_DATASET=1
export V29_SAMPLES_PER_EVENT=6
export V29_SOURCE_PAIRS_PER_EVENT=1.0
export V29_PSEUDO_PAIRS_PER_EVENT=0.45
```

论文主实验建议保留真实窗口和相邻源数据为主，不要提高随机 pseudo pair 比例。

### 训练

```bash
export V29_RETRAIN_TRANSITION=1
export V29_EPOCHS=420
export V29_BATCH_SIZE=64
export V29_HIDDEN_DIM=384
export V29_DIFFUSION_TRAIN_STEPS=100
export V29_AMP=1
```

4090 显存不足时：

```bash
export V29_BATCH_SIZE=32
export V29_HIDDEN_DIM=320
```

### 推理

```bash
export V29_ENABLE_TRANSITION_DIFFUSION=1
export V29_TRANSITION_BLEND=0.18
export V29_TRANSITION_INFER_STEPS=32
export V29_TRANSITION_NOISE_STRENGTH=0.55
export V29_TRANSITION_FILTER_WINDOW=5
export V29_TRANSITION_FILTER_STRENGTH=0.20
```

不建议把 blend 直接提高到 0.35。先用 0.12、0.18、0.24 做消融。

### 事件边缘

```bash
export V29_EDGE_DAMPING_FRAMES=4
export V29_EDGE_DAMPING_STRENGTH=0.25
```

V29 的 edge damping 是 SO(3) 时间重参数化，不是原来的 raw-6D 线性混合。

### 消融

无扩散：

```bash
export V29_ENABLE_TRANSITION_DIFFUSION=0
```

无边缘减速：

```bash
export V29_EDGE_DAMPING_FRAMES=0
export V29_EDGE_DAMPING_STRENGTH=0
```

无局部滤波：

```bash
export V29_TRANSITION_FILTER_WINDOW=1
export V29_TRANSITION_FILTER_STRENGTH=0
```

## 6. 科研评价要求

每首音乐会生成：

- `*_v29.evaluation.json`：转场入口和出口、yaw、FK acceleration/jerk
- `*_v29.public_metrics.json`：FK/SO(3) BAS 与 FGD-style
- `*_v29.jitter.json`：最抖关节和最差帧
- `*_v29_scientific_fixed.mp4`：无渲染平滑
- `*_v29_display_fixed.mp4`：仅用于展示的轻量渲染平滑

论文指标与人工检查必须使用 scientific 视频。display 视频不能替代动作张量评价。

## 7. 推荐消融矩阵

1. `safe_so3`：diffusion=0，SO(3) transition + SO(3) edge damping
2. `v29_no_filter`：diffusion=1，filter=0
3. `v29_blend_012`
4. `v29_blend_018`
5. `v29_blend_024`
6. `v29_full`

选择 full 模型时同时看：

- BAS
- FK FGD-style
- transition entry/exit velocity jump
- joint jerk p95
- yaw max
- 主观固定相机视频

不能只按 diffusion val loss 选模型。
