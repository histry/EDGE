# EDGE V32 Continuous Contact-INR 替换与运行指南

## 0. 安装前

确认正在运行的 V31/C2 tmux 已结束。运行中的 Python 进程重新加载被覆盖文件
时可能发生 checkpoint 架构冲突。

```bash
tmux ls
ps -ef | grep -E 'schedule_v31|run_v31|train_v27' | grep -v grep
```

## 1. 安装

解压后：

```bash
python EDGE_V32_CONTACT_INR_PATCH/install_v32_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

原文件会备份至：

```text
EDGE/backup_v32_时间戳/
```

## 2. 当前可运行模式：weak

你现在没有完整 source manifest，因此先使用：

```bash
export V32_SUPERVISION_MODE=weak
```

该模式仍会使用约 4225 个事件构建大量真实 `intra_event_real` 样本，并将
合成相邻桥接限制为低权重条件正则。Contact fine-tuning 只使用 real target。

### 完整指令

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

export V27_HYPERBOLIC_CKPT=output/v28_ragdiff_hyper_clap_public_20260610_015040/v27_hyperbolic_hierarchy/best.pt
export V26_HIERARCHY_INDEX_NPZ=output/v28_diffusion_retrain_musicclap_20260610_024828/v28_hyperbolic_hierarchical_event_index.npz

export V26_MUSIC='test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav'
export V32_KEYS='dunhuangwu2;dunhuangwu3;dunhuangwu4'

export V32_SUPERVISION_MODE=weak
bash scripts/run_v32_contact_inr_full.sh
```

## 3. Weak 模式关键开关

```bash
export V32_SAMPLES_PER_EVENT=6
export V32_SOURCE_PAIRS_PER_EVENT=0.75
export V32_ALLOW_SYNTHETIC_ADJACENT=1
export V32_PSEUDO_PAIRS_PER_EVENT=0.05

export V32_AE_EPOCHS=220
export V32_CONTACT_EPOCHS=90
export V32_DIFFUSION_EPOCHS=320

export V32_AE_BATCH_SIZE=40
export V32_CONTACT_BATCH_SIZE=32
export V32_LATENT_BATCH_SIZE=192

export V32_FOURIER_BANDS=5
export V32_ROTATION_RESIDUAL_SCALE=0.16
export V32_ROOT_Y_RESIDUAL_SCALE=0.045

export V32_W_CONTACT_SKATE=1.20
export V32_W_FOOT_PENETRATION=0.80
export V32_W_CONTACT_HEIGHT=0.45

export V32_CANDIDATES=8
export V32_GUIDANCE=1.0
export V32_INR_TRUST=0.35
export V32_ENABLE_EDGE_DAMPING=0
```

4090 显存不足时优先降低：

```bash
export V32_AE_BATCH_SIZE=24
export V32_CONTACT_BATCH_SIZE=20
export V32_LATENT_BATCH_SIZE=96
export V32_DECODED_BATCH_LIMIT=4
```

不要提高 Fourier bands 或 rotation residual scale 来追求训练拟合。

## 4. Strict 模式

找回完整长序列后构建 manifest：

```bash
python tools/build_v30_source_manifest.py \
  --index_json "$V26_INDEX_JSON" \
  --duration_index_npz "$V26_DURATION_INDEX_NPZ" \
  --full_motion_root /path/to/full_motion \
  --audio_root /path/to/audio \
  --out_json data/v32_source_manifest.json
```

确认 resolved sources 不再为 0，然后：

```bash
export V32_SUPERVISION_MODE=strict
export V32_SOURCE_MANIFEST=data/v32_source_manifest.json
export V32_FULL_MOTION_ROOT=/path/to/full_motion

export V32_REQUIRE_REAL_BOUNDARY_SAMPLES=1000
export V32_REQUIRE_UNIQUE_BOUNDARIES=250
export V32_REQUIRE_REAL_BOUNDARY_RATIO=0.10

bash scripts/run_v32_contact_inr_full.sh
```

Strict 模式固定：

```text
synthetic adjacent = off
pseudo pairs = off
include synthetic in training = false
```

## 5. 分阶段运行

只重建数据库：

```bash
export V32_BUILD_DATASET=1
export V32_TRAIN=0
export V32_DATASET=/path/to/dataset.npz
```

复用数据库重训：

```bash
export V32_BUILD_DATASET=0
export V32_DATASET=/path/to/v32_transition_dataset.npz
export V32_TRAIN=1
```

复用 checkpoint 仅生成：

```bash
export V32_BUILD_DATASET=0
export V32_DATASET=/path/to/v32_transition_dataset.npz
export V32_TRAIN=0
export V27_TRANSITION_DIFFUSION_CKPT=/path/to/best.pt
```

## 6. 训练输出

```text
v32_contact_dataset_audit.json
v32_contact_inr_training/checkpoints/autoencoder_best.pt
v32_contact_inr_training/checkpoints/contact_finetuned_best.pt
v32_contact_inr_training/checkpoints/best.pt
v32_contact_inr_training/history.json
```

主 checkpoint 架构必须是：

```text
v32_continuous_c2_contact_inr_latent_diffusion
```

V29/V30/V31 checkpoint 会被 loader 明确拒绝，防止静默混用。

## 7. 生成输出

每首音乐生成：

```text
*_v26.npy
*_v26.schedule_report.json
*_v32.long_dance.json
*_v32.public_metrics.json
*_v32.frequency_foot.json
*_v32.contact_metrics.json
*_v32.jitter.json
*_v32_scientific_fixed.mp4
```

此外：

```text
v32_transition_gate_summary.json
v32_experiment_summary.json
```

## 8. 主要判定指标

V32 必须同时优于 C2 baseline：

```text
transition_exit pose jump
transition_exit velocity/acceleration
joint jerk P95
angular jerk P95
transition high-frequency ratio
contact slip
foot penetration
BAS
FGDkin
```

若 fallback rate 接近 100%，说明模型没有稳定优于 C2，不能把 latent diffusion
作为有效贡献；若 fallback rate 接近 0% 但 jitter 更差，说明安全阈值过松。

## 9. 推荐消融

```text
C2 deterministic baseline
V32 INR autoencoder only
V32 without contact fine-tuning
V32 without contact skate loss
V32 without penetration loss
V32 without spectral loss
V32 single candidate without gate
V32 trust 0.20 / 0.35 / 0.50
V32 weak
V32 strict
```

所有实验固定音乐、事件选择、时长分配、随机种子和 scientific render。

## 10. 不能夸大的结论

- Weak 模式不等于真实 event-to-event transition learning；
- 可微接触损失降低 foot sliding 需要实验验证；
- 连续 INR 提供任意长度表达，不自动保证比 C2 更平滑；
- 只有 V32-Strict 才能支撑“破除合成转场闭环”的主张。
