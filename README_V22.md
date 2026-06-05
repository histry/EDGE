# V22 Turn-Aware Temporal Pace Refinement

## 1. 为什么不是继续调 V21 time-warp 权重

V21 的 `time_warp_weight/min/max` 只影响候选筛选。实验证明 mild 与 medium 输出的三个 NPY 哈希完全一致，因此权重没有改变最终动作。V22 对问题做两层处理：

1. **Turn-Speed-Aware Retrieval Gate**：在 Event-RAG 中记录每个动态事件的转身角度、持续时间和峰值 Root-Yaw 速度；Scheduler 按实际重采样比例估计输出转速，过快候选被惩罚或拒绝。
2. **Learned All-Position Turn Pace Refiner**：从连续敦煌动作中自动构造“自然转身 → 人工压缩过快转身”的训练对，训练局部 TCN 将任意位置的过快转身恢复到自然速度。它不是“首个转身补丁”。

Router 不需要重训。V21 Router 继续负责音乐语义路由，V22 只训练时间动力学模块。

## 2. 文件

- `tools/annotate_v22_turn_index.py`：为共享索引增加通用转身元信息。
- `tools/build_v22_turn_pace_dataset.py`：从 72 条连续 151D 敦煌动作构造训练集。
- `model/v22_turn_pace.py`：FiLM-TCN 局部残差 Refiner。
- `train_v22_turn_pace.py`：按源视频分组切分训练/验证，避免相邻窗口泄漏。
- `tools/schedule_v22_multi_music.py`：V21 Scheduler 的 V22 完整替代入口，加入转速门控并可加载 Refiner。
- `tools/v22_turn_runtime.py`：检测和修复所有过快转身事件。
- `scripts/run_v22_*.sh`：建库、建数据、训练和推理入口。

## 3. 安装

```bash
cd /home/disk/lsm/storage/EDGE
unzip -o EDGE_V22_TurnAwarePace_patch.zip

EDGE_ROOT=/home/disk/lsm/storage/EDGE \
  bash EDGE_V22_TurnAwarePace_patch/install_v22.sh

export PYTHONPATH=$PWD:${PYTHONPATH:-}
python tools/smoke_v22_turn_pace.py
```

安装器不会执行 `conda activate`，当前 `edge` 环境保持不变。

## 4. 一整晚重新训练

### 推荐：tmux 一次完成建索引、建数据和训练

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

RUN=output/v22_turn_pace_overnight_$(date +%Y%m%d_%H%M%S)

tmux kill-session -t v22_turn_pace 2>/dev/null || true

tmux new-session -d -s v22_turn_pace \
  "cd /home/disk/lsm/storage/EDGE && \
   export PATH=/home/disk/lsm/conda_envs/edge/bin:\$PATH && \
   export PYTHONPATH=/home/disk/lsm/storage/EDGE:\${PYTHONPATH:-} && \
   export CUDA_VISIBLE_DEVICES=0 && \
   export V22_OVERNIGHT_ROOT='$RUN' && \
   export V22_MOTION_GLOB='data/dunhuang_151d_physical/*.npy' && \
   export V22_DATA_MAX_SAMPLES=12000 && \
   export V22_TURN_EPOCHS=320 && \
   bash scripts/run_v22_prepare_and_train_overnight.sh"

sleep 3
tmux ls
tail -f "$RUN/overnight.log"
```

退出监控但保持训练：`Ctrl+B`，然后 `D`。

### 训练阶段

1. `v21_shared_event_index` → `v22_turn_aware_event_index`
2. 连续敦煌动作 → `data/v22_turn_pace_dataset.npz`
3. 训练 Refiner → `$RUN/train/checkpoints/best.pt`

## 5. 训练完成后运行三首音乐

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

export V22_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v22_turn_aware_event_index
export V22_ROUTER_CKPT=/home/disk/lsm/storage/EDGE/output/v21_music_router_985songs_20260605_154801/seed_20260605/checkpoints/best.pt
export V22_PACE_REFINER_CKPT=/完整路径/v22_turn_pace_overnight_时间/train/checkpoints/best.pt

export V22_AUDIO_GLOB='test_music_bank/dunhuangwu[234].wav'
export V22_RUN_ROOT=output/v22_turn_aware_eval_$(date +%Y%m%d_%H%M%S)
export V22_RENDER=1
export V22_RENDER_FOLLOW=0

bash scripts/run_v22_multi_music.sh
```

## 6. 推荐默认环境开关

```bash
# 检索阶段：避免把本来正常的转身压得过快
export V22_TURN_SPEED_WEIGHT=0.95
export V22_TURN_SPEED_HARD_RATIO=1.55

# 学习式局部修复：只处理超过音乐允许速度的转身
export V22_TURN_REFINE_THRESHOLD_RATIO=1.08
export V22_TURN_REFINE_WINDOW=72
export V22_TURN_REFINE_CONTEXT=10
export V22_TURN_REFINE_STRENGTH=0.90
export V22_TURN_REFINE_MAX_EVENTS=4

# 保持 V21 已验证的风格优先级
export V22_STYLE_WEIGHT=1.45
export V22_MUSIC_WEIGHT=0.70
export V22_EVENT_WEIGHT=0.60
export V22_ACTIVITY_WEIGHT=0.27
```

### 消融：只启用 Turn Gate，不启用 Refiner

```bash
unset V22_PACE_REFINER_CKPT
bash scripts/run_v22_multi_music.sh
```

### 消融：关闭 Turn Gate，只测试 Refiner

```bash
export V22_DISABLE_TURN_GATE=1
export V22_PACE_REFINER_CKPT=/path/to/best.pt
bash scripts/run_v22_multi_music.sh
```

## 7. 评价重点

`V22_EVALUATION.json` 新增：

- `max_yaw_speed_dps`
- `p95_yaw_speed_dps`
- `boundary_mean_velocity_jump`
- `turn_events_refined`
- `refined_peak_before_dps / refined_peak_after_dps`

建议通过标准：

- `mean_style_score >= 0.87`
- `tail_activity >= 0.035`
- `boundary_max_velocity_jump` 不高于 V21
- 被修复转身的峰值角速度下降 20% 以上
- `event_overlap` 与 `family_overlap` 仍接近 0
- fixed-camera 主观观察中不再出现快速甩身

## 8. 论文定位

建议命名：**Turn-Aware Temporal Pace Refinement (TAPR)**。

它是通用模块：处理所有时间位置的转身，不假设“第一个转身”，也不为特定测试视频手工设定帧区间。
