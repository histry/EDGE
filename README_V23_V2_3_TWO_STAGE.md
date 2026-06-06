# V23-v2.3：分桶—残差自然时长预测与两阶段单调时间学习

适用项目：`/home/disk/lsm/storage/EDGE`  
推荐数据：`data/v23_v2_2_slowaware_w120_d88_6k.npz`  
默认窗口：120 帧；自然事件时长：12–88 帧。

## 1. 为什么需要 v2.3

W120 慢动作事件分裂已经有效，但旧联合训练存在：

- 长事件被系统性低估：target P90=84，pred P90≈66；
- Duration MAE≈10.3 帧；
- 最佳 checkpoint 在第 3–4 epoch，说明共享 Encoder 很快过拟合；
- Duration、Tau、Motion、Yaw、Edit 同时训练，梯度相互干扰；
- 三个模型随机种子同时改变了验证源划分，排名不可直接比较。

v2.3 不再修改事件边界，而是升级学习协议。

## 2. 方法结构

### Stage 1：Duration Calibration

输入过快/正常动作，预测：

1. 六分类自然时长区间；
2. 目标区间内部的连续残差；
3. 是否需要执行时间修复。

损失：

- duration-bin cross entropy；
- intra-bin residual loss；
- relative duration loss；
- log-duration loss；
- linear duration loss；
- pairwise ranking loss；
- balanced edit BCE；
- 对长时长样本提高权重。

### Stage 2：Monotonic Time Warp

冻结 Stage 1。单独训练 Tau Encoder：

- 初期使用真实自然时长；
- 随训练逐步切换到 Stage 1 预测时长；
- 仅预测正时间增量，Tau 天然单调且首尾固定；
- 使用 SO(3) 重采样，不预测姿态残差。

### 数据划分

采用固定的 duration-stratified source split：

- 同一原始动作源不会跨 train/val；
- 验证集尽量保持时长桶、identity 和样本量分布；
- 三个模型随机种子共用同一验证源，只改变模型初始化。

## 3. 替换文件

```text
model/v23_monotonic_duration.py
train_v23_monotonic_duration.py

tools/v23_duration_utils.py
tools/build_v23_monotonic_duration_dataset.py
tools/evaluate_v23_checkpoint.py
tools/apply_v23_monotonic_duration.py
tools/smoke_v23_monotonic_duration.py

scripts/run_v23_build_dataset.sh
scripts/run_v23_train_one_seed.sh
scripts/run_v23_duration_overnight.sh
scripts/launch_v23_v2_3_full.sh
scripts/launch_v23_v2_full.sh
```

## 4. 安装

```bash
cd /补丁解压目录/EDGE_V23_v2_3_two_stage
bash install_into_EDGE.sh /home/disk/lsm/storage/EDGE
```

旧文件自动备份至：

```text
/home/disk/lsm/storage/EDGE/backup_v23_before_v2_3_时间戳/
```

## 5. 语法与 Smoke

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python -m py_compile \
  model/v23_monotonic_duration.py \
  train_v23_monotonic_duration.py \
  tools/v23_duration_utils.py \
  tools/build_v23_monotonic_duration_dataset.py \
  tools/evaluate_v23_checkpoint.py \
  tools/apply_v23_monotonic_duration.py \
  tools/smoke_v23_monotonic_duration.py

bash -n scripts/*.sh

python tools/smoke_v23_monotonic_duration.py \
  --window_len 120 \
  --duration_edges 12,24,37,50,63,76,89
```

## 6. 使用现有 W120 数据先跑一个 Seed

现有数据已经通过慢动作事件分裂检查，不必重建：

```bash
cd /home/disk/lsm/storage/EDGE

export CUDA_VISIBLE_DEVICES=0
export V23_DATASET=data/v23_v2_2_slowaware_w120_d88_6k.npz
export V23_REBUILD_DATASET=0
export V23_SEEDS='20260610'

export V23_HIDDEN_DIM=128
export V23_DROPOUT=0.18
export V23_WEIGHT_DECAY=5e-4
export V23_BATCH_SIZE=48

export V23_STAGE1_EPOCHS=160
export V23_STAGE1_PATIENCE=35
export V23_STAGE1_LR=8e-5

export V23_STAGE2_EPOCHS=220
export V23_STAGE2_PATIENCE=50
export V23_STAGE2_LR=1e-4
export V23_TF_START=1.0
export V23_TF_END=0.0
export V23_TF_DECAY_EPOCHS=70

bash scripts/launch_v23_v2_3_full.sh
```

## 7. 三随机种子正式实验

```bash
unset V23_SEEDS
export V23_SEEDS='20260610 20260611 20260612'
export V23_REBUILD_DATASET=0
export V23_DATASET=data/v23_v2_2_slowaware_w120_d88_6k.npz

bash scripts/launch_v23_v2_3_full.sh
```

输出结构：

```text
output/v23_v2_3_two_stage_时间戳/
├── seed_20260610/
│   ├── stage1_duration/
│   ├── stage2_timewarp/
│   ├── BEST_DURATION_CKPT.txt
│   └── BEST_V23_CKPT.txt
├── CHECKPOINT_RANKING.tsv
├── BEST_V23_CKPT.txt
└── heldout_eval_best/
    └── V23_V2_3_HELDOUT_EVALUATION.json
```

## 8. 重新构建 9000 样本数据（可选）

仅在 6K 两阶段结果通过后执行：

```bash
export V23_REBUILD_DATASET=1
export V23_DATASET=data/v23_v2_3_slowaware_w120_d88_9k.npz
export V23_MAX_SAMPLES=9000
export V23_WINDOW_LEN=120
export V23_MIN_TARGET_DURATION=12
export V23_MAX_TARGET_DURATION=88
export V23_DURATION_BINS='auto:6'
export V23_IDENTITY_FRACTION=0.25
export V23_MIN_SPEED_FACTOR=1.15
export V23_MAX_SPEED_FACTOR=3.0

bash scripts/launch_v23_v2_3_full.sh
```

数据构建仍使用 v2.2 Slow-Aware Phase Splitter，并在 NPZ 中额外保存 `duration_edges`。

## 9. 关键环境开关

### 模型与训练

```bash
V23_HIDDEN_DIM=128
V23_DROPOUT=0.18
V23_WEIGHT_DECAY=5e-4
V23_BATCH_SIZE=48
V23_BALANCED_SAMPLER=1
V23_SPLIT_SEED=20260620
V23_SPLIT_TRIALS=4096
```

### Stage 1

```bash
V23_STAGE1_EPOCHS=160
V23_STAGE1_PATIENCE=35
V23_STAGE1_LR=8e-5
V23_LAMBDA_BIN=1.0
V23_LAMBDA_RESIDUAL=1.0
V23_LAMBDA_RELATIVE=1.2
V23_LAMBDA_LOG_DURATION=0.8
V23_LAMBDA_LINEAR_DURATION=0.5
V23_LAMBDA_DURATION_RANK=0.15
V23_LAMBDA_EDIT=0.30
V23_LONG_DURATION_WEIGHT=1.25
```

### Stage 2

```bash
V23_STAGE2_EPOCHS=220
V23_STAGE2_PATIENCE=50
V23_STAGE2_LR=1e-4
V23_TF_START=1.0
V23_TF_END=0.0
V23_TF_DECAY_EPOCHS=70
```

## 10. 接受标准

### Stage 1

```text
Duration correlation ≥ 0.88
Overall Duration MAE ≤ 5 frames
Long-duration MAE ≤ 7 frames
Duration-bin accuracy ≥ 0.75
Edit accuracy ≥ 0.90
Pred P90 与 Target P90 差 ≤ 7 frames
```

### Stage 2

```text
Tau MAE ≤ 0.015
Motion MSE improvement ≥ 65%
Yaw MAE improvement ≥ 45%
Peak error improvement ≥ 40%
Activity preservation ≥ 0.82
Pose-range preservation ≥ 0.95
Identity Tau MAE ≤ 0.01
Edit accuracy ≥ 0.90
Monotonic violation = 0
```

## 11. 在 V21 输出上使用

```bash
RUN=$(ls -td output/v23_v2_3_two_stage_* | head -1)
BEST=$(cat "$RUN/BEST_V23_CKPT.txt")

python tools/apply_v23_monotonic_duration.py \
  --motion output/你的V21结果/dunhuangwu2.npy \
  --checkpoint "$BEST" \
  --out_dir "$RUN/v21_runtime_test"
```

运行时额外检查：

- edit probability；
- duration-bin confidence；
- duration expansion；
- yaw peak；
- activity；
- pose range；
- rotation jump。

不通过时自动保留原 V21 动作。

## 12. 推荐实验顺序

```text
1. 先用现有6K数据跑单Seed
2. 检查Stage-1长时长MAE和P90校准
3. 检查Stage-2 Tau/Yaw/Activity
4. 单Seed通过后跑三个Seed
5. 再决定是否扩展到9K数据
6. 最后接入真实V21多音乐输出
```
