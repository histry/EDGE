# EDGE V34 直接替换与运行指南

## 1. 安装前停止旧任务

```bash
cd /home/disk/lsm/storage/EDGE

tmux ls
ps -ef | grep -E 'schedule_v3|train_v27|run_v3' | grep -v grep
```

确认需要保留的运行已结束，再安装。不要在旧 Python 进程仍加载 V33 模块时覆盖文件。

## 2. 安装

将 ZIP 放到 `/home/disk/lsm/storage/` 后：

```bash
cd /home/disk/lsm/storage
unzip -o EDGE_V34_C3_BOUNDARY_PATCH.zip

python EDGE_V34_C3_BOUNDARY_PATCH/install_v34_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

旧文件自动备份到：

```text
/home/disk/lsm/storage/EDGE/backup_v34_时间戳/
```

检查：

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python -m py_compile \
  tools/v34_boundary_dynamics.py \
  tools/build_v34_contact_event_library.py \
  tools/calibrate_v34_boundary_thresholds.py \
  tools/v34_warp_aware_retrieval.py \
  tools/schedule_v34_whole_song.py \
  tools/v32_transition_quality.py \
  tools/v27_transition_diffusion.py \
  train_v27_transition_diffusion.py

bash -n scripts/run_v34_whole_song.sh
bash -n scripts/run_v34_full_research.sh

python tools/smoke_test_v34_patch.py
```

## 3. 快速验证：复用 V33 checkpoint

该模式不重训，只验证 V34 的调度和边界修复：

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge

tmux kill-session -t v34_inference 2>/dev/null || true

tmux new-session -d -s v34_inference \
  "cd /home/disk/lsm/storage/EDGE && \
   bash scripts/launch_v34_inference_from_v33.sh"
```

实时查看：

```bash
RUN_ROOT=$(cat output/LATEST_V34_INFERENCE_LAUNCH.txt)
echo "$RUN_ROOT"
tail -F "$RUN_ROOT/run.log"
```

该结果应标为：

```text
V34 scheduler + V33 checkpoint compatibility experiment
```

不能标为完整 V34。

## 4. 正式版本：重训 + 严格生成

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge

tmux kill-session -t v34_full 2>/dev/null || true

tmux new-session -d -s v34_full \
  "cd /home/disk/lsm/storage/EDGE && \
   bash scripts/launch_v34_full_overnight.sh"
```

查看：

```bash
RUN_ROOT=$(cat output/LATEST_V34_OVERNIGHT_LAUNCH.txt)
tail -F "$RUN_ROOT/run.log"
```

进入 tmux：

```bash
tmux attach -t v34_full
```

离开但不中止：`Ctrl+B`，松开后按 `D`。

## 5. V34 主流程

```text
V33 event contact cache
→ V34 mirrored contact event library
→ natural-boundary threshold calibration
→ optional full retraining
→ strict warp-aware retrieval
→ septic transition candidate generation
→ cross-boundary absolute gate
→ adaptive exit handshake
→ long-motion/contact/frequency/jitter/boundary evaluation
→ scientific render
```

## 6. 重要环境开关

### 6.1 Contact 回注

```bash
export V33_EVENT_CONTACT_CACHE=data/v33_event_contact_cache.npz
export V34_LIBRARY_DIR=data/v34_event_library
export V34_INDEX_JSON=data/v34_shared_event_index.json
export V34_BUILD_EVENT_LIBRARY=1
```

复用已经审计通过的镜像库：

```bash
export V34_BUILD_EVENT_LIBRARY=0
```

### 6.2 阈值校准

```bash
export V34_CALIBRATE_THRESHOLDS=1
export V34_CALIBRATION_SAMPLES_PER_EVENT=4
export V34_CALIBRATION_QUANTILE=0.995
export V34_CALIBRATION_MULTIPLIER=2.0
```

用户显式设置的 `V34_MAX_*` 优先于自动校准。

### 6.3 Warp-aware 检索

```bash
export V34_WARP_HARD_PRUNE=1
export V34_WARP_MIN=0.82
export V34_WARP_MAX=1.30
export V34_WARP_TOLERANCE=0.0
export V34_WARP_PREFILTER_TOP_K=512
export V34_WARP_PENALTY_WEIGHT=1.25
export V32_MAX_WARP_VIOLATIONS=0
```

正式主实验不要把 violation 数设为 999。

### 6.4 七次边界路径

```bash
export V34_JERK_MATCH_SHRINK=0.35
export V34_VELOCITY_TANGENT_CAP=0.90
export V34_ACCELERATION_TANGENT_CAP=1.40
export V34_JERK_TANGENT_CAP=2.20
```

若真实结果出现 septic overshoot，应优先降低 jerk shrink，而不是提高各类 cap。

### 6.5 Exit Handshake

```bash
export V34_EXIT_HANDSHAKE=1
export V34_EXIT_HANDSHAKE_FRAMES=10
export V34_EXIT_HANDSHAKE_CANDIDATES=8,10,12,16,20
export V34_EXIT_HANDSHAKE_STRENGTH=1.0
export V34_HANDSHAKE_MODE=replace
export V34_HANDSHAKE_MAX_ROTATION_DEG=18
export V34_HANDSHAKE_MAX_ROOT=0.08
```

`replace` 是正式模式。`taper` 仅用于消融，因为渐弱混合可能把出口脉冲移动到握手尾部。

### 6.6 绝对安全门

阈值默认由校准脚本产生。严格模式：

```bash
export V34_POST_HANDSHAKE_ABSOLUTE_VETO=1
export V34_FAIL_ON_UNSAFE_BOUNDARY=1
```

兼容诊断可以设为 0，使程序保存不安全边界报告，但不能用作正式结果。

### 6.7 完整重训

```bash
export V34_TRAIN=1
export V34_DATASET=data/v33_transition_dataset.npz

export V34_AE_EPOCHS=180
export V34_CONTACT_EPOCHS=70
export V34_DIFFUSION_EPOCHS=280

export V34_AE_BATCH_SIZE=36
export V34_CONTACT_BATCH_SIZE=28
export V34_LATENT_BATCH_SIZE=160

export V34_W_ENDPOINT_ACCELERATION=0.45
export V34_W_ENDPOINT_JERK=0.20
```

4090 OOM 时：

```bash
export V34_AE_BATCH_SIZE=24
export V34_CONTACT_BATCH_SIZE=18
export V34_LATENT_BATCH_SIZE=96
export V34_DECODED_BATCH_LIMIT=4
```

## 7. 输出目录

正式运行包括：

```text
v34_event_library/
v34_shared_event_index.json
v34_boundary_thresholds.json
v34_boundary_thresholds.env
v34_contact_inr_training/checkpoints/best.pt
septic_handshake_baseline/
v34_contact_inr/
v34_transition_gate_summary.json
```

每首音乐包括：

```text
*_v26.npy
*_v26.schedule_report.json
*_v34.long_dance.json
*_v34.public_metrics.json
*_v34.frequency_foot.json
*_v34.contact_metrics.json
*_v34.jitter.json
*_v34.boundary_v34.json
*_scientific_fixed.mp4
```

## 8. 验收条件

严格版本应同时满足：

```text
warp violations = 0
unsafe post-handshake boundaries = 0
contact output 不再全零
exit jerk max 显著下降
worst frames 不再集中于 transition end
BAS 不明显下降
FGDkin 优于 V33 或至少不显著恶化
```

重点比较：

```text
V33 Contact-INR
V34 septic baseline + handshake
V34 Contact-INR full
V34 without Contact back-injection
V34 without absolute gate
V34 without handshake
V34 without warp hard pruning
V34 quintic p/v/a vs regularised septic p/v/a/j
```

## 9. Git 提交

```bash
cd /home/disk/lsm/storage/EDGE

git add .
git commit -m "Upgrade whole-song stitching to V34 contact-consistent C3 boundaries"
git push origin main
```
