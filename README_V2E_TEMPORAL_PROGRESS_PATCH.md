# EDGE V2E Temporal Progress Patch

## 目的

V2D freeze-aware 解决了完全静止，但没有解决 early endpoint arrival：生成序列很快到达终点姿态，然后 hold / jitter。

本补丁将 Temporal Progress Supervision 加入现有 `_motion_energy_loss(model_x0, target_x0)`，不重写模型结构：

- distance-to-end progress curve matching
- cumulative motion progress matching
- front-loaded motion penalty
- top-k jump dominance penalty
- two-sided motion energy ratio，避免 freeze 和 jitter

## 替换 / 新增文件

把压缩包内容复制到 EDGE 根目录：

- `freeze_aware_motion_patch.py` 替换
- `train.py` 替换
- `tools/diagnose_endpoint_collapse_units.py` 替换/新增
- `scripts/run_train_v2e_temporal_progress_best5.sh` 新增
- `scripts/eval_v2e_temporal_progress_best5.sh` 新增

## 安装

```bash
cd /home/disk/lsm/storage/EDGE
unzip /mnt/data/edge_v2e_temporal_progress_patch.zip -d /tmp/edge_v2e_patch
cp -r /tmp/edge_v2e_patch/* .
chmod +x scripts/run_train_v2e_temporal_progress_best5.sh
chmod +x scripts/eval_v2e_temporal_progress_best5.sh
chmod +x tools/diagnose_endpoint_collapse_units.py
python -m py_compile train.py freeze_aware_motion_patch.py tools/diagnose_endpoint_collapse_units.py
```

## 验证 patch

```bash
export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_TEMPORAL_PROGRESS_SUPERVISION=1
python - <<'PY'
import inspect
from freeze_aware_motion_patch import install_freeze_aware_motion_patch
install_freeze_aware_motion_patch(verbose=True)
from model.diffusion import GaussianDiffusion
src = inspect.getsource(GaussianDiffusion._motion_energy_loss)
print('progress' in src.lower(), 'front' in src.lower(), 'topk' in src.lower())
PY
```

预期：

```text
✅ Installed V2E temporal-progress motion patch ...
True True True
```

## 训练

```bash
tmux new -s v2e_temporal_progress
bash scripts/run_train_v2e_temporal_progress_best5.sh
```

退出 tmux：`Ctrl+B` 然后 `D`。

## 评估

评估脚本使用 soft endpoint + soft temporal anchors，不使用 hard projection：

```bash
bash scripts/eval_v2e_temporal_progress_best5.sh
```

查看：

```bash
cat output/stationary_v2e_best5_temporal_progress_eval/endpoint_collapse_diag_e100.csv
find output/stationary_v2e_best5_temporal_progress_eval -name '*.mp4' | sort
```

## 关键开关

```bash
export EDGE_FREEZE_AWARE_MOTION=1
export EDGE_TEMPORAL_PROGRESS_SUPERVISION=1
export EDGE_PROGRESS_CURVE=gt
export EDGE_PROGRESS_LOSS_WEIGHT=6.0
export EDGE_MOTION_MAX_TARGET_ENERGY_RATIO=2.0
```

如果视频仍然早到终点，提高：

```bash
export EDGE_PROGRESS_LOSS_WEIGHT=8.0
export EDGE_PROGRESS_FRONTLOAD_WEIGHT=1.5
export EDGE_PROGRESS_TOPK_WEIGHT=1.0
```

如果视频抖动过强，降低：

```bash
export EDGE_MOTION_ENERGY_LOSS_SCALE=1.0
export EDGE_MOTION_MAX_TARGET_ENERGY_RATIO=1.5
```
