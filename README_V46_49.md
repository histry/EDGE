# V46.49 Optimization-Retarget + Gravity Contract

## 结论

V46.47 的 `yaw-only root` 只能令根旋转矩阵的 up 轴竖直，不能保证
`pelvis -> head` 人体轴竖直。当前 Chang-E 加载器仍将源 BVH 的局部旋转按
关节名称直接复制到 EDGE/SMPL-like 骨架；由于两套骨架的 bind pose、
bone roll 和 joint-local frame 不一致，数值合法的 Rot6D 仍可能生成横躺人体。

V46.49 将数据入口改为：

```text
raw Chang-E 6DoF BVH
  -> source BVH FK global keypoints
  -> rest-pose weighted similarity calibration
  -> optimization-based target IK/retargeting
     (root orientation + root translation + 24-joint pose + global scale)
  -> EDGE151D gravity contract
  -> Event-RAG slicing
  -> V44 / gravity-regularised V45 / gravity-regularised V46
  -> V46.49 closed-loop + final gravity hard gate
```

不再把 `yaw-only` 当作正式重定向算法。

## 包含文件

### 直接替换

- `render_from_npy.py`
  - 保留 column-concat Rot6D；
  - 渲染前强制人体重力合同审计；
  - 默认科研渲染：fixed camera、smooth=1。

### 新增核心文件

- `tools/chang_e_edge_retarget.py`
  - 完整解析 6DoF BVH；
  - 先在源骨架做 FK；
  - 以全局 3D keypoints 优化 EDGE 24-joint 目标骨架；
  - 输出合法 EDGE151D。
- `tools/v46_49_gravity_contract.py`
  - NumPy/Torch FK；
  - torso-up、head-above-pelvis、feet-below-pelvis；
  - 可微 gravity loss。
- `tools/v46_49_build_retarget_cache.py`
  - 将 `change/**/*.bvh` 批量转换为 `.npy`。
- `tools/v46_49_audit_gravity_contract.py`
  - 对 retarget cache、Event-RAG DB、最终输出做硬审计。
- `tools/apply_v46_49_gravity_training_patch.py`
  - 只修改最新 `v46_motionrag_diff.py` 中 V45/V46 两处 loss；
  - 自动备份，不覆盖其他 V46.38–V46.47 修复。
- `tools/v46_49_boundary_closed_loop.py`
  - 包装现有 V46.46；
  - 合并 1 秒以下尾部残余 slot；
  - 空 transition 按 direct join 审计；
  - 最终结果必须通过重力合同。
- `tools/v46_49_make_final_mssd.py`
  - 从 V26 schedule report 构造正式 final MSSD；
  - 或复用旧 report 做严格受控对照。
- `scripts/run_v46_49_full_retarget_retrain.sh`
  - 重定向、审计、重建 DB、AESD、split、V44/V45/V46 重训、生成、渲染。
- `configs/v46_49_retarget.env.example`
- `tests/test_v46_49_retarget_contract.py`

## 安装

```bash
cd /解压后的/v46_49_retarget_solution
bash install_v46_49.sh /home/disk/lsm/storage/EDGE
```

安装器会把已有同名文件备份到：

```text
EDGE/output/v46_49_install_backup_<timestamp>/
```

然后：

```bash
cd /home/disk/lsm/storage/EDGE
source configs/v46_49_retarget.env.example
```

## 正式实验的 MSSD

正式主实验必须使用 V21/V26/V23 输出的 final schedule：

```bash
export V26_REPORT="output/.../dunhuangwu2_v26.schedule_report.json"
```

已有 final MSSD 时：

```bash
export FINAL_MSSD="output/.../dunhuangwu2.final.mssd.json"
```

仅做与 V46.47 的受控对照时：

```bash
export PREVIOUS_REPORT="output/v46_47_yaw_full_20260709_222231/dunhuangwu2_v46_47_yaw_closed_loop.report.json"
```

不能把 `usage=train_semantic, is_final_schedule=false` 的 sidecar 静默当作最终调度。

## 运行

```bash
OUT_ROOT="output/v46_49_retarget_full_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run_v46_49_full_retarget_retrain.sh \
  > "output/v46_49_launcher_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

监控：

```bash
tail -f output/v46_49_launcher_*.log
```

## 先做单文件验收

正式批量处理前，建议先选一个 BVH：

```bash
python tools/chang_e_edge_retarget.py \
  --input change/某个文件.bvh \
  --output output/v46_49_single_test.npy \
  --report output/v46_49_single_test.retarget.json
```

审计：

```bash
python tools/v46_49_audit_gravity_contract.py \
  --input output/v46_49_single_test.npy \
  --out output/v46_49_single_test.gravity.json
```

渲染：

```bash
python render_from_npy.py \
  --motion output/v46_49_single_test.npy \
  --audio test_music_bank/dunhuangwu2.wav \
  --output output/v46_49_single_test.mp4 \
  --camera_mode fixed \
  --render_smooth_window 1
```

只有单文件能够稳定站立，才允许批量建库。

## 推荐硬门槛

默认重力合同：

```text
torso_up_cos_p05          >= 0.45
torso_up_cos_median       >= 0.70
head_above_pelvis_ratio   >= 0.92
feet_below_pelvis_ratio   >= 0.90
horizontal_body_ratio     <= 0.10
nonfinite_count           == 0
retarget_fit_rmse_p95     <= 0.18 m
```

敦煌舞存在下腰、飞天和曲线姿态，所以没有把单帧躯干弯曲全部判错；门槛针对的是
“持续横躺/整体倒伏”，而不是抹除艺术动作。

## 为什么必须重建数据库并重训

旧 Event-RAG event 已经包含错误局部骨架坐标解释。V44、V45 和 V46 的训练样本都来自
这些 event。只修 renderer 或 final IK 不能恢复原始动作几何，因此旧 DB 和旧 checkpoint
只能作为失败对照，不能作为 V46.49 主实验。
