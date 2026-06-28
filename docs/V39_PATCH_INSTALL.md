# V39 Contact Stability Scientific Patch

## 目标

V39 面向 V34/V38 后仍存在的脚部接触不稳和局部高频抖动问题。它不改变宏观 RAG / Dense Boundary 路线，而是在运动质量后处理层增加更稳定的微观物理约束：

1. contact confidence + hysteresis：融合 contact label、足部高度、足部 XZ 速度，使用 on/off 双阈值避免接触标签闪烁。
2. support-aware footplant root solver：用所有接触脚的鲁棒 anchor 解 root X/Z 修正，并限制 root correction 的速度和加速度。
3. residual-only Butterworth：只滤除 IK / foot lock 产生的残差高频，不对原始舞蹈低频身法做全局平滑。

## 安装

```bash
cd /mnt/data/V39_Contact_Stability_Scientific_Patch
mkdir -p /tmp/V39_Contact_Stability_Scientific_Patch
cp -r * /tmp/V39_Contact_Stability_Scientific_Patch/

bash install_v39_contact_stability_patch.sh /home/disk/lsm/storage/EDGE
```

## 先不重训，重处理已有结果

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v39_contact_stability_env.sh

bash scripts/run_v39_reprocess_motion.sh \
  output/v38_source_aware_full_train_20260625_212818/v34_contact_inr/dunhuangwu2_v26.npy
```

输出：

```text
*_v39_contact_stable.npy
*_v39_contact_stable.motion_quality_postprocess.v39.json
*_v39_contact_stable.contact_stability_audit.v39.json
```

## 整晚重训

```bash
cd /home/disk/lsm/storage/EDGE
nohup bash scripts/run_v39_overnight_source_aware_full_train.sh \
  > output/v39_overnight_launcher.log 2>&1 &
```

查看：

```bash
RUN_ROOT=$(cat output/LATEST_V39_CONTACT_STABLE_FULL_TRAIN.txt)
tail -f "$RUN_ROOT/run.log"
```

## 推荐默认开关

```bash
source scripts/v39_contact_stability_env.sh
```

关键默认值：

```bash
V39_CONTACT_HYSTERESIS=1
V39_CONTACT_ON_THRESHOLD=0.58
V39_CONTACT_OFF_THRESHOLD=0.42
V38_CONTACT_CLOSE_HOLES=7
V38_CONTACT_OPEN_SPIKES=4
V39_FOOTPLANT_SOLVER=1
V38_CONTACT_LOCK_BLEND_FRAMES=6
V34_CONTACT_LOCK_STRENGTH=0.88
V39_ROOT_CORR_MAX_STEP=0.026
V39_ROOT_CORR_MAX_ACCEL=0.012
V39_SUPPORT_ROOT_VELOCITY_DAMPING=0.10
V39_BUTTERWORTH_RESIDUAL_ONLY=1
V38_BUTTERWORTH_CUTOFF_HZ=4.2
V38_BUTTERWORTH_STRENGTH=0.78
V34_OUTPUT_SMOOTH=0
```

## 如果脚滑仍明显

```bash
export V38_CONTACT_CLOSE_HOLES=9
export V38_CONTACT_OPEN_SPIKES=4
export V34_CONTACT_LOCK_STRENGTH=0.92
export V39_SUPPORT_ROOT_VELOCITY_DAMPING=0.14
```

## 如果动作变僵或脚像被焊住

```bash
export V34_CONTACT_LOCK_STRENGTH=0.80
export V39_SUPPORT_ROOT_VELOCITY_DAMPING=0.06
export V39_ROOT_CORR_MAX_STEP=0.020
```

## 如果手部仍抖

```bash
export V38_BUTTERWORTH_CUTOFF_HZ=3.6
export V38_BUTTERWORTH_STRENGTH=0.86
```

## 如果手部过软

```bash
export V38_BUTTERWORTH_CUTOFF_HZ=5.0
export V38_BUTTERWORTH_STRENGTH=0.60
```
