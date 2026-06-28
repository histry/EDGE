# V38 Contact Denoise + SmoothStep Lock + Butterworth Filter Patch

## 1. 主要改动

本补丁只替换 `tools/v34_motion_quality_postprocess.py`，并新增一个 delta 工具和两个脚本。现有 `run_v34_full_research.sh` 已经调用该文件，因此不需要重写整条长训练脚本。

修复点：

1. `contact_denoise`：在 IK 锁脚前对 contact mask 做中值滤波、短离地闭合、短伪接触剔除，避免 contact binary 毛刺导致脚踝/膝盖/骨盆反复锁定-释放。
2. `smoothstep contact lock`：每个 contact segment 前后用 `w(t)=3t^2-2t^3` 做 S 型软进入/软退出，避免落地第一帧锁定权重突变。
3. `butterworth_filter`：仅对手、腕、肘、肩等 IK 敏感关节的 6D rotation 做 3-5Hz 低通滤波，避免全局平滑破坏敦煌舞低频身法。

## 2. 安装

```bash
cd /mnt/data/V38_Contact_Denoise_SmoothStep_Filter_Patch
mkdir -p /tmp/V38_Contact_Denoise_SmoothStep_Filter_Patch
cp -r * /tmp/V38_Contact_Denoise_SmoothStep_Filter_Patch/
bash install_v38_contact_patch.sh /home/disk/lsm/storage/EDGE
```

## 3. 先重处理已有输出

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v38_contact_jitter_env.sh
bash scripts/run_v38_reprocess_motion.sh   output/v36_source_aware_full_train_20260625_003612/v34_contact_inr/dunhuangwu2_v26.npy
```

输出：

```text
*_v38_contact_filtered.npy
*.motion_quality_postprocess.json
*.motion_quality_postprocess.delta.json
```

## 4. 用于后续完整训练/推理

```bash
cd /home/disk/lsm/storage/EDGE
source scripts/v38_contact_jitter_env.sh
bash launch_v34_source_aware_rag.sh
```

## 5. 推荐开关

```bash
export V38_CONTACT_DENOISE=1
export V38_CONTACT_MEDIAN_SIZE=5
export V38_CONTACT_CLOSE_HOLES=5
export V38_CONTACT_OPEN_SPIKES=3
export V38_CONTACT_LOCK_BLEND_FRAMES=5

export V38_BUTTERWORTH_FILTER=1
export V38_BUTTERWORTH_CUTOFF_HZ=4.0
export V38_BUTTERWORTH_ORDER=2
export V38_BUTTERWORTH_STRENGTH=0.85
export V38_BUTTERWORTH_JOINTS=lelbow,relbow,lwrist,rwrist,lhand,rhand,lshoulder,rshoulder
```

如果还滑脚，先加强 contact 去噪和锁脚：

```bash
export V38_CONTACT_CLOSE_HOLES=7
export V38_CONTACT_OPEN_SPIKES=4
export V38_CONTACT_LOCK_BLEND_FRAMES=6
export V34_CONTACT_LOCK_STRENGTH=0.90
```

如果手部速度被削弱，则放宽低通：

```bash
export V38_BUTTERWORTH_CUTOFF_HZ=5.0
export V38_BUTTERWORTH_STRENGTH=0.65
```
