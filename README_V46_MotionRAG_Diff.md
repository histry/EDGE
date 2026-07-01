# V46 MotionRAG-Diff for EDGE 151D Dunhuang Whole-Song Generation

本补丁面向当前 `histry/EDGE` 仓库的 V42.2 状态继续升级。V42.2 的核心是 root-footplant 物理后处理与 IK target 生成；V46 将路线升级为：

- **V43 真 Lower-Body IK**：直接优化并写回 EDGE 151D 的 lower-body 6D rotation 通道，同时允许小幅 root 修正，解决脚滑时不再只保存 foot target。
- **V44 音乐—动作对比学习**：从新 `change` 数据集重建 source-aware event database，并训练音乐/动作共享嵌入用于检索对齐。
- **V45 残差式 Motion Refiner**：训练 temporal residual refiner，对拼接边界做残差修复，摆脱纯拼接。
- **V46 条件扩散重生成**：以检索动作、音乐条件和 seam mask 为条件，做 residual diffusion regeneration，最后再接 V43 IK。

## 安装

```bash
cd /mnt/data/V46_EDGE_MotionRAG_Diff_PATCH
bash install_v46_motionrag_diff_patch.sh /home/disk/lsm/storage/EDGE
```

## 一键重建数据库、训练并生成 dunhuangwu2

```bash
cd /home/disk/lsm/storage/EDGE
# 推荐把新 Chang-E/change 数据转换后的 EDGE-151 .npy/.npz/.pkl 放到 ./change
nohup bash scripts/run_v46_build_train_generate_dunhuangwu2.sh \
  > output/v46_motionrag_diff_launcher.log 2>&1 &
tail -f output/v46_motionrag_diff_launcher.log
```

## 只对已有 motion 运行 V43 真 IK

```bash
cd /home/disk/lsm/storage/EDGE
bash scripts/run_v43_true_ik_on_existing_motion.sh output/xxx/dunhuangwu2_xxx.npy
```

## 环境开关

```bash
export V46_ENABLE_TRUE_IK=1      # 开启/关闭 V43 真 lower-body IK
export V46_ENABLE_REFINER=1      # 开启/关闭 V45 residual refiner
export V46_ENABLE_DIFFUSION=1    # 开启/关闭 V46 conditional diffusion
export V46_DEVICE=cuda           # cuda 或 cpu
export V46_IK_ITERS=90           # IK 迭代次数
export V46_DIFFUSION_STEPS=50    # 采样步数
export V46_TOP_K=40
export V46_BEAM_SIZE=10
export V46_ENABLE_ROOT_Y_PHYSICS=1 # 开启 C1-safe root-Y 飞行/落地阻尼
export V46_IK_SLIDE_RELEASE_M=0.12 # 接触脚 XZ 游走超过 12cm 时放行滑步
```

## 输出

一键脚本会写入：

```text
output/v46_motionrag_diff_YYYYMMDD_HHMMSS/
  db/events.npz
  db/events_meta.json
  v44_contrastive.pt
  v45_refiner.pt
  v46_diffusion.pt
  dunhuangwu2_v46_MotionRAG_Diff.npy
  dunhuangwu2_v46_MotionRAG_Diff.report.json
  dunhuangwu2_v46_MotionRAG_Diff.mp4  # render_from_npy.py 与音频存在时生成
```

## 数据要求

`build-db` 接受 `.npy/.npz/.pkl/.pickle`，数组需要为 EDGE 151D：

```text
[T,151] 或 [N,T,151]
root translation = motion[:,4:7]
local joint rot6d = motion[:,7:151].reshape(T,24,6)
```

如果你的 Chang-E/change 数据是 BVH，需要先用现有 EDGE/SMPL 转换脚本转换为 EDGE-151 后放入 `change/`。
