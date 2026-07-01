# V46.9 MotionRAG-Diff Filename-Semantic Project-Aligned Patch

本补丁面向当前 EDGE 项目的实际状态：`change/` 下是 Chang-E 风格 `.bvh` 文件，且没有逐段 BVH-音乐强配对。代码路线不再把 V44 写成强监督 paired music-motion，而是采用 **source-aware Event-RAG + unpaired slot-level semantic grounding + constrained routing + true lower-body IK + residual refiner + conditional residual diffusion**。

## 为什么是 V46.9

- V43：真 Lower-Body IK，写回 EDGE 151D 的 lower-body rot6d 通道，解决 dunhuangwu2 脚滑。
- V44：无配对音乐条件下的 unpaired audio semantic OT，对齐音乐 slot 和动作事件原型。
- V45：残差式 Motion Refiner，修复拼接边界，摆脱纯拼接。
- V46：条件残差扩散重生成，只在 seam / repair mask 附近增强连续性。
- V46.9：新增 Chang-E BVH 高帧率重采样、可选 manifest.csv 上游语义切片、source-aware event metadata 和数据库审计。

## 数据放置

```text
EDGE/change/
  female_36pose_1.bvh
  female_36pose_2.bvh
  female_lotus.bvh
  female_meditation.bvh
  male_36pose_1.bvh
  male_36pose_2.bvh
  male_drum_1.bvh
  male_drum_2.bvh
  male_meditation.bvh
  male_pipa_1.bvh
  male_pipa_2.bvh
  male_ribbon.bvh
```

如果有 `manifest.csv`，放在 `change/manifest.csv`；没有也可以直接按 BVH 自动建库。

## 安装

```bash
cd /mnt/data/V46_8_EDGE_MotionRAG_Diff_PROJECT_PATCH
bash install_v46_motionrag_diff_patch.sh /home/disk/lsm/storage/EDGE
```

## 一键运行

```bash
cd /home/disk/lsm/storage/EDGE
nohup bash scripts/run_v46_build_train_generate_dunhuangwu2.sh \
  > output/v46_8_motionrag_diff_launcher.log 2>&1 &
tail -f output/v46_8_motionrag_diff_launcher.log
```

## 关键开关

```bash
export V46_BVH_RESAMPLE_TO_CONFIG_FPS=1
export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=0
export V46_ENABLE_TRUE_IK=1
export V46_ENABLE_REFINER=1
export V46_ENABLE_DIFFUSION=1
export V46_DEVICE=cuda
export V46_IK_CHUNK_OVERLAP=24
export V46_ROOT_Y_DAMPING_MAX_SECONDS=0.28
```

## 输出

```text
output/v46_8_motionrag_diff_YYYYMMDD_HHMMSS/
  db/events.npz
  db/events_meta.json
  v44_contrastive.pt
  v45_refiner.pt
  v46_diffusion.pt
  dunhuangwu2_v46_8_MotionRAG_Diff.npy
  dunhuangwu2_v46_8_MotionRAG_Diff.report.json
  dunhuangwu2_v46_8_MotionRAG_Diff.mp4
```

## 科研表述边界

本补丁的推荐论文口径是：低资源 motion-only 敦煌舞数据条件下，音乐到动作不是强监督逐帧生成，而是 source-aware 动作事件记忆库上的 slot-level 检索与路由。若没有同名/同步音乐，V44 checkpoint 会记录为 `unpaired_audio_semantic_ot`，而不是 paired supervision。


## V46.9：按 change/*.bvh 文件名重建 source-aware 语义

当前 `EDGE/change/` 下的文件名本身具有语义：`gender + dance category + take`。V46.9 不再把 `female_lotus.bvh`、`female_meditation.bvh`、`male_meditation.bvh`、`male_ribbon.bvh` 这类双 token 文件错误折叠成同一个目录级 source group，而是：

```text
source_uid / source_group = 完整文件名 stem，例如 female_lotus、male_pipa_1
gender = female / male
dance_key = thirty_six_postures / lotus_steps / revelation_meditation / pipa_behind_back / lei_gong_drum / ribbon_flow
take_id = 1 / 2 / -1
semantic_text = 面向 schedule_report 和论文审计的可读语义
name_semantic[32] = 与音乐 slot 特征同构的弱语义先验
```

因此你当前 12 个 BVH 会得到 12 个 source group，而不是 9 个。类别语义只作为检索和未配对音乐语义接地的弱先验，不替代 motion descriptor、contact、boundary、duration 和真实运动学特征。

新增环境开关：

```bash
export V46_SOURCE_GROUP_MODE=filename
export V46_FILENAME_SEMANTIC_ENABLE=1
export V46_FILENAME_SEMANTIC_WEIGHT=0.35
export V46_FILENAME_SEMANTIC_RETRIEVAL_WEIGHT=0.20
export V46_FILENAME_SEMANTIC_OT_WEIGHT=0.35
```
