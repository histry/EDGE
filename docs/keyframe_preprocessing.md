# Keyframe Preprocessing Notes

当前模型主体接收的是 151D SMPL motion keyframe，而不是原始 2D 图片。

151D 表示为：

- 4D foot contact
- 3D root position
- 24 joints × 6D rotation = 144D

总计 151D。

## 当前阶段支持的 keyframe 来源

1. 从已有 `[T,151]` motion `.npy` 中抽取指定帧。
2. 从敦煌 `.pkl` 文件中的 `pos` 和 `q` 转成 151D。
3. 使用 checkpoint normalizer 把 physical pose 转成 normalized pose，供 diffusion constraint 使用。

## 尚未完成的部分

真正的 2D skeleton image → 151D SMPL keyframe 还没有端到端实现。

后续需要增加：

1. 2D pose detection / 手工骨架点标注。
2. 2D-to-3D pose lifting。
3. SMPL fitting 或 IK retargeting。
4. 转换为 root position + 24-joint 6D rotation。
5. 与训练 checkpoint 的 normalizer 对齐。

因此当前汇报口径是：

> 当前系统已经支持预处理后的 151D keyframe 控制；2D 骨架图到 3D SMPL keyframe 是下一阶段的前处理模块。