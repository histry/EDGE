# single_unit45 v15 bridge dense4 success

结论：
- 10000-step checkpoint 视觉效果较好。
- phys MSE ≈ 0.000706。
- rootXZ MSE = 0。
- dense keyframes 全部命中。
- 当前剩余问题：frame-to-frame jump 明显大于 GT，最高约 16x，下一步需要 checkpoint sweep + keyframe density ablation + smooth-recon。

下一步：
1. 比较 train-1000 到 train-10000。
2. 比较 dense4 / dense8 / sparse-mid。
3. 训练 v16 smooth-recon。
4. 扩展到 5–10 个 whitelist stationary units。
