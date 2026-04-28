# Model Optimization Notes

本轮优化目标：降低 foot sliding，并避免 contact channels 在生成时整体饱和。

## 已修改内容

### 1. 物理 loss 改到真实物理空间

训练数据进入模型前是 z-score normalized 的 151 维特征。此前部分 FK、foot、body stability、turn 等物理 loss 直接在 normalized tensor 上解释 root 位移和 6D rotation，物理量纲不一致。

现在在 `model/diffusion.py` 中：

1. reconstruction / feature velocity loss 仍在 normalized 空间计算。
2. FK、foot slide、body stability、turn、motion energy、MMR motion encoding 等物理相关项先通过 normalizer 反归一化，再计算。

这样 foot loss 的梯度对应真实米制 root/foot 位移，rotation_6d 也回到原始 6D rotation 表示。

### 2. 新增 contact channel loss

新增 `Contact Loss`，直接约束反归一化后的 4 维 foot contact channel 接近 0/1 目标。

目的：

- 避免生成时 contact channel 全部接近 1。
- 让 foot loss 和后处理 foot lock 使用更可信的接触信号。

### 3. 提高 foot/sync 权重

新增训练参数：

```bash
--contact_loss_weight 0.8
--foot_loss_weight 2.5
--sync_loss_weight 1.2
```

默认值已经写入 `args.py`。物理项仍保留前 10 epoch warmup，避免一开始破坏 reconstruction。

## 建议 fine-tune 命令

从当前 stage2B checkpoint 继续短训：

```bash
/home/disk/lsm/conda_envs/edge/bin/python train.py \
  --project runs/train \
  --exp_name exp_dunhuang_stage2B_phys_opt \
  --data_path data/dunhuang_bvh/processed \
  --checkpoint runs/train/exp_dunhuang_keyframe_stage2B_stability_from102/weights/train-5.pt \
  --train_stage stage2 \
  --epochs 20 \
  --save_interval 5 \
  --batch_size 64 \
  --learning_rate 1e-4 \
  --mmr_loss_weight 0 \
  --keyframe_condition_prob 0.8 \
  --keyframe_condition_width 3 \
  --mid_keyframe_condition_prob 0.8 \
  --mid_keyframe_count 2 \
  --mid_keyframe_condition_width 3 \
  --contact_loss_weight 0.8 \
  --foot_loss_weight 2.5 \
  --sync_loss_weight 1.2
```

如果显存不够，把 `--batch_size` 降到 32。

## 训练后验证

1. 用新 checkpoint 重新生成 v5/v6 风格输出。
2. 跑 `eval_quantitative.py`。
3. 重点观察：

```text
foot_slide_rate
foot_contact_speed_p95_mps
BeatAlign symmetric
keyframe_mpjpe_m_mean
trajectory_ade_m
```

理想结果：

- `foot_slide_rate` 降低。
- `foot_contact_speed_p95_mps` 降低。
- 关键帧误差不明显升高。
- v5 轨迹误差仍接近 0。
