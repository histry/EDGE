# EDGE V33 Event-Level Contact 代码审计报告

## 1. 当前代码中的根本问题

当前转场数据构建器已经先加载 4,225 个完整事件，然后在事件内部随机选择
`left/right` 并切出 `motion[left+1:right]`。这说明最可靠的接触重建位置不是
28,730 个窗口之后，而是 `motions` 列表生成后、任何窗口采样之前。

旧链路的问题是：

1. 完整事件的前四维 contact 全零；
2. 数据构建器直接把全零 contact 同步带入每个窗口；
3. 后补脚本又在每个窗口上独立估计 contact；
4. 重叠窗口的边缘速度、滤波邻域和地面分位数不同；
5. 同一事件帧可能得到互相矛盾的监督；
6. 原数据集没有保存 `event frame origin`，无法证明重叠一致性；
7. 训练器没有读取 contact confidence，伪标签与真值被同等对待；
8. 原滑步损失主要由预测 contact 概率门控，模型可通过预测无接触降低损失。

## 2. V33 解决方案

```text
4,225 complete events
        ↓
full-event differentiable FK
        ↓
global robust contact calibration
        ↓
per-event Viterbi / run filtering
        ↓
immutable event contact cache
        ↓
synchronised pose + contact slicing
        ↓
overlap consistency assertion
        ↓
confidence-weighted Contact-INR training
```

### 事件级缓存

缓存为每个事件保存：

- event index / event id / event path；
- event length；
- motion SHA1 fingerprint；
- concatenated hard contact `[sumT,4]`；
- concatenated confidence `[sumT,4]`；
- offsets `[N+1]`；
- 全局阈值和校准统计。

缓存指纹不包含原 contact 通道，只包含 root 和 rotations，因此可以安全地为
contact 全零的动作重建标签，同时仍能防止缓存与错误版本动作混用。

### 全局校准

所有完整事件先计算：

- per-foot ground-relative height；
- horizontal speed；
- vertical speed。

再在整个事件库上得到统一尺度和统一 score threshold。默认接触占用目标为：

```text
left ankle  0.42
right ankle 0.42
left toe    0.38
right toe   0.38
```

这不是人工 ground truth，而是用于防止全零和全一坍缩的全局伪标签校准。

### 同步切片

每个真实事件窗口同时切分：

```text
motion[left+1:right]
contact[left+1:right]
confidence[left+1:right]
```

数据集中额外保存：

```text
contact_origin_id
contact_target_start
contact_target_end_exclusive
```

构建结束后遍历所有重叠窗口，并对每个 `(origin_id, original_frame)` 执行：

```python
assert np.array_equal(old_contact, new_contact)
assert np.allclose(old_confidence, new_confidence, atol=1e-7)
```

任何冲突都会阻止数据库保存。

### 置信度加权可微接触损失

V33 将事件级 confidence 传入 trainer：

```text
confidence-weighted BCE
teacher-gated foot skating
confidence-weighted contact height
confidence-weighted swing clearance
confidence-weighted temporal consistency
unconditional penetration loss
```

滑步门控以 target contact 为主、predicted contact 为辅，避免模型通过输出
`contact=0` 来逃避滑步约束。

## 3. 数据声明

当前仍然没有完整长视频 source manifest，因此：

- 事件内部窗口是真实动作监督；
- contact 是 event-level kinematic pseudo label；
- synthetic adjacent 只在 weak 模式作为低权重正则；
- 不能声称已经学习真实 event-to-event gap；
- 论文中应命名为 `V33-Weak`。

找回完整长序列后使用 strict 模式，才可构建真实跨事件边界监督。
