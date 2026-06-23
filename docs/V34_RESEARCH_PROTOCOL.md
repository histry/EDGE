# V34 科研实验协议

## 主假设

V34 的提升来自边界动力学与数据条件一致性，而不是单纯增加网络规模。

## 必做消融

| 实验 | Contact 回注 | Warp hard prune | Septic | Absolute gate | Handshake |
|---|---:|---:|---:|---:|---:|
| V33 | 否 | 否 | 否 | 否 | 否 |
| A | 是 | 否 | 否 | 否 | 否 |
| B | 是 | 是 | 否 | 否 | 否 |
| C | 是 | 是 | 是 | 否 | 否 |
| D | 是 | 是 | 是 | 是 | 否 |
| V34 Full | 是 | 是 | 是 | 是 | 是 |

另加：

- Generalized quintic：匹配 p/v/a；
- Regularised septic：匹配 p/v/a + 收缩后的 jerk；
- Handshake 8/12/16/20 帧；
- 固定 5000 阈值 vs 数据校准阈值；
- V33 checkpoint 兼容推理 vs V34 full retrain。

## 主要指标

### 边界指标

- exit pose/geodesic step；
- exit FK jump；
- exit velocity mismatch；
- exit acceleration；
- cross-boundary jerk max/P95；
- angular jerk max/P95；
- unsafe boundary count。

### 足部指标

- nonzero contact rate；
- contact/proxy agreement；
- contact-conditioned foot slip；
- penetration；
- ankle/toe jerk。

### 整曲指标

- BAS；
- FGDkin；
- DCT high-frequency ratio；
- exact length match；
- warp violation count；
- candidate acceptance/fallback rate。

## 公平性控制

所有对比固定：

- 三首测试音乐；
- 音乐分段参数；
- Event-RAG 候选库；
- 随机种子；
- 渲染相机；
- `render_smooth_window=1`；
- 不允许使用视频渲染后处理掩盖动作问题。

## 论文表述边界

推荐：

> event-level kinematic pseudo-contact supervision

不推荐：

> manually annotated physical contact ground truth

推荐：

> regularised C3 boundary matching in the SO(3) tangent representation

不推荐：

> mathematically exact C3 continuity for arbitrary human motion

推荐：

> empirically calibrated absolute safety limits

不推荐：

> universal biomechanical physical constants
