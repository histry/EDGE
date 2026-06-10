# EDGE V31 安全优先科研版：代码审计与替换说明

## 1. 审计结论

V30/V29 并没有解决全部审稿风险，而且“结果更加抖动”与代码结构一致，
不是单纯调参问题。主要原因：

1. V30 使用最高 `2^9` 的 Fourier 时间频率，叠加高频 Sine 网络；
2. 单关节旋转残差上限为 0.40 rad，推理默认信任权重为 0.85；
3. 潜扩散 checkpoint 主要按 latent noise/x0 loss 选择；
4. 每个转场生成后直接采用，没有与确定性基线比较和自动回退；
5. V30 VAE 使用随机 posterior latent 重构，但潜扩散训练 posterior mean，
   存在训练—推理潜变量分布错位；
6. 合成相邻桥接和 pseudo pair 默认仍可进入训练；
7. 数据构建器曾用 `boundary + max_len` 判断完整源序列长度，
   会误拒绝序列尾部的有效真实边界；
8. 外部 prior 被错误标为 `real_target=True`；
9. DCT evaluator 的 `_spectrum_power()` 自递归，调用会无限递归；
10. V30 用“与规则查询最近的短语”恢复几何音乐条件，
    对重复或近似短语存在错配风险。

V31 不再增加生成自由度，而是限制自由度。

## 2. V31 主算法

```text
音乐边界锁定 + V23 时长 + Event-RAG/图调度
                    ↓
确定性 C² quintic SO(3) 转场基线
                    ↓
低阶 C²-zero Legendre 残差系数
                    ↓
PCA 系数潜空间条件扩散
                    ↓
多候选采样
                    ↓
真实前后动作上下文安全门
                    ↓
安全候选 / 自动回退确定性基线
```

残差基函数：

```text
E(t) = 64 t^3 (1-t)^3
R(t) = E(t) Σ c_k P_k(2t-1)
```

`E(t)` 在两端的值、一阶导数和二阶导数均为零，因此学习残差不能破坏
确定性基路径的端点姿态、速度和加速度。

V31 删除：

- SIREN 高频 INR；
- 2^9 Fourier 时间编码；
- 转场 contact diffusion；
- VAE posterior sampling；
- 默认 0.85 生成接管；
- 无条件采用扩散候选；
- 默认 synthetic/pseudo 转场训练。

## 3. 安装前

不要在旧任务仍运行时覆盖代码。确认旧 tmux 任务已结束：

```bash
tmux ls
ps -ef | grep -E 'train_v27|schedule_v3|run_v3' | grep -v grep
```

## 4. 安装

```bash
python install_v31_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

旧文件自动备份到：

```text
/home/disk/lsm/storage/EDGE/backup_v31_时间戳/
```

## 5. 必须准备的真实数据

### 5.1 完整长动作与同步音频 manifest

```json
{
  "sources": {
    "source_001": {
      "motion": "/abs/source_001.pkl",
      "audio": "/abs/source_001.wav"
    }
  },
  "events": {
    "event_001": {
      "source_id": "source_001",
      "source_motion": "/abs/source_001.pkl",
      "source_start": 120,
      "source_end": 173
    }
  }
}
```

必须明确 `source_start/source_end` 语义。V31 报告中固定记录：

```text
source_start inclusive
source_end treated as indexed endpoint
real gap = source_end+1 : next_source_start
```

### 5.2 专家复核的音乐—动作配对

```jsonl
{"audio":"/abs/source_001.wav","start_frame":120,"end_frame":174,"event_index":0,"group":"source_001","weight":1.0}
```

旧 schedule 导出的配对只能作弱监督预训练，不能作为最终论文的真实配对证据。

## 6. 一键严格运行

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

export V26_INDEX_JSON=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json
export V26_DURATION_INDEX_NPZ=data/v26_music_dominant_duration_index.npz
export V26_ROUTER_CKPT=output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt
export V26_V23_CKPT=output/v23_v2_5_continuous_gate_20260607_004858/seed_20260612/stage2_timewarp/checkpoints/best.pt
export V26_PLANNER_CKPT=$(cat output/v26_music_dominant_whole_song_planner_985/BEST_V26_PLANNER_CKPT.txt)
export V26_START_POSE=data/canonical_dunhuang_start_pose.npy
export V27_HYPERBOLIC_CKPT=/path/to/hyperbolic/best.pt

export V27_CLAP_CKPT=$PWD/pretrained/laion_clap/music_audioset_epoch_15_esc_90.14.pt
export V27_CLAP_AMODEL=HTSAT-base
export V27_CLAP_DEVICE=cuda:0
export V27_CLAP_ENABLE_FUSION=0
export V27_CLAP_USE_FILELIST=0
export V27_DEEP_MUSIC_FEATURES=1
export V27_REQUIRE_DEEP_MUSIC=1

export V31_SOURCE_MANIFEST=data/v31_source_manifest.json
export V31_PAIR_MANIFEST=data/v31_pairs_expert_checked.jsonl
export V31_AUDIO_ROOT=/path/to/source_audio

export V26_MUSIC='test_music_bank/dunhuangwu2.wav;test_music_bank/dunhuangwu3.wav;test_music_bank/dunhuangwu4.wav'
export V31_KEYS='dunhuangwu2;dunhuangwu3;dunhuangwu4'

bash scripts/run_v31_full_research.sh
```

## 7. 默认安全参数

```bash
export V31_BASIS_COUNT=6
export V31_PCA_DIM=96
export V31_ROTATION_RESIDUAL_CAP=0.16
export V31_ROOT_Y_RESIDUAL_CAP=0.045

export V31_CANDIDATES=6
export V31_GUIDANCE=1.0
export V31_RESIDUAL_TRUST=0.20
export V31_MAX_TOTAL_RISK_RATIO=1.02
export V31_MAX_BOUNDARY_RATIO=1.04
export V31_MAX_JERK_RATIO=1.03
export V31_MAX_FOOT_RATIO=1.05
export V31_MAX_ROTATION_STEP_RAD=0.22

export V31_ENABLE_EDGE_DAMPING=0
export V31_ENABLE_GEOMETRIC_RETRIEVAL=0
export V31_GEOMETRIC_RETRIEVAL_WEIGHT=0.0
```

不要先提高 trust。推荐消融：

```text
0.00 / 0.10 / 0.20 / 0.30
```

只有 `0.20` 相对 C² baseline 在三个随机种子中稳定改善，才将其作为主模型。

## 8. 真实数据硬门

```bash
export V31_REQUIRE_REAL_BOUNDARY_SAMPLES=1000
export V31_REQUIRE_UNIQUE_BOUNDARIES=250
export V31_REQUIRE_REAL_BOUNDARY_RATIO=0.10
export V31_REQUIRE_REAL_MUSIC_SAMPLES=1000
export V31_REQUIRE_REAL_MUSIC_RATIO=0.10
```

注意：

- 多次遮挡同一边界只算一个 unique boundary；
- external prior 不计 `real_target`；
- synthetic adjacent 和 pseudo pair 在主训练中固定为 0；
- 达不到门槛时应补数据，而不是降低门槛并继续宣称真实转场学习。

## 9. 双曲检索使用规则

默认关闭：

```bash
export V31_ENABLE_GEOMETRIC_RETRIEVAL=0
```

先查看：

```text
retrieval_geometry_audit.json
```

只有当：

```text
Poincare Recall@10 > Euclidean Recall@10
且 bootstrap 95% CI 下界 > 0
```

才运行：

```bash
export V31_ENABLE_GEOMETRIC_RETRIEVAL=1
export V31_GEOMETRIC_RETRIEVAL_WEIGHT=0.25
```

否则论文中应如实报告双曲版本没有显著优势，而不是将其包装为主贡献。

## 10. 生成结果中的安全门证据

脚本输出：

```text
v31_transition_gate_summary.json
```

其中必须报告：

- sampled candidate count；
- candidate acceptance rate；
- C² fallback rate；
- 每个边界的 baseline risk；
- 被拒候选的具体失败检查。

高 fallback rate 不代表系统失败，而说明学习模型尚未稳定优于确定性基线。
但若主模型几乎全部 fallback，也不能宣称扩散模块带来显著贡献。

## 11. 正确的频域评价

V31 修复了 V30 `_spectrum_power()` 自递归错误。输出：

- whole-song DCT jerk spectrum；
- transition-only DCT jerk spectrum；
- `<2 Hz / 2–6 Hz / >6 Hz` 能量；
- spectral entropy；
- contact-conditioned foot sliding；
- transparent multi-scale RBF MMD；
- transparent foot-motion Fréchet。

这些是内部透明指标，不可冒充某个未经统一定义的社区标准 MMMD/FMD。

## 12. 审稿风险是否被“解决”

### Frankenstein Pipeline

代码层面已减轻：

- V26 是唯一整曲规划器；
- V31 只替换几何和转场生成；
- 几何检索默认关闭；
- edge damping 默认关闭；
- old transition refiner checkpoint 强制为空；
- 每个学习模块都有失败回退。

但论文仍必须提供：

- 模块级错误传播分析；
- 复杂度、显存和耗时；
- 每个模块的独立消融；
- 简化版与完整版比较。

### Math Washing

代码层面增加了 held-out Euclidean/Poincaré audit，且默认关闭双曲得分。
是否解决取决于真实实验，而不是代码名称。

### Ouroboros Effect

代码层面修复了统计和门控：

- unique real boundaries；
- real boundary mask；
- synthetic off；
- external prior 不算 real；
- source tail 不再被错误长度条件拒绝。

但没有完整长序列就无法真正解决。代码不能制造真实过渡。

### Baselines

代码不能替你完成 LODGE++、OpenDanceNet、MotionRAG-Diff 的公平复现。
需要统一数据、音乐、长度、随机种子、渲染和评价协议。

## 13. 必做主表与消融

主表：

```text
EDGE
LODGE / LODGE++
OpenDanceNet
MotionRAG-Diff
C² deterministic baseline
V31 full
```

消融：

```text
V31 without learned residual
V31 trust 0.10/0.20/0.30
V31 without candidate gate
V31 synthetic-only
V31 real-boundary-only
Euclidean retrieval
Poincare retrieval
No graph edge cost
No music condition
```

每项至少三个随机种子，并报告均值、标准差和显著性检验。

## 14. 锁定音乐边界与自然时长的硬冲突

原 V26 在 `lock_music_boundaries=1` 时会直接使用 `phrase_length-transition_length`
作为动作内容长度；即使该长度超过 `min_time_warp/max_time_warp`，也只在报告中
标记 override，而不会阻止生成。V31 默认开启：

```bash
export V31_STRICT_LOCKED_WARP=1
export V31_MAX_WARP_VIOLATIONS=0
export V31_WARP_TOLERANCE=0.02
```

超限时任务直接失败。应通过重新分段、检索更匹配自然时长的事件或减少转场预算
解决，不能继续强行压缩/拉伸后把抖动归因于转场模型。

## 15. 4090 推荐训练资源

```bash
export V31_BATCH_SIZE=128
export V31_FIT_BATCH_SIZE=64
export V31_DECODED_BATCH_LIMIT=12
```

显存不足时先改为：

```bash
export V31_BATCH_SIZE=64
export V31_FIT_BATCH_SIZE=32
export V31_DECODED_BATCH_LIMIT=8
```

`decoded_batch_limit` 只控制每个 batch 中参与 SO(3)/FK/jerk 解码损失的样本数，
noise/x0 训练仍使用完整 batch。
