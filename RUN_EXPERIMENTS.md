# ChoreoRAG 实验执行说明

## 0. 放置文件

把本代码包中的文件复制到 EDGE 仓库根目录：

```bash
cp auto_keyframe_planner.py /home/disk/lsm/storage/EDGE/auto_keyframe_planner.py
cp build_choreo_unit_rag_db.py /home/disk/lsm/storage/EDGE/build_choreo_unit_rag_db.py
cp music_choreo_planner.py /home/disk/lsm/storage/EDGE/music_choreo_planner.py
cp planner_edit_loop.py /home/disk/lsm/storage/EDGE/planner_edit_loop.py
mkdir -p /home/disk/lsm/storage/EDGE/model
cp model/text_bridge_encoder.py /home/disk/lsm/storage/EDGE/model/text_bridge_encoder.py
```

可选安装文本模型依赖：

```bash
pip install sentence-transformers
```

不安装也能跑，会回退到 hash embedding；正式实验建议安装。

---

## 1. 构建 motion-unit RAG DB

推荐 45 帧 unit、15 帧 stride：

```bash
cd /home/disk/lsm/storage/EDGE

python build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out data/dunhuang_choreo_unit_rag/index_u45_s15.npz \
  --checkpoint runs/train/exp17_dunhuang_stage1_from_aist300/weights/train-30.pt \
  --pose_space normalized \
  --unit_len 45 \
  --stride 15 \
  --text_model BAAI/bge-small-zh-v1.5 \
  --text_device cuda
```

如果显存/网络环境不方便：

```bash
--text_model hash --text_device cpu
```

第二个库用于消融：

```bash
python build_choreo_unit_rag_db.py \
  --input_dir data/dunhuang_bvh/processed \
  --out data/dunhuang_choreo_unit_rag/index_u60_s20.npz \
  --checkpoint runs/train/exp17_dunhuang_stage1_from_aist300/weights/train-30.pt \
  --pose_space normalized \
  --unit_len 60 \
  --stride 20
```

---

## 2. 生成 choreography plan

如果你能把音频特征保存为 .npy，先生成 plan：

```bash
python music_choreo_planner.py \
  --audio_feature output/check_mmr_rag/dyl002_audio_feat.npy \
  --num_frames 150 \
  --out output/choreo_plan/dyl002_plan.json \
  --style_hint "敦煌舞，飞天感，上肢舒展，重心稳定，含蓄优雅"
```

如果暂时没有音频特征文件，也可以不生成 plan；新 planner 会自动 heuristic fallback。

如果使用人工/LLM 写的 plan，保存成同 schema JSON，并在推理前：

```bash
export EDGE_CHOREO_PLAN_JSON=output/choreo_plan/dyl002_plan.json
```

---

## 3. 不改 generate_controlled.py 的推理方式

替换 `auto_keyframe_planner.py` 后，旧的 `generate_controlled.py` 会自动调用新 planner。

设置 ChoreoRAG 环境变量：

```bash
export EDGE_MOTION_UNIT_MODE=on
export EDGE_CHOREO_PLAN_JSON=output/choreo_plan/dyl002_plan.json
export EDGE_TEXT_BRIDGE_MODEL=BAAI/bge-small-zh-v1.5
export EDGE_TEXT_BRIDGE_DEVICE=cuda
export EDGE_TEXT_BRIDGE_WEIGHT=0.50
export EDGE_UNIT_ENTRY_WEIGHT=0.60
export EDGE_UNIT_EXIT_WEIGHT=0.60
export EDGE_UNIT_CONTACT_PHASE_WEIGHT=0.85
```

auto1 主实验：

```bash
python generate_controlled.py \
  --checkpoint runs/train/exp17_dunhuang_stage1_from_aist300/weights/train-30.pt \
  --music test_music/dyl002.wav \
  --feature_type hybrid \
  --audio_dim 803 \
  --start_pose assets/start.npy \
  --end_pose assets/end.npy \
  --trajectory "0,0;0.5,0.7;-0.3,1.2;0,1.6" \
  --auto_mid_keyframes \
  --auto_mid_count 1 \
  --rag_db data/dunhuang_choreo_unit_rag/index_u45_s15.npz \
  --auto_mid_pose_space normalized \
  --mid_keyframe_strength 0.20 \
  --infer_keyframe_width 0 \
  --no_tto \
  --out output/choreorag/dyl002_auto1.npy
```

auto2 用于长序列或 240 帧以上：

```bash
python generate_controlled.py \
  ... \
  --auto_mid_count 2 \
  --mid_keyframe_strength 0.15 \
  --out output/choreorag/dyl002_auto2.npy
```

---

## 4. 生成后诊断与轻量 edit 建议

推理后会生成类似：

```text
output/choreorag/dyl002_auto1_choreorag_plan.json
```

运行诊断：

```bash
python planner_edit_loop.py \
  --motion output/choreorag/dyl002_auto1.npy \
  --plan_json output/choreorag/dyl002_auto1_choreorag_plan.json \
  --target_traj output/choreorag/dyl002_auto1_target_traj.npy \
  --out output/choreorag/dyl002_auto1_edit_diag.json
```

若 `transition_jerk` 高：

```bash
export EDGE_UNIT_ENTRY_WEIGHT=0.85
export EDGE_UNIT_EXIT_WEIGHT=0.85
# 同时把 --mid_keyframe_strength 改成 0.15
```

若 `contact_phase_break` 高：

```bash
export EDGE_UNIT_CONTACT_PHASE_WEIGHT=1.20
```

然后重新跑同一条 `generate_controlled.py` 命令，输出命名为 `*_edit1.npy`。

---

## 5. 推荐消融矩阵

### A. 当前旧路线对照

```text
old_frame_rag_auto1
```

使用旧 RAG DB 或设置：

```bash
export EDGE_MOTION_UNIT_MODE=off
export EDGE_TEXT_BRIDGE_WEIGHT=0.0
```

### B. motion-unit RAG，不使用文本

```bash
export EDGE_MOTION_UNIT_MODE=on
export EDGE_TEXT_BRIDGE_WEIGHT=0.0
```

### C. text + unit RAG

```bash
export EDGE_MOTION_UNIT_MODE=on
export EDGE_TEXT_BRIDGE_WEIGHT=0.50
```

### D. choreo plan + text + unit RAG

```bash
export EDGE_CHOREO_PLAN_JSON=output/choreo_plan/dyl002_plan.json
export EDGE_TEXT_BRIDGE_WEIGHT=0.50
```

### E. choreo plan + edit

先跑 D，再跑 `planner_edit_loop.py`，按建议调权重后重跑。

---

## 6. 建议报告指标

保留原指标：

```text
trajectory_ade_m
foot_slide_rate
foot_contact_speed_p95_mps
keyframe_mpjpe_m_mean
keyframe_rot_err_deg_mean
beatalign_symmetric
```

新增：

```text
segment_semantic_alignment = 1 - text_cost
entry_compat_cost
exit_compat_cost
contact_phase_cost
transition_jerk@auto_mid
contact_phase_break@auto_mid
freezing_score
```

这些都已保存在 `*_choreorag_plan.json` 或可由 `planner_edit_loop.py` 输出。

---

## 7. 推荐主表

| setting | BeatAlign ↑ | Traj ADE ↓ | Foot p95 ↓ | Keyframe MPJPE ↓ | Transition Jerk ↓ | Contact Break ↓ | Freezing ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_frame_rag_auto1 | | | | | | | |
| unit_rag_auto1 | | | | | | | |
| text_unit_rag_auto1 | | | | | | | |
| choreo_unit_rag_auto1 | | | | | | | |
| choreo_unit_rag_auto1_edit | | | | | | | |

结论目标：
- `unit_rag_auto1` 应降低突变；
- `text_unit_rag_auto1` 应提高语义匹配；
- `choreo_unit_rag_auto1` 应让不同音乐段落选到不同动作单元；
- `edit` 应进一步降低 transition jerk / contact phase break。
