# V46.34 Pretrained Music-Router Slot MotionRAG-Diff

## 1. 版本定位

V46.34 的目标是把已经训练好的 **V21 music router / V26 whole-song planner** 接回当前 V46.33 Reference-Conditioned Transition-Masked MotionRAG-Diff 主线。

核心变化：

```text
V21 music router + V26 planner
    → unseen whole-song music slot plan
    → V46 --slots_json
    → source-aware Event-RAG retrieval
    → transition-budgeted motion_ref
    → transition-mask refiner / diffusion
    → lower-body IK / contact physics
```

这解决的是：当前 V46.33 默认使用 `audio_slots()` 或 `music_semantics/*.json` 构造 slot，没有显式加载已训练的 V21/V26 音乐结构权重。V46.34 使没听过的整曲音乐先经过训练式音乐结构层，再进入 MotionRAG-Diff。

---

## 2. 文件说明

复制到 EDGE 后使用：

```text
tools/v46_33_reference_transition_patch.py
    保留 V46.33 的 transition-budget inbetweening、motion_ref 和 transition_mask 逻辑。

tools/v46_33_relabel_change_event_db.py
    对 change 数据库事件进行敦煌/Chang-E 语义重标注。

tools/v46_34_pretrained_music_slot_plan.py
    调用 V21/V26 训练权重，为没听过的整曲音乐生成 V46-compatible slots_json。

tools/v46_34_router_slot_patch.py
    修改 tools/v46_motionrag_diff.py 的 audio_slots()，使 generate 强制读取 pretrained router slot plan。

scripts/run_v46_34_pretrained_router_reference_transition_overnight.sh
    一键重建 DB、生成 slot plan、训练 V44/V45/V46，并输出 motion_mg / refiner_ik / diffusion_ik。
```

---

## 3. 安装

在本机 EDGE 根目录执行：

```bash
cd /home/disk/lsm/storage/EDGE

cp /path/to/V46_34_pretrained_router_slot_solution/tools/*.py tools/
cp /path/to/V46_34_pretrained_router_slot_solution/scripts/*.sh scripts/
chmod +x tools/v46_34_*.py tools/v46_33_*.py scripts/run_v46_34_pretrained_router_reference_transition_overnight.sh
```

---

## 4. 必须确认的预训练权重

推荐默认：

```bash
export V26_ROUTER_CKPT="output/v21_music_router_985songs_20260605_154801/seed_20260607/checkpoints/best.pt"
export V26_PLANNER_CKPT="output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt"
export V27_HG_PLANNER_CKPT="output/v27_hg_overnight_20260609_222844/planner_seed_20260615/checkpoints/best.pt"
export V34_CONTACT_INR_CKPT="output/v34_dense_boundary_train_overnight_20260621_201033/v34_contact_inr_training/checkpoints/contact_finetuned_best.pt"
```

`tools/schedule_v26_whole_song.py` 还需要 V23 duration/time-warp checkpoint：

```bash
find . -type f \( -iname "*.pt" -o -iname "*.pth" -o -iname "*.ckpt" \) \
  | grep -Ei "v23|duration|monotonic|timewarp|warp"
```

找到后设置：

```bash
export V26_V23_CKPT="/path/to/v23_checkpoint.pt"
```

如果只想调试 slot 文件生成，可临时使用：

```bash
export V46_34_ALLOW_SEMANTIC_FALLBACK=1
```

正式实验不建议打开 fallback。

---

## 5. 正式运行

```bash
cd /home/disk/lsm/storage/EDGE

CUDA_VISIBLE_DEVICES=0 \
PYTHONUNBUFFERED=1 \
V46_DEVICE=cuda \
V46_REQUIRE_PRETRAINED_ROUTER_SLOTS=1 \
V46_34_ALLOW_SEMANTIC_FALLBACK=0 \
V46_TARGET_AUDIO=test_music_bank/dunhuangwu2.wav \
V46_CONTRASTIVE_EPOCHS=160 \
V46_REFINER_TRAIN_STEPS=10000 \
V46_DIFFUSION_TRAIN_STEPS=20000 \
V46_DIFFUSION_STEPS=50 \
V46_TRANSITION_BUDGET_ENABLE=1 \
V46_TRANSITION_INBETWEEN_ENABLE=1 \
V46_REFINER_CORE_STRENGTH=0.02 \
V46_REFINER_TRANSITION_STRENGTH=1.00 \
V46_DIFFUSION_CORE_STRENGTH=0.00 \
V46_DIFFUSION_TRANSITION_STRENGTH=0.72 \
bash scripts/run_v46_34_pretrained_router_reference_transition_overnight.sh
```

如果希望后台运行：

```bash
nohup bash scripts/run_v46_34_pretrained_router_reference_transition_overnight.sh \
  > /tmp/v46_34_router_overnight.log 2>&1 &
```

---

## 6. 输出文件

运行目录形如：

```text
output/v46_34_pretrained_router_reference_transition_YYYYMMDD_HHMMSS/
```

重点文件：

```text
*_v46_34_pretrained_router_slots.json
    V21/V26 训练式音乐 slot plan。必须检查 slot_source 是否为 v21_router_v26_planner。

train_all_db/events.npz
    使用 change 数据重建后的 all-train Event-RAG 数据库。

v44_contrastive_v46_34.pt
v45_refiner_v46_34.pt
v46_diffusion_v46_34.pt
    本轮训练权重。

dunhuangwu2_v46_34_router_motion_mg.mp4
    Stage-1 RAG / motion graph baseline。

dunhuangwu2_v46_34_router_refiner_ik.mp4
    Refiner + IK 结果。

dunhuangwu2_v46_34_router_diffusion_ik.mp4
    Reference-conditioned masked diffusion + IK 结果。

V46_34_FINAL_SUMMARY.json
    汇总。
```

---

## 7. 检查是否真正用了 pretrained router slots

```bash
RUN_ROOT=$(ls -td output/v46_34_pretrained_router_reference_transition_* | head -1)
cat "$RUN_ROOT"/*_v46_34_pretrained_router_slots.json | grep -E 'slot_source|router_ckpt|planner_ckpt|num_slots|total_target_frames'

grep -RIn "loaded pretrained router slot plan\|V46.34" "$RUN_ROOT/logs" | head -50
```

应看到：

```text
slot_source = v21_router_v26_planner
router_ckpt = output/v21_music_router_985songs_.../best.pt
planner_ckpt = output/v26_music_dominant_whole_song_planner_985/checkpoints/best.pt
[V46.34] loaded pretrained router slot plan
```

---

## 8. 论文实验建议

建议保留以下消融：

```text
A. V46.33 default music-semantic-slot baseline
B. V46.34 pretrained V21 router slot plan + RAG-only motion_mg
C. V46.34 pretrained router slots + transition-budget motion_ref + V45 refiner
D. V46.34 pretrained router slots + transition-mask diffusion + IK
```

这样能证明：训练式音乐结构层、reference trajectory、transition mask、diffusion 与 IK 的独立贡献。
