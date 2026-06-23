# EDGE V33 Event-Level Contact-INR 替换与运行指南

## 1. 不再使用旧窗口级数据库

以下文件必须视为无效实验数据：

```text
v32_transition_dataset_contact_overdense_INVALID.npz
v32_transition_dataset_contact_v2.npz
任何在 28,730 个窗口上独立生成 contact 的 NPZ
```

V33 必须从 4,225 个完整事件重新开始。

## 2. 安装

```bash
cd /home/disk/lsm/storage
unzip -o EDGE_V33_EVENT_CONTACT_PATCH.zip

python EDGE_V33_EVENT_CONTACT_PATCH/install_v33_patch.py \
  --edge_root /home/disk/lsm/storage/EDGE
```

安装器自动备份至：

```text
EDGE/backup_v33_时间戳/
```

## 3. 先单独构建事件级 contact cache

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge
export PYTHONPATH=$PWD:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

python tools/build_v33_event_contact_cache.py \
  --index_json data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json \
  --duration_index_npz data/v26_music_dominant_duration_index.npz \
  --out_npz data/v33_event_contact_cache.npz \
  --out_json data/v33_event_contact_cache.json \
  --device cuda:0 \
  --batch_size 96 \
  --fps 30 \
  --target_rates 0.42,0.42,0.38,0.38 \
  --transition_penalty 1.40 \
  --min_run 2 \
  --max_gap 2 \
  --existing_contact_policy auto
```

检查：

```bash
python -m json.tool data/v33_event_contact_cache.json
```

重点要求：

```text
valid_events 接近 4225
unresolved_events = 0
summary.overall_rate = 0.05～0.75
summary.all_four_rate < 0.55
每个 channel 不应全零或接近 1
```

## 4. 重建同步窗口数据库

```bash
python tools/build_v27_transition_diffusion_dataset.py \
  --index_json data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json \
  --duration_index_npz data/v26_music_dominant_duration_index.npz \
  --event_contact_cache data/v33_event_contact_cache.npz \
  --out_npz data/v33_transition_dataset.npz \
  --require_event_contacts 1 \
  --assert_contact_consistency 1 \
  --max_len 120 \
  --min_len 8 \
  --samples_per_event 6 \
  --source_pairs_per_event 0.75 \
  --allow_synthetic_adjacent 1 \
  --pseudo_pairs_per_event 0.05 \
  --seed 20260610
```

## 5. 严格审计

```bash
python tools/audit_v32_contact_dataset.py \
  --data data/v33_transition_dataset.npz \
  --out_json data/v33_transition_dataset.audit.json \
  --require_real_samples 1000 \
  --min_contact_rate 0.05 \
  --max_contact_rate 0.75 \
  --min_channel_rate 0.02 \
  --max_channel_rate 0.80 \
  --max_all_four_rate 0.55 \
  --min_mean_switch_rate 0.001 \
  --require_overlap_comparisons 1000 \
  --require_event_level_pipeline 1
```

必须看到：

```text
overlap_consistency.conflicts = 0
overlap_consistency.overlap_comparisons > 1000
contact_pipeline.level = complete_event_before_window_sampling
window_level_relabeling = false
```

## 6. 一键整夜训练

```bash
cd /home/disk/lsm/storage/EDGE
conda activate edge

tmux kill-session -t v33_event_contact 2>/dev/null || true

tmux new-session -d -s v33_event_contact \
  "cd /home/disk/lsm/storage/EDGE && \
   bash scripts/launch_v33_event_contact_overnight.sh"
```

实时查看：

```bash
RUN_ROOT=$(cat output/LATEST_V33_OVERNIGHT_LAUNCH.txt)
tail -F "$RUN_ROOT/run.log"
```

进入 tmux：

```bash
tmux attach -t v33_event_contact
```

退出但不中止：`Ctrl+B`，松开后按 `D`。

## 7. 主要环境开关

事件 contact：

```bash
export V33_CONTACT_TARGET_RATES=0.42,0.42,0.38,0.38
export V33_CONTACT_TRANSITION_PENALTY=1.40
export V33_CONTACT_MIN_RUN=2
export V33_CONTACT_MAX_GAP=2
export V33_CONTACT_TEMPERATURE=0.20
```

数据安全门：

```bash
export V33_MIN_CONTACT_RATE=0.05
export V33_MAX_CONTACT_RATE=0.75
export V33_MIN_CHANNEL_RATE=0.02
export V33_MAX_CHANNEL_RATE=0.80
export V33_MAX_ALL_FOUR_RATE=0.55
export V33_REQUIRE_OVERLAP_COMPARISONS=1000
```

保守伪接触损失：

```bash
export V32_W_CONTACT_BCE=0.20
export V32_W_CONTACT_SKATE=0.60
export V32_W_CONTACT_HEIGHT=0.25
export V32_W_FOOT_PENETRATION=0.50
export V32_W_SWING_CLEARANCE=0.06
export V32_W_CONTACT_TEMPORAL=0.04
export V32_W_CONTACT_BINARY=0.01
```

## 8. 复用 event cache

第一次构建成功后可复用：

```bash
export V33_BUILD_EVENT_CONTACT_CACHE=0
export V33_EVENT_CONTACT_CACHE=data/v33_event_contact_cache.npz
```

不要在改变事件动作文件后继续复用旧 cache。motion fingerprint 不一致时构建器会
直接失败。

## 9. Strict 模式

找到完整源序列后：

```bash
export V32_SUPERVISION_MODE=strict
export V32_SOURCE_MANIFEST=data/v32_source_manifest.json
export V32_FULL_MOTION_ROOT=/path/to/full_motion
export V32_REQUIRE_REAL_BOUNDARY_SAMPLES=1000
export V32_REQUIRE_UNIQUE_BOUNDARIES=250
export V32_REQUIRE_REAL_BOUNDARY_RATIO=0.10
```

完整 source sequence 的 contact 也会按同一个全局 calibration 一次性重建，所有
source-boundary masks 同步切片同一标签数组。
