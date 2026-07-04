# V46.12 MotionRAG-Diff: External Classical-Music Semantic Encoder

本补丁适配 `histry/EDGE` 当前代码和 `EDGE/change/*.bvh` 形式的 Chang-E/Dunhuang BVH 数据。核心原则：**不假设 BVH 与音乐逐段强配对**，而是把 BVH 文件名与运动学特征构建为 source-aware 强语义动作事件库，再把古典音乐模型输出的 slot-level 语义标签作为音乐查询，进行 semantic OT 和 RAG 检索路由。

## 与 EDGE 当前代码的关系

当前仓库已有的音乐侧代码主要是：

- `tools/extract_music_emotion_features.py`：librosa/rms/onset/tempo/arousal/calmness 等 8D 代理特征；
- `tools/extract_music_event_stream.py`：在上述特征上构造 12D event stream 与 `calm_flow/accent/climax/build_up/release` 等规则标签；
- `tools/extract_v21_music_features.py`：12D compact frame-level feature，包括 energy/onset/beat/tempo/arousal/tension/calm/novelty/brightness/section/accent。

这些是可用的音乐特征抽取器，但不是训练好的古典音乐语义分类模型。V46.12 因此新增外部语义编码接口：如果你已有训练好的古典音乐模型，只需让它输出 JSON/NPZ sidecar，或通过命令模板生成 sidecar。

## 外部音乐语义标签体系

固定使用 7 类，与 RAG 动作语义完全对齐：

```text
calm_meditative
lyrical_flow
pose_hold
instrument_phrase
percussive_accent
turning_climax
footwork_flow
```

动作侧对应关系：

```text
revelation_meditation  -> calm_meditative
thirty_six_postures    -> pose_hold
lotus_steps            -> footwork_flow / lyrical_flow
pipa_behind_back       -> instrument_phrase
lei_gong_drum          -> percussive_accent
ribbon_flow            -> turning_climax / lyrical_flow
```

## 外部模型 JSON 输出格式

推荐让你的古典音乐模型对每首音乐输出：

```json
{
  "audio": "test_music_bank/dunhuangwu2.wav",
  "labels": ["calm_meditative", "lyrical_flow", "pose_hold", "instrument_phrase", "percussive_accent", "turning_climax", "footwork_flow"],
  "slots": [
    {
      "slot_id": 0,
      "start_sec": 0.0,
      "end_sec": 4.0,
      "top_label": "calm_meditative",
      "probs": {
        "calm_meditative": 0.62,
        "lyrical_flow": 0.18,
        "pose_hold": 0.12,
        "instrument_phrase": 0.03,
        "percussive_accent": 0.02,
        "turning_climax": 0.01,
        "footwork_flow": 0.02
      }
    }
  ]
}
```

文件名可为：

```text
dunhuangwu2.music_semantic.json
dunhuangwu2_semantic.json
dunhuangwu2.json
```

放到以下任意目录：

```text
music_semantics/
external_music_semantics/
output/music_semantics/
```

也支持 NPZ：

```text
slot_start: [N]
slot_end: [N]
slot_label: [N]
slot_probs: [N,7]
label_names: [7]
```

## 安装

```bash
cd /mnt/data/V46_12_EDGE_MotionRAG_Diff_EXTERNAL_MUSIC_SEMANTIC_PATCH
bash install_v46_motionrag_diff_patch.sh /home/disk/lsm/storage/EDGE
```

## 重新建库

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

RUN_ROOT="output/v46_12_change_bvh_external_music_semantic_db_$(date +%Y%m%d_%H%M%S)"
DB_DIR="$RUN_ROOT/db"
mkdir -p "$DB_DIR"
echo "$RUN_ROOT" > output/LATEST_V46_CHANGE_BVH_DB.txt

export V46_BVH_RESAMPLE_TO_CONFIG_FPS=1
export V46_SOURCE_GROUP_MODE=filename
export V46_FILENAME_SEMANTIC_ENABLE=1
export V46_CLASSIFICATION_SEMANTIC_ENABLE=1

python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  build-db \
  --motion_dirs change \
  --out_db "$DB_DIR"
```

## 使用外部古典音乐语义模型训练 V44

如果你的模型已经生成 sidecar：

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT="$(cat output/LATEST_V46_CHANGE_BVH_DB.txt)"
DB="$RUN_ROOT/db/events.npz"

export V46_DEVICE=cuda
export V46_UNPAIRED_AUDIO_ENABLE=1
export V46_UNPAIRED_DISABLE_MOTION_PROXY=1
export V46_EXTERNAL_MUSIC_SEMANTIC_ENABLE=1
export V46_EXTERNAL_MUSIC_SEMANTIC_REQUIRED=1
export V46_EXTERNAL_MUSIC_SEMANTIC_DIRS="music_semantics:external_music_semantics:output/music_semantics"

python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-contrastive \
  --db "$DB" \
  --unpaired_audio_dirs test_music_bank data/music custom_music proxy_music \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --out "$RUN_ROOT/v44_contrastive.pt"
```

如果你的模型可命令行调用，使用模板：

```bash
export V46_EXTERNAL_MUSIC_SEMANTIC_CMD='python /path/to/your_encoder.py --audio {audio} --out_json {out_json}'
```

V46.12 会为每首音乐自动调用该命令并缓存输出。

## 没有外部模型时的代理侧车文件

可以先用内置代理脚本生成 JSON，验证流程：

```bash
mkdir -p music_semantics
python tools/v46_classical_music_semantic_proxy.py \
  --audio test_music_bank/dunhuangwu2.wav \
  --out_json music_semantics/dunhuangwu2.music_semantic.json
```

注意：这个代理脚本不是训练好的模型，只是输出同一 JSON schema 的 fallback。

## 训练 V45 / V46

```bash
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-refiner \
  --db "$DB" \
  --out "$RUN_ROOT/v45_refiner.pt"

python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  train-diffusion \
  --db "$DB" \
  --out "$RUN_ROOT/v46_diffusion.pt"
```

## 生成

```bash
python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  generate \
  --audio test_music_bank/dunhuangwu2.wav \
  --music_semantic_dirs music_semantics external_music_semantics output/music_semantics \
  --db "$DB" \
  --contrastive "$RUN_ROOT/v44_contrastive.pt" \
  --refiner "$RUN_ROOT/v45_refiner.pt" \
  --diffusion "$RUN_ROOT/v46_diffusion.pt" \
  --out "$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.npy" \
  --json "$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.report.json" \
  --render_output "$RUN_ROOT/dunhuangwu2_v46_12_MotionRAG_Diff.mp4"
```

## 论文口径

建议写作：

> 当前 Chang-E BVH 不提供逐段同步音乐监督。本文引入外部古典音乐语义编码器，将目标音乐解析为 slot-level 语义标签和概率分布，并与 source-aware 动作事件库中的动作类别、节奏属性、身体焦点和音乐对齐标签进行弱监督语义匹配。基于该外部语义先验，系统通过 semantic OT 构造伪正样本，训练 V44 检索对齐模型，并结合 V45 残差边界修复、V46 条件扩散重绘与 V43 真 lower-body IK，形成可追溯的 MotionRAG-Diff 框架。
