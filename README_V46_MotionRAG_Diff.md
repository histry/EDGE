# V46.11 MotionRAG-Diff Strong Semantic RAG Patch

本补丁面向当前 `histry/EDGE` 的 V42.2/V46 系列项目状态，适配 `EDGE/change/*.bvh` 中的 Chang-E Dunhuang BVH 数据，并将 RAG 动作语义标签与未配对音乐 slot 对齐标签升级为强分类语义版本。

## 核心定位

当前 `change/` 数据是 motion-only BVH，不假设存在逐帧同步的音乐-动作配对。因此 V46.11 不把 V44 写成强监督 paired music-motion learning，而是：

```text
Chang-E BVH filename / motion statistics -> source-aware action event semantics
unpaired target music -> slot-level music alignment semantics
semantic OT + contrastive grounding -> weak cross-modal retrieval prior
Beam/Viterbi routing -> source-aware schedule_report
V43 IK + V45 Refiner + V46 residual diffusion -> boundary repair and regeneration
```

## 相比 V46.9 的新增内容

1. **完整 BVH 文件名作为 source_group**：`female_lotus.bvh -> female_lotus`，12 个 BVH 应得到 12 个 source group。
2. **动作强分类语义**：为每个 event 写入 `energy_label / rhythm_label / body_focus_label / spatial_label / music_alignment_label / classification_text`。
3. **音乐 slot 对齐标签**：音频 slot 自动推断 `calm_meditative / lyrical_flow / pose_hold / instrument_phrase / percussive_accent / turning_climax / footwork_flow`。
4. **分类语义向量 `class_semantic[32]`**：与 `name_semantic[32]` 和 `desc_z[32]` 融合，用于 V44 semantic OT、检索排序和报告。
5. **RAG 检索分类加权**：`retrieve_schedule()` 对符合 slot 偏好的舞蹈类别、语义角色和音乐对齐标签给予可控加权。
6. **schedule_report 增强**：每个 slot 输出 top candidate 的 `label / dance_key / music_alignment_label / class_bonus / candidate_preview`。


## V46.11 canonical 修正

本版专门修复 `female_mediation.bvh / male_mediation.bvh` 这类本地文件名拼写变体：

```text
source_group / source_bvh 保留原始 stem，例如 female_mediation，用于 source-aware 审计；
内部 RAG 语义统一 canonicalize 为 revelation_meditation；
dance_category 统一为 Revelation Meditation；
music_alignment_label 统一为 calm_meditative；
classification_text / semantic_text 不再出现 mediation。
```

因此，重新建库后可以继续保留原始 BVH 文件名，不需要手动重命名文件。

## 安装

```bash
cd /mnt/data/V46_11_EDGE_MotionRAG_Diff_CANONICAL_SEMANTIC_PATCH
bash install_v46_motionrag_diff_patch.sh /home/disk/lsm/storage/EDGE
```

## 重建 change BVH 数据库

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}

RUN_ROOT="output/v46_11_change_bvh_strong_semantic_db_$(date +%Y%m%d_%H%M%S)"
DB_DIR="$RUN_ROOT/db"
mkdir -p "$DB_DIR"
echo "$RUN_ROOT" > output/LATEST_V46_CHANGE_BVH_DB.txt

export V46_BVH_RESAMPLE_TO_CONFIG_FPS=1
export V46_SOURCE_GROUP_MODE=filename
export V46_FILENAME_SEMANTIC_ENABLE=1
export V46_FILENAME_SEMANTIC_WEIGHT=0.35
export V46_CLASSIFICATION_SEMANTIC_ENABLE=1
export V46_CLASSIFICATION_SEMANTIC_RATIO=0.70
export V46_CLASSIFICATION_RETRIEVAL_WEIGHT=0.34
export V46_CLASSIFICATION_OT_WEIGHT=0.45
export V46_CLASSIFICATION_RETRIEVAL_BONUS=0.28
export V46_DEVICE=cuda

python tools/v46_motionrag_diff.py \
  --config configs/v46_motionrag_diff_config.json \
  build-db \
  --motion_dirs change \
  --out_db "$DB_DIR"
```

## 检查建库结果

```bash
python - <<'PY'
import json, numpy as np, os
from collections import Counter
run_root = open('output/LATEST_V46_CHANGE_BVH_DB.txt').read().strip()
db = np.load(os.path.join(run_root, 'db/events.npz'), allow_pickle=True)
meta = json.load(open(os.path.join(run_root, 'db/events_meta.json'), 'r', encoding='utf-8'))
print('RUN_ROOT:', run_root)
print('num_events:', meta.get('num_events'))
print('source groups:', len(set(db['source_groups'].tolist())))
print('class_semantic shape:', db['class_semantic'].shape)
print('dance_key counts:', Counter(db['dance_keys'].tolist()))
assert 'mediation' not in set(map(str, db['dance_keys'].tolist())), 'internal dance_key still contains mediation'
assert 'revelation_meditation' in set(map(str, db['dance_keys'].tolist())), 'revelation_meditation missing after canonicalization'
print('energy labels:', Counter(db['energy_labels'].tolist()))
print('rhythm labels:', Counter(db['rhythm_labels'].tolist()))
print('music alignment:', Counter(db['music_alignment_labels'].tolist()))
for s in sorted(set(db['source_groups'].tolist())):
    print(' ', s)
print('first classification:', meta['events'][0].get('classification_text'))
PY
```

预期 source groups 为 12：

```text
female_36pose_1, female_36pose_2, female_lotus, female_mediation 或 female_meditation,
male_36pose_1, male_36pose_2, male_drum_1, male_drum_2,
male_mediation 或 male_meditation, male_pipa_1, male_pipa_2, male_ribbon
```

## 训练与生成

```bash
cd /home/disk/lsm/storage/EDGE
RUN_ROOT="$(cat output/LATEST_V46_CHANGE_BVH_DB.txt)"
DB="$RUN_ROOT/db/events.npz"

python tools/v46_motionrag_diff.py --config configs/v46_motionrag_diff_config.json \
  train-contrastive --db "$DB" \
  --unpaired_audio_dirs test_music_bank data/music custom_music proxy_music \
  --out "$RUN_ROOT/v44_contrastive.pt"

python tools/v46_motionrag_diff.py --config configs/v46_motionrag_diff_config.json \
  train-refiner --db "$DB" --out "$RUN_ROOT/v45_refiner.pt"

python tools/v46_motionrag_diff.py --config configs/v46_motionrag_diff_config.json \
  train-diffusion --db "$DB" --out "$RUN_ROOT/v46_diffusion.pt"
```

## 关键环境开关

```bash
export V46_CLASSIFICATION_SEMANTIC_ENABLE=1
export V46_CLASSIFICATION_SEMANTIC_RATIO=0.70
export V46_CLASSIFICATION_RETRIEVAL_WEIGHT=0.34
export V46_CLASSIFICATION_OT_WEIGHT=0.45
export V46_CLASSIFICATION_RETRIEVAL_BONUS=0.28
export V46_CLASSIFICATION_REPORT_TOPK=8
```

