# V21 Scalable Multi-Music Dunhuang ChoreoRAG

## 1. Why V21

The previous V20 route proved three useful points:

- dynamic rhythm events preserve more movement “breathing” than fixed 45-frame units;
- style-first filtering is necessary, otherwise high-activity units drift away from Dunhuang dance;
- permanent one-song-one-database exclusivity can force differences, but it does not scale to unseen or many songs.

V21 therefore uses one shared Dunhuang style-safe event database and performs music-conditioned routing at query time.

```text
Shared Dunhuang Event-RAG
        ↓
Phrase-level music event queries
        ↓
Style-first relevance
        + music-motion matching
        + event compatibility
        + boundary cost
        + family-level MMR
        + optional batch overlap penalty
        ↓
Variable-length phrase schedule
        ↓
Optional DPN + endpoint transition refiner
        ↓
150-frame Dunhuang motion
```

The batch overlap penalty is soft. It helps paper comparison batches avoid collapse, but new or single music inputs never require rebuilding the database.

---

## 2. Files

### Shared index and music features

- `tools/v21_common.py`
- `tools/build_v21_shared_event_index.py`
- `tools/extract_v21_music_features.py`
- `tools/export_v21_start_pose.py`

### Query-time multi-music scheduling

- `tools/schedule_v21_multi_music.py`
- `tools/evaluate_v21_multi_music.py`
- `scripts/run_v21_multi_music.sh`

### Optional learned music router

- `model/v21_music_router.py`
- `tools/build_v21_router_dataset.py`
- `train_v21_music_router.py`
- `scripts/run_v21_train_router.sh`

### Optional Dunhuang style ranker

- `model/v21_style_ranker.py`
- `tools/build_v21_style_ranker_dataset.py`
- `train_v21_style_ranker.py`
- `tools/rescore_v21_index_with_style_ranker.py`
- `scripts/run_v21_train_style_ranker.sh`
- `scripts/run_v21_rescore_index.sh`

### Optional transition models

- `model/v21_transition.py`
- `tools/build_v21_transition_dataset.py`
- `train_v21_transition.py`
- `scripts/run_v21_train_transition.sh`

### Optional EDGE event adapter

- `tools/build_v21_event_manifest.py`
- `v21_event_adapter_patch.py`
- `train_v21_event_adapter.py`
- `scripts/run_v21_event_adapter_train.sh`

---

## 3. Install

The installer does not call `conda activate`; it assumes the EDGE environment is already active.

```bash
cd /path/to/EDGE_V21_MultiMusic_patch
EDGE_ROOT=/home/disk/lsm/storage/EDGE bash install_v21.sh
```

Then:

```bash
cd /home/disk/lsm/storage/EDGE
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

---

## 4. Recommended execution order

### Stage A — Build one shared index

Use the prototype-style database that already passed the style gate:

```bash
export V21_INPUT_DB=data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_prototype.json
export V21_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index
export V21_MAX_EVENTS=7000
export V21_MIN_STYLE_PERCENTILE=10
bash scripts/run_v21_build_index.sh
```

Outputs:

```text
data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.json
data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index.npz
```

This is the only Event-RAG index required for any number of songs.

### Stage B — Use a manually accepted common start pose

```bash
python tools/export_v21_start_pose.py \
  --motion output/night_v17b_conservative_20260604_170625/refined/music_dunhuangwu2_s0_32_64_96_ew0.45_music_dunhuangwu2_s0_32_64_96_ew0.45_emotion_refined.npy \
  --out data/dunhuang_dynamic_event_rag_physical/v20_common_start_pose.npy
```

### Stage C — Run arbitrary numbers of songs without training

For the current three-song bank:

```bash
export V21_AUDIO_GLOB='test_music_bank/dunhuangwu*.wav'
export V21_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index
export V21_START_POSE=data/dunhuang_dynamic_event_rag_physical/v20_common_start_pose.npy
export V21_RENDER=1
bash scripts/run_v21_multi_music.sh
```

For 50 or 500 songs:

```bash
export V21_AUDIO_GLOB='/path/to/music_folder/*.wav'
bash scripts/run_v21_multi_music.sh
```

No database rebuild is required.

For online single-song inference, turn off cross-song comparison penalties:

```bash
export V21_AUDIO_GLOB='/path/to/one_song.wav'
export V21_BATCH_OVERLAP_WEIGHT=0
export V21_BATCH_FAMILY_OVERLAP_WEIGHT=0
bash scripts/run_v21_multi_music.sh
```

---

## 5. Default retrieval priorities

The scheduler score is intentionally style-first:

```text
1. Dunhuang style score
2. unit quality and physical safety
3. music-query relevance and event compatibility
4. entry/exit transition cost
5. within-video family/MMR diversity
6. optional cross-music soft overlap penalty
```

Recommended defaults:

```bash
export V21_STYLE_WEIGHT=1.35
export V21_QUALITY_WEIGHT=0.65
export V21_SAFETY_WEIGHT=0.35
export V21_MUSIC_WEIGHT=0.80
export V21_EVENT_WEIGHT=0.65
export V21_TRANSITION_WEIGHT=0.55
export V21_MMR_WEIGHT=0.38
export V21_BATCH_OVERLAP_WEIGHT=0.30
export V21_BATCH_FAMILY_OVERLAP_WEIGHT=0.20
export V21_BATCH_MMR_WEIGHT=0.18
```

Do not solve music diversity by sharply increasing `V21_MUSIC_WEIGHT` or by permanently deleting shared candidates. That can bring back non-Dunhuang movement.

---

## 6. Optional learned music router

First cache music features:

```bash
export V21_AUDIO_GLOB='test_music_bank/*.wav'
bash scripts/run_v21_extract_music.sh
```

Train:

```bash
export V21_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index
export V21_MUSIC_FEATURE_GLOB='data/v21_music_features/*_v21_music.npy'
export V21_ROUTER_EPOCHS=250
bash scripts/run_v21_train_router.sh
```

Use the checkpoint:

```bash
export V21_ROUTER_CKPT=output/v21_music_router_YYYYMMDD_HHMMSS/train/checkpoints/best.pt
bash scripts/run_v21_multi_music.sh
```

The router is a weakly supervised calibration layer. It is optional; the rule-based query-time scheduler is the formal baseline.

---

## 7. Optional learned Dunhuang style ranker

Train from manually accepted V17B examples and low-style event negatives:

```bash
export V21_STYLE_EVENT_DB=data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_balanced_strict.json
bash scripts/run_v21_train_style_ranker.sh
```

Then rescore the shared index:

```bash
export V21_STYLE_RANKER_CKPT=output/v21_style_ranker_YYYYMMDD_HHMMSS/train/checkpoints/best.pt
export V21_INDEX_PREFIX=data/dunhuang_dynamic_event_rag_physical/v21_shared_event_index
bash scripts/run_v21_rescore_index.sh
```

This is preferable to treating `style_tension` as a complete Dunhuang style metric.

---

## 8. Optional DPN and endpoint transition refiner

Build real adjacent-boundary reconstruction pairs and train both models:

```bash
export V21_TRANSITION_EVENT_DB=data/dunhuang_dynamic_event_rag_physical/index_dynamic_event_style_balanced_strict.json
export V21_TRANSITION_EPOCHS=600
bash scripts/run_v21_train_transition.sh
```

Use them:

```bash
export V21_TRANSITION_CKPT=output/v21_transition_YYYYMMDD_HHMMSS/train/checkpoints/best.pt
bash scripts/run_v21_multi_music.sh
```

Without this checkpoint, V21 uses a conservative rule-based duration and linear rough transition. With the checkpoint, DPN predicts the length and the local refiner reconstructs a small endpoint-conditioned transition.

---

## 9. Optional EDGE event adapter

This is the final stage, not the starting point. Only enable it after retrieval and transition results are visually correct.

Create a music-event manifest from existing weak pairs:

```bash
python tools/build_v21_event_manifest.py \
  --weak_pairs_csv data/proxy_weak_pairs/weak_pairs.csv \
  --out_dir data/v21_event_adapter_features \
  --out_manifest data/v21_event_adapter_manifest.json \
  --num_frames 150
```

Train the adapter branch while preserving the pretrained motion prior:

```bash
export EDGE_ENABLE_V21_EVENT_ADAPTER=1
export EDGE_V21_EVENT_MANIFEST=data/v21_event_adapter_manifest.json
export EDGE_V21_EVENT_DROP_PROB=0.15
export EDGE_V21_TRAIN_STAGE=adapter
export EDGE_V21_ADAPTER_TRAIN_DECODER=0

bash scripts/run_v21_event_adapter_train.sh \
  --checkpoint /path/to/base_checkpoint.pt \
  --enable_rag_summary_token \
  --rag_summary_dim 12 \
  [other existing EDGE train options]
```

The adapter uses the existing RAG summary branch instead of rewriting `model/model.py`.

---

## 10. Main environment switches

```text
V21_INPUT_DB
V21_INDEX_PREFIX
V21_AUDIO_GLOB
V21_START_POSE
V21_ROUTER_CKPT
V21_TRANSITION_CKPT
V21_RENDER
V21_PHRASE_COUNT
V21_BEAM_SIZE
V21_CANDIDATE_TOP_K
V21_REFINE_ROUNDS
V21_STYLE_WEIGHT
V21_QUALITY_WEIGHT
V21_SAFETY_WEIGHT
V21_MUSIC_WEIGHT
V21_EVENT_WEIGHT
V21_ACTIVITY_WEIGHT
V21_DURATION_WEIGHT
V21_TRANSITION_WEIGHT
V21_MMR_WEIGHT
V21_FAMILY_REPEAT_WEIGHT
V21_SOURCE_REPEAT_WEIGHT
V21_BATCH_OVERLAP_WEIGHT
V21_BATCH_FAMILY_OVERLAP_WEIGHT
V21_BATCH_MMR_WEIGHT
V21_TIME_WARP_WEIGHT
V21_MIN_TIME_WARP
V21_MAX_TIME_WARP
V21_HARD_FAMILY_UNIQUE
EDGE_ENABLE_V21_EVENT_ADAPTER
EDGE_V21_EVENT_MANIFEST
EDGE_V21_EVENT_DROP_PROB
EDGE_V21_TRAIN_STAGE
EDGE_V21_ADAPTER_TRAIN_DECODER
```

---

## 11. Formal experiment groups

```text
A. Fixed 45-frame Visual-First baseline
B. Dynamic Event-RAG + independent scheduling
C. V21 shared DB + within-video family MMR
D. V21 + batch soft overlap penalty
E. V21 + learned music router
F. V21 + learned style ranker
G. V21 + DPN / endpoint transition refiner
H. V21 + optional EDGE event adapter
```

The final method should not claim that every song owns a private action database. It should be described as a shared Dunhuang event knowledge base with query-time music-conditioned routing and soft family-level diversity.
