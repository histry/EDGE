# EDGE In-wild Video-Aligned ChoreoRAG Patch

Files:

- `inwild_video_alignment_utils.py`
- `scripts/build_inwild_video_aligned_rag_db.py`
- `scripts/merge_choreo_rag_dbs.py`
- `functional_dual_context_selector.py`

Install:

```bash
cd /home/disk/lsm/storage/EDGE
cp /path/to/inwild_video_alignment_utils.py .
cp /path/to/scripts/build_inwild_video_aligned_rag_db.py scripts/
cp /path/to/scripts/merge_choreo_rag_dbs.py scripts/
cp /path/to/functional_dual_context_selector.py .
chmod +x scripts/build_inwild_video_aligned_rag_db.py scripts/merge_choreo_rag_dbs.py
python -m py_compile inwild_video_alignment_utils.py scripts/build_inwild_video_aligned_rag_db.py scripts/merge_choreo_rag_dbs.py functional_dual_context_selector.py
```

Manifest example:

```csv
source_id,title,motion_path,audio_path,video_path,rights_tag,fps
silkroad_001,Silk Road clip,output/inwild/silkroad_001_motion151.npy,output/inwild/silkroad_001.wav,/data/videos/silkroad_001.mp4,owned_or_permitted,30
```

Build video RAG DB:

```bash
python scripts/build_inwild_video_aligned_rag_db.py \
  --manifest data/inwild_dunhuang_video/manifest.csv \
  --out data/dunhuang_choreo_unit_rag/index_inwild_video_u45_s15_sync.npz \
  --unit_len 45 --stride 15 \
  --smooth_radius 1 --root_smooth_radius 1 \
  --freeze_stationary_root
```

Merge with existing curated DB:

```bash
python scripts/merge_choreo_rag_dbs.py \
  --inputs data/dunhuang_choreo_unit_rag/index_u45_s15_e10_expr_loco_mobility.npz \
           data/dunhuang_choreo_unit_rag/index_inwild_video_u45_s15_sync.npz \
  --out data/dunhuang_choreo_unit_rag/index_merged_curated_plus_inwild_sync.npz
```

Enable in selector:

```bash
export EDGE_ENABLE_INWILD_VIDEO_RAG=1
export EDGE_VIDEO_SYNC_CONTEXT_WEIGHT=0.35
export EDGE_VIDEO_SYNC_EVENT_GAIN=0.50
```

Rights note: only use videos you own, have permission to use, or have a license for research use. Do not redistribute raw videos.
