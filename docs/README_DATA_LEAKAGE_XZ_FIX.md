# Dunhuang Data Leakage + X/Z Trajectory Contract Fix

Files in this patch:

- `dataset/dance_dataset.py`: direct replacement.  It splits Dunhuang data by inferred original source video **before** overlapping windows are created, and fails if train/val sources overlap.  It also enforces the X/Z ground-plane trajectory contract.
- `tools/audit_dunhuang_split_xz_contract.py`: optional audit script.  It constructs train and val datasets and verifies source disjointness plus `cond["trajectory"] == motion[:, [4,6]]` in normalized space.
- `scripts/run_train_dunhuang_no_leakage.sh`: optional strict training launcher.

Recommended formal environment:

```bash
export EDGE_DUNHUANG_SPLIT_MODE=source_file
export EDGE_DUNHUANG_STRICT_SPLIT=1
export EDGE_DUNHUANG_ALLOW_SINGLE_SOURCE_SPLIT=0
export EDGE_DUNHUANG_ASSERT_TRAJ_MATCH=1
export EDGE_TRAJECTORY_PLANE=xz
export EDGE_DUNHUANG_SPLIT_REPORT_DIR=output/split_reports
```

Audit before large training:

```bash
python tools/audit_dunhuang_split_xz_contract.py \
  --data_path data/dunhuang_bvh/processed \
  --seq_len 150 \
  --split_ratio 0.9 \
  --split_seed 42
```

For hand-curated splits, set:

```bash
export EDGE_DUNHUANG_SPLIT_MANIFEST=data/dunhuang_bvh/split_manifest.json
```

Manifest format:

```json
{
  "train": ["source_video_001", "source_video_002"],
  "val": ["source_video_003"]
}
```

Do not set `EDGE_DUNHUANG_SPLIT_MODE=all` except for local smoke tests; strict mode rejects it because it can leak original source videos across train and validation.
