"""Create a generic paired MMR index JSON for AIST++.

This script pairs audio feature files and motion files by filename stem.
It is intentionally simple because different AIST++ preprocessing pipelines use
slightly different directories. Prepare .npy files first, then run this script.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def stem_key(path: Path):
    stem = path.stem
    for suffix in ["_audio", "_audio_feature", "_feature", "_motion", "_151", "_slice"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio_feature_dir", required=True)
    p.add_argument("--motion_dir", required=True)
    p.add_argument("--out_dir", default="data/mmr_aist")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def main():
    args = parse_args()
    audio_files = list(Path(args.audio_feature_dir).rglob("*.npy"))
    motion_files = list(Path(args.motion_dir).rglob("*.npy"))
    audio_map = {stem_key(p): p for p in audio_files}
    motion_map = {stem_key(p): p for p in motion_files}
    keys = sorted(set(audio_map) & set(motion_map))
    if not keys:
        raise RuntimeError("No paired clips found by filename stem. Check your dirs or rename files.")
    items = [{"audio_feature": str(audio_map[k]), "motion": str(motion_map[k]), "pair_id": k} for k in keys]
    random.seed(args.seed)
    random.shuffle(items)
    n_val = max(1, int(round(len(items) * args.val_ratio))) if len(items) > 10 else max(0, len(items) // 5)
    val = items[:n_val]
    train = items[n_val:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(train, open(out / "index_train.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(val, open(out / "index_val.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("paired clips:", len(items))
    print("train:", len(train), "val:", len(val))
    print("saved:", out)


if __name__ == "__main__":
    main()
