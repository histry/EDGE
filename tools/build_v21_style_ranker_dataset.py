#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build pairwise style data from manually accepted motions and low-style events."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List

import numpy as np

from tools.v21_common import load_json_items, load_motion, motion_mmr_embedding, resample_motion


def windows(x: np.ndarray, length: int = 48, stride: int = 24):
    if len(x) <= length:
        return [resample_motion(x, length)]
    out = [x[s : s + length] for s in range(0, len(x) - length + 1, stride)]
    if not np.array_equal(out[-1], x[-length:]):
        out.append(x[-length:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive_glob", action="append", required=True)
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--negative_glob", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_pairs", type=int, default=20000)
    ap.add_argument("--low_style_percentile", type=float, default=35.0)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    positive_files: List[Path] = []
    for pattern in args.positive_glob:
        positive_files.extend(Path(x) for x in glob.glob(pattern, recursive=True))
    positive_files = sorted(set(positive_files))
    if not positive_files:
        raise RuntimeError("No positive motions matched")

    pos_embed = []
    for path in positive_files:
        x = load_motion(path)
        for w in windows(x):
            pos_embed.append(motion_mmr_embedding(w))
        print("[POS]", path)

    _, items = load_json_items(args.event_db)
    style_values = np.asarray(
        [
            float(
                item.get(
                    "dunhuang_style_score_v20f3",
                    item.get("dunhuang_style_proxy", item.get("visual_score", 0.0)),
                )
            )
            for item in items
        ],
        dtype=np.float32,
    )
    threshold = float(np.percentile(style_values, args.low_style_percentile))
    negative_files = []
    for item, score in zip(items, style_values):
        if score <= threshold:
            p = Path(str(item.get("pkl", item.get("path", ""))))
            if p.is_file():
                negative_files.append(p)
    for pattern in args.negative_glob:
        negative_files.extend(Path(x) for x in glob.glob(pattern, recursive=True))
    negative_files = sorted(set(negative_files))
    if not negative_files:
        raise RuntimeError("No negative motions available")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(negative_files)
    neg_embed = []
    for path in negative_files[: min(len(negative_files), 5000)]:
        try:
            neg_embed.append(motion_mmr_embedding(load_motion(path)))
        except Exception:
            continue
    pos = np.stack(pos_embed).astype(np.float32)
    neg = np.stack(neg_embed).astype(np.float32)
    pidx = rng.integers(0, len(pos), size=args.num_pairs)
    nidx = rng.integers(0, len(neg), size=args.num_pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, positive=pos[pidx], negative=neg[nidx])
    print("saved:", out)
    print("positive_windows:", len(pos))
    print("negative_events:", len(neg))
    print("pairs:", args.num_pairs)


if __name__ == "__main__":
    main()
