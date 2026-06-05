#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a simple pairwise ranking dataset from Dynamic Event-RAG items.
A > B if rule quality/visual/safety scores are higher.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from tools.v20_motion_utils import load_json, write_json


def unit_score(item):
    d = item.get("descriptor", {})
    return float(0.35 * item.get("quality_score", 0.0) + 0.25 * item.get("visual_score", 0.0) + 0.20 * item.get("safety_score", 0.0) + 0.20 * d.get("style_tension", 0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    db = load_json(args.event_db); items = db.get("items", [])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    if len(items) < 2:
        raise RuntimeError("Need at least two event units")
    for i in range(args.num_pairs):
        a, b = rng.sample(items, 2)
        sa, sb = unit_score(a), unit_score(b)
        if abs(sa - sb) < 1e-6:
            continue
        win, lose = (a, b) if sa > sb else (b, a)
        pairs.append({"winner": win["event_id"], "loser": lose["event_id"], "winner_pkl": win["pkl"], "loser_pkl": lose["pkl"], "score_diff": abs(sa - sb)})
    write_json({"num_pairs": len(pairs), "pairs": pairs}, out_dir / "pairwise_rank_dataset.json")
    print(f"saved: {out_dir / 'pairwise_rank_dataset.json'} pairs={len(pairs)}")

if __name__ == "__main__":
    main()
