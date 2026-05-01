"""Build weak Chinese-music <-> Dunhuang-motion pairs by rhythm similarity.

This does NOT create strong ground-truth pairs. Use the resulting index only for
low-weight weak fine-tuning or analysis. Strong MMR should be pretrained on AIST++.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mmr_data_utils import (
    iter_motion_files,
    load_audio_feature,
    load_motion_151,
    resample_sequence,
    rhythm_summary_from_audio,
    rhythm_summary_from_motion,
)


def distance(a, b):
    keys = ["onset_mean", "onset_std", "energy_mean", "energy_std"]
    # Map motion stats into comparable rhythm keys.
    bm = {
        "onset_mean": b.get("motion_energy_mean", 0.0),
        "onset_std": b.get("motion_energy_std", 0.0),
        "energy_mean": b.get("root_speed_mean", 0.0),
        "energy_std": b.get("motion_energy_std", 0.0),
    }
    return float(sum((a.get(k, 0.0) - bm.get(k, 0.0)) ** 2 for k in keys) ** 0.5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chinese_music_feature_dir", required=True)
    p.add_argument("--dunhuang_motion_dir", required=True)
    p.add_argument("--out", default="data/mmr_weak/dunhuang_weak_pairs.json")
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    audio_items = []
    for afile in sorted(Path(args.chinese_music_feature_dir).rglob("*.npy")):
        try:
            audio = resample_sequence(load_audio_feature(afile), args.seq_len)
            audio_items.append({"path": str(afile), "summary": rhythm_summary_from_audio(audio)})
        except Exception as exc:
            print("skip audio", afile, exc)
    motion_items = []
    for mfile in tqdm(list(iter_motion_files(args.dunhuang_motion_dir)), desc="motion stats"):
        try:
            motion = resample_sequence(load_motion_151(mfile), args.seq_len)
            motion_items.append({"path": str(mfile), "summary": rhythm_summary_from_motion(motion)})
        except Exception as exc:
            print("skip motion", mfile, exc)
    pairs = []
    for a in audio_items:
        scored = sorted([(distance(a["summary"], m["summary"]), m) for m in motion_items], key=lambda x: x[0])[: args.topk]
        for rank, (score, m) in enumerate(scored, start=1):
            pairs.append({
                "audio_feature": a["path"],
                "motion": m["path"],
                "pair_type": "weak_rhythm_match",
                "weak_weight": 0.2,
                "rank": rank,
                "rhythm_distance": score,
            })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pairs, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", args.out, "pairs:", len(pairs))


if __name__ == "__main__":
    main()
