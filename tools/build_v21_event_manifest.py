#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a window_id -> V21 music-event feature manifest for optional EDGE adapter training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

from tools.extract_v21_music_features import extract_audio_features


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak_pairs_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--num_frames", type=int, default=150)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    audio_cache: Dict[str, str] = {}

    with open(args.weak_pairs_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            window_id = str(row.get("window_id", "")).strip()
            audio_path = str(row.get("audio_path", "")).strip()
            if not window_id or not audio_path:
                continue
            audio = Path(audio_path)
            candidates = [audio, Path("test_music_bank") / audio.name, Path("proxy_music") / audio.name]
            audio = next((p for p in candidates if p.is_file()), audio)
            if not audio.is_file():
                continue
            key = str(audio.resolve())
            if key not in audio_cache:
                feat, _ = extract_audio_features(audio, num_frames=args.num_frames)
                out = out_dir / f"{audio.stem}_v21_event.npy"
                np.save(out, feat.astype(np.float32))
                audio_cache[key] = str(out)
            manifest[window_id] = audio_cache[key]

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(
            {
                "version": "v21_event_adapter_manifest",
                "feature_dim": 12,
                "num_frames": args.num_frames,
                "num_windows": len(manifest),
                "windows": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved:", out_manifest)
    print("windows:", len(manifest))
    print("unique_audio:", len(audio_cache))


if __name__ == "__main__":
    main()
