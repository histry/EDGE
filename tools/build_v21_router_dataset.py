#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build weak contrastive pairs for the optional V21 music-motion router.

The dataset is intentionally weakly supervised: positive motion events are selected
by style-safe semantic compatibility and activity matching, while negatives are
hard events with similar quality but incompatible phrase semantics. It is an
optional calibration layer; the default V21 scheduler also works without it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from tools.v21_common import EVENT_TYPES, event_compatibility
from tools.v21_music_event_calibrated import build_phrase_query as calibrated_phrase_query


def load_index(json_path: Path, npz_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    arrays = np.load(npz_path)
    return meta, arrays


def phrase_query(features: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, str]:
    w = np.asarray(features[start:end], dtype=np.float32)
    if len(w) == 0:
        w = np.zeros((1, 12), dtype=np.float32)
    energy = float(w[:, 0].mean())
    arousal = float(w[:, 4].mean())
    delta = float(w[:, 5].mean())
    tension = float(w[:, 6].mean())
    calm = float(w[:, 7].mean())
    novelty = float(w[:, 8].mean())
    section = float(w[:, 10].max())
    accent = float(w[:, 11].mean())
    beat = float(w[:, 2].mean())
    if section > 0.70:
        event = "section_change"
    elif calm > 0.68 and tension < 0.50:
        event = "calm_flow"
    elif arousal > 0.72 and tension > 0.68:
        event = "climax"
    elif delta > 0.025 or (arousal > 0.62 and tension > 0.58):
        event = "build_up"
    elif delta < -0.025 and arousal < 0.58:
        event = "release"
    elif accent > 0.55 or beat > 0.28:
        event = "accent"
    else:
        event = "neutral_flow"

    desired = {
        "accent": (0.85, 0.65, 0.35),
        "climax": (0.95, 0.75, 0.45),
        "section_change": (0.65, 0.70, 0.75),
        "build_up": (0.75, 0.70, 0.40),
        "release": (0.45, 0.50, 0.35),
        "calm_flow": (0.40, 0.55, 0.25),
        "neutral_flow": (0.55, 0.55, 0.40),
    }[event]
    upper, torso, lower = desired
    q = np.asarray(
        [
            np.clip(arousal, 0, 1), upper, torso, lower,
            np.clip(tension, 0, 1), np.clip(calm, 0, 1),
            np.clip(section + 0.5 * beat, 0, 1),
            np.clip(max(delta, 0.0) * 8.0, 0, 1),
            np.clip(max(-delta, 0.0) * 8.0, 0, 1),
            np.clip(accent, 0, 1), np.clip(novelty, 0, 1),
            np.clip((end - start) / 60.0, 0, 1),
        ],
        dtype=np.float32,
    )
    return q, event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_json", required=True)
    ap.add_argument("--index_npz", required=True)
    ap.add_argument("--music_glob", required=True, help="Glob for *_v21_music.npy or equivalent [T,12] features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phrases", type=int, default=3)
    ap.add_argument("--positives_per_phrase", type=int, default=4)
    ap.add_argument("--negatives_per_positive", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    meta, arrays = load_index(Path(args.index_json), Path(args.index_npz))
    items = meta["items"]
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    music_files = sorted(Path().glob(args.music_glob)) if not Path(args.music_glob).is_absolute() else sorted(Path("/").glob(args.music_glob.lstrip("/")))
    if not music_files:
        # pathlib's glob does not support every absolute pattern consistently; use glob module fallback.
        import glob
        music_files = [Path(x) for x in sorted(glob.glob(args.music_glob))]
    if not music_files:
        raise RuntimeError(f"No music feature files matched {args.music_glob}")

    music_rows: List[np.ndarray] = []
    pos_rows: List[np.ndarray] = []
    neg_rows: List[np.ndarray] = []
    labels: List[str] = []

    event_names = [str(x.get("event_type", "neutral_flow")) for x in items]
    base = 0.55 * style + 0.25 * quality + 0.20 * safety

    for music_file in music_files:
        feat = np.load(music_file).astype(np.float32)
        if feat.ndim != 2 or feat.shape[1] < 12:
            print(f"[SKIP] {music_file}: expected [T,12], got {feat.shape}")
            continue
        boundaries = np.linspace(0, len(feat), args.phrases + 1).astype(int)
        for slot in range(args.phrases):
            query, music_event = calibrated_phrase_query(feat[int(boundaries[slot]):int(boundaries[slot + 1])], int(boundaries[slot]), int(boundaries[slot + 1]))
            compat = np.asarray([event_compatibility(music_event, e) for e in event_names], dtype=np.float32)
            dist = np.linalg.norm(motion_desc - query[None, :], axis=1)
            score = base + 0.70 * compat - 0.55 * dist
            pos_idx = np.argsort(score)[::-1][: max(1, args.positives_per_phrase)]
            hard_pool = np.argsort(base - 0.25 * compat + 0.20 * dist)[::-1]
            for pi in pos_idx:
                for _ in range(max(1, args.negatives_per_positive)):
                    # Choose a high-quality negative that is not among top positives.
                    candidates = [int(x) for x in hard_pool[: min(400, len(hard_pool))] if int(x) not in set(map(int, pos_idx))]
                    if not candidates:
                        continue
                    ni = int(rng.choice(candidates))
                    music_rows.append(query)
                    pos_rows.append(motion_desc[int(pi)])
                    neg_rows.append(motion_desc[ni])
                    labels.append(f"{music_file.stem}:slot{slot}:{music_event}")

    if not music_rows:
        raise RuntimeError("Router dataset is empty")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        music=np.stack(music_rows).astype(np.float32),
        positive=np.stack(pos_rows).astype(np.float32),
        negative=np.stack(neg_rows).astype(np.float32),
        label=np.asarray(labels, dtype=object),
    )
    print("saved:", out)
    print("samples:", len(music_rows))
    print("music_files:", len(music_files))


if __name__ == "__main__":
    main()
