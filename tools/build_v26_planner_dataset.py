#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build weakly supervised whole-song planner sequences from a music bank.

Pseudo labels are generated with the frozen V21 router, style/quality/safety
scores and the duration-augmented Event-RAG.  The learned planner is therefore
an optional amortized sequence planner, not a claim of ground-truth paired
music-dance supervision.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from model.v21_music_router import load_router_checkpoint
from tools.schedule_v21_multi_music import (
    load_shared_index,
    precompute_music_similarity,
    rule_transition_len,
)
from tools.v21_common import EVENT_TO_ID, EVENT_TYPES, event_compatibility, load_motion
from tools.v26_music_phrase_segmentation import (
    segment_music_phrases,
    whole_song_features,
)


def choose_sequence(
    phrases,
    arrays,
    items,
    router,
    device: torch.device,
    candidate_top_k: int,
    family_repeat_weight: float,
) -> List[int]:
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    style = np.asarray(arrays["style_score"], dtype=np.float32)
    quality = np.asarray(arrays["quality_score"], dtype=np.float32)
    safety = np.asarray(arrays["safety_score"], dtype=np.float32)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    queries = [np.asarray(p.query, dtype=np.float32) for p in phrases]
    similarities = precompute_music_similarity(router, queries, motion_desc, device)
    event_types = [str(x.get("event_type", "neutral_flow")) for x in items]
    families = [str(x.get("family_id", "")) for x in items]
    selected: List[int] = []
    for slot, phrase in enumerate(phrases):
        phrase_len = max(1, phrase.length)
        compat = np.asarray(
            [event_compatibility(phrase.music_event, event) for event in event_types],
            dtype=np.float32,
        )
        duration_match = 1.0 - np.minimum(np.abs(natural - phrase_len) / phrase_len, 1.0)
        base = (
            1.35 * style
            + 0.65 * quality
            + 0.35 * safety
            + 0.85 * similarities[slot]
            + 0.70 * compat
            + 0.35 * duration_match
        )
        shortlist = np.argsort(base)[::-1][: min(candidate_top_k, len(items))]
        best_idx = None
        best_score = -1e30
        for raw_idx in shortlist:
            idx = int(raw_idx)
            same_family = sum(1 for previous in selected if families[previous] == families[idx])
            score = float(base[idx] - family_repeat_weight * same_family)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            raise RuntimeError(f"No pseudo-label candidate for phrase {slot}")
        selected.append(best_idx)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--music_glob", required=True)
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--router_ckpt", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_phrase_seconds", type=float, default=2.5)
    parser.add_argument("--max_phrase_seconds", type=float, default=7.5)
    parser.add_argument("--boundary_quantile", type=float, default=0.68)
    parser.add_argument("--beat_snap_seconds", type=float, default=0.35)
    parser.add_argument("--candidate_top_k", type=int, default=1200)
    parser.add_argument("--family_repeat_weight", type=float, default=0.55)
    parser.add_argument("--max_songs", type=int, default=0)
    args = parser.parse_args()

    paths = [Path(x) for x in sorted(glob.glob(args.music_glob))]
    if args.max_songs > 0:
        paths = paths[: args.max_songs]
    if not paths:
        raise RuntimeError(f"No music matched: {args.music_glob}")

    meta, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    required = {"natural_duration", "motion_desc", "style_score", "quality_score", "safety_score"}
    missing = required.difference(arrays.files)
    if missing:
        raise RuntimeError(f"Duration index is missing arrays: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = load_router_checkpoint(args.router_ckpt, device=device)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    transition_lengths = (6, 8, 10, 12, 14, 16)

    sequence_features: List[np.ndarray] = []
    event_labels: List[np.ndarray] = []
    duration_targets: List[np.ndarray] = []
    transition_labels: List[np.ndarray] = []
    activity_targets: List[np.ndarray] = []
    song_keys: List[str] = []
    phrase_boundaries: List[np.ndarray] = []

    for song_index, path in enumerate(paths):
        features, _ = whole_song_features(path, fps=args.fps, cache_dir=args.cache_dir or None)
        phrases, segmentation = segment_music_phrases(
            features,
            fps=args.fps,
            min_phrase_seconds=args.min_phrase_seconds,
            max_phrase_seconds=args.max_phrase_seconds,
            boundary_quantile=args.boundary_quantile,
            beat_snap_seconds=args.beat_snap_seconds,
        )
        selected = choose_sequence(
            phrases,
            arrays,
            items,
            router,
            device,
            candidate_top_k=args.candidate_top_k,
            family_repeat_weight=args.family_repeat_weight,
        )
        feat = np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])
        labels = np.asarray(
            [EVENT_TO_ID.get(str(items[idx].get("event_type", "neutral_flow")), EVENT_TO_ID["neutral_flow"]) for idx in selected],
            dtype=np.int64,
        )
        durations = np.asarray([natural[idx] for idx in selected], dtype=np.float32)
        transitions: List[int] = []
        for i, idx in enumerate(selected):
            if i == 0:
                transitions.append(0)
            else:
                value = rule_transition_len(
                    phrases[i - 1].music_event,
                    str(items[idx].get("event_type", "neutral_flow")),
                    np.asarray(phrases[i].query, dtype=np.float32),
                )
                transitions.append(int(np.argmin(np.abs(np.asarray(transition_lengths) - value))))
        activities = np.asarray([motion_desc[idx, 0] for idx in selected], dtype=np.float32)

        sequence_features.append(feat)
        event_labels.append(labels)
        duration_targets.append(durations)
        transition_labels.append(np.asarray(transitions, dtype=np.int64))
        activity_targets.append(activities)
        song_keys.append(path.stem)
        phrase_boundaries.append(np.asarray(segmentation["boundaries"], dtype=np.int32))
        print(f"[V26 planner data] {song_index + 1}/{len(paths)} {path.name}: phrases={len(phrases)}")

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=np.asarray(sequence_features, dtype=object),
        event_labels=np.asarray(event_labels, dtype=object),
        duration_targets=np.asarray(duration_targets, dtype=object),
        transition_labels=np.asarray(transition_labels, dtype=object),
        activity_targets=np.asarray(activity_targets, dtype=object),
        song_keys=np.asarray(song_keys, dtype=object),
        phrase_boundaries=np.asarray(phrase_boundaries, dtype=object),
        transition_lengths=np.asarray(transition_lengths, dtype=np.int32),
        event_types=np.asarray(EVENT_TYPES, dtype=object),
    )
    print(f"[SAVED] {out}: songs={len(song_keys)}")


if __name__ == "__main__":
    main()
