#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build weakly supervised whole-song planner sequences from a music bank.

This replacement aligns the weak labels with the music-dominant V26 scheduler:

- duration pseudo labels prefer motions whose natural duration can be scaled
  toward the local music phrase by the phrase speed factor;
- transition pseudo labels use the music-derived transition profile with the
  expanded 12-48 frame vocabulary;
- physical boundary constraints remain a runtime safety lower bound, not a
  planner target that the planner cannot infer from music alone.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from model.v21_music_router import load_router_checkpoint
from model.v26_whole_song_planner import MUSIC_DOMINANT_TRANSITION_LENGTHS
from tools.schedule_v21_multi_music import load_shared_index, precompute_music_similarity
from tools.v21_common import EVENT_TO_ID, EVENT_TYPES, event_compatibility
from tools.v26_hierarchical_graph_scheduler import (
    build_slot_query,
    hierarchical_node_scores,
    load_or_build_hierarchy,
)
from tools.v26_music_phrase_segmentation import (
    segment_music_phrases,
    split_music_phrases_for_events,
    whole_song_features,
)


def _nearest_transition_class(frames: float, transition_lengths: Sequence[int]) -> int:
    values = np.asarray(transition_lengths, dtype=np.float32)
    return int(np.argmin(np.abs(values - float(frames))))


def _music_transition_target(phrase) -> int:
    base = int(getattr(phrase, "transition_base_frames", 24))
    profile = str(getattr(phrase, "transition_profile", "balanced"))
    if profile == "accent_cut":
        base = min(base, 24)
    elif profile in {"calm_sustain", "section_sustain"}:
        base = max(base, 24)
    elif profile == "tense_drive":
        base = int(round(0.65 * base + 0.35 * 18))
    return int(np.clip(base, MUSIC_DOMINANT_TRANSITION_LENGTHS[0], MUSIC_DOMINANT_TRANSITION_LENGTHS[-1]))


def choose_sequence(
    phrases,
    arrays,
    hierarchy,
    items,
    router,
    device: torch.device,
    candidate_top_k: int,
    graph_node_top_k: int,
    family_repeat_weight: float,
    hierarchical_retrieval: bool,
    hierarchy_weight: float,
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
        speed = max(float(getattr(phrase, "speed_factor", 1.0)), 1e-6)
        phrase_len = max(1.0, float(phrase.length))
        # If music is faster, a longer natural action may be compressed into
        # the phrase; if music is calmer, a shorter action may be stretched.
        target_natural = max(12.0, phrase_len * speed)
        compat = np.asarray(
            [event_compatibility(phrase.music_event, event) for event in event_types],
            dtype=np.float32,
        )
        duration_match = 1.0 - np.minimum(np.abs(natural - target_natural) / max(target_natural, 1.0), 1.0)
        activity_target = float(np.clip(getattr(phrase, "arousal", 0.5), 0.0, 1.0))
        activity_match = 1.0 - np.minimum(np.abs(motion_desc[:, 0] - activity_target), 1.0)
        hierarchy_score = np.zeros_like(style, dtype=np.float32)
        if hierarchical_retrieval:
            query = build_slot_query(
                phrase,
                predicted_event=str(getattr(phrase, "music_event", "neutral_flow")),
                target_natural=target_natural,
                desired_activity=activity_target,
            )
            hierarchy_score, _ = hierarchical_node_scores(hierarchy, query)
        base = (
            1.35 * style
            + 0.65 * quality
            + 0.35 * safety
            + 0.90 * similarities[slot]
            + 0.70 * compat
            + 0.45 * duration_match
            + 0.20 * activity_match
            + float(hierarchy_weight) * hierarchy_score
        )
        node_top_k = int(candidate_top_k)
        if hierarchical_retrieval and int(graph_node_top_k) > 0:
            node_top_k = min(node_top_k, int(graph_node_top_k))
        shortlist = np.argsort(base)[::-1][: min(node_top_k, len(items))]
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
    parser.add_argument("--hierarchy_index_npz", default="")
    parser.add_argument("--hyperbolic_ckpt", default="")
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--cache_dir", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_phrase_seconds", type=float, default=2.5)
    parser.add_argument("--max_phrase_seconds", type=float, default=7.5)
    parser.add_argument("--boundary_quantile", type=float, default=0.68)
    parser.add_argument("--beat_snap_seconds", type=float, default=0.35)
    parser.add_argument("--multi_event_phrases", type=int, default=1)
    parser.add_argument("--max_single_event_seconds", type=float, default=3.20)
    parser.add_argument("--calm_max_single_event_seconds", type=float, default=2.80)
    parser.add_argument("--min_subphrase_seconds", type=float, default=1.60)
    parser.add_argument("--max_events_per_phrase", type=int, default=4)
    parser.add_argument("--slot_beat_snap_seconds", type=float, default=0.25)
    parser.add_argument("--candidate_top_k", type=int, default=1200)
    parser.add_argument("--graph_node_top_k", type=int, default=0)
    parser.add_argument("--family_repeat_weight", type=float, default=0.55)
    parser.add_argument("--hierarchical_retrieval", type=int, default=1)
    parser.add_argument("--hierarchy_weight", type=float, default=0.55)
    parser.add_argument("--max_songs", type=int, default=0)
    args = parser.parse_args()

    paths = [Path(x) for x in sorted(glob.glob(args.music_glob, recursive=True))]
    if args.max_songs > 0:
        paths = paths[: args.max_songs]
    if not paths:
        raise RuntimeError(f"No music matched: {args.music_glob}")

    _, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    required = {"natural_duration", "motion_desc", "style_score", "quality_score", "safety_score"}
    missing = required.difference(arrays.files)
    if missing:
        raise RuntimeError(f"Duration index is missing arrays: {sorted(missing)}")
    hierarchy = load_or_build_hierarchy(arrays, items, args.hierarchy_index_npz, hyperbolic_ckpt=args.hyperbolic_ckpt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = load_router_checkpoint(args.router_ckpt, device=device)
    natural = np.asarray(arrays["natural_duration"], dtype=np.float32)
    motion_desc = np.asarray(arrays["motion_desc"], dtype=np.float32)
    transition_lengths = MUSIC_DOMINANT_TRANSITION_LENGTHS

    sequence_features: List[np.ndarray] = []
    event_labels: List[np.ndarray] = []
    duration_targets: List[np.ndarray] = []
    transition_labels: List[np.ndarray] = []
    activity_targets: List[np.ndarray] = []
    song_keys: List[str] = []
    phrase_boundaries: List[np.ndarray] = []
    speed_factors: List[np.ndarray] = []
    transition_frame_targets: List[np.ndarray] = []

    for song_index, path in enumerate(paths):
        features, _ = whole_song_features(path, fps=args.fps, cache_dir=args.cache_dir or None)
        source_phrases, segmentation = segment_music_phrases(
            features,
            fps=args.fps,
            min_phrase_seconds=args.min_phrase_seconds,
            max_phrase_seconds=args.max_phrase_seconds,
            boundary_quantile=args.boundary_quantile,
            beat_snap_seconds=args.beat_snap_seconds,
        )
        phrases, slot_expansion = split_music_phrases_for_events(
            features,
            source_phrases,
            fps=args.fps,
            enabled=bool(args.multi_event_phrases),
            max_slot_seconds=args.max_single_event_seconds,
            min_slot_seconds=args.min_subphrase_seconds,
            max_events_per_phrase=args.max_events_per_phrase,
            beat_snap_seconds=args.slot_beat_snap_seconds,
            calm_max_slot_seconds=args.calm_max_single_event_seconds,
        )
        selected = choose_sequence(
            phrases,
            arrays,
            hierarchy,
            items,
            router,
            device,
            candidate_top_k=args.candidate_top_k,
            graph_node_top_k=args.graph_node_top_k,
            family_repeat_weight=args.family_repeat_weight,
            hierarchical_retrieval=bool(args.hierarchical_retrieval),
            hierarchy_weight=args.hierarchy_weight,
        )
        feat = np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])
        labels = np.asarray(
            [EVENT_TO_ID.get(str(items[idx].get("event_type", "neutral_flow")), EVENT_TO_ID["neutral_flow"]) for idx in selected],
            dtype=np.int64,
        )
        durations = np.asarray([natural[idx] for idx in selected], dtype=np.float32)
        transition_frames: List[int] = []
        transitions: List[int] = []
        for i, phrase in enumerate(phrases):
            if i == 0:
                frames = 0
                transitions.append(0)
            else:
                frames = _music_transition_target(phrase)
                transitions.append(_nearest_transition_class(frames, transition_lengths))
            transition_frames.append(int(frames))
        activities = np.asarray(
            [
                0.65 * float(motion_desc[idx, 0])
                + 0.35 * float(np.clip(getattr(phrases[i], "arousal", 0.5), 0.0, 1.0))
                for i, idx in enumerate(selected)
            ],
            dtype=np.float32,
        )

        sequence_features.append(feat)
        event_labels.append(labels)
        duration_targets.append(durations)
        transition_labels.append(np.asarray(transitions, dtype=np.int64))
        activity_targets.append(activities)
        song_keys.append(path.stem)
        phrase_boundaries.append(np.asarray(slot_expansion.get("slot_boundaries", segmentation["boundaries"]), dtype=np.int32))
        speed_factors.append(np.asarray([float(getattr(p, "speed_factor", 1.0)) for p in phrases], dtype=np.float32))
        transition_frame_targets.append(np.asarray(transition_frames, dtype=np.int32))
        print(
            f"[V26 planner data] {song_index + 1}/{len(paths)} {path.name}: "
            f"source_phrases={len(source_phrases)} slots={len(phrases)} "
            f"transition_mean={np.mean(transition_frames[1:] or [0]):.1f}",
            flush=True,
        )

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
        music_speed_factors=np.asarray(speed_factors, dtype=object),
        transition_frame_targets=np.asarray(transition_frame_targets, dtype=object),
        transition_lengths=np.asarray(transition_lengths, dtype=np.int32),
        event_types=np.asarray(EVENT_TYPES, dtype=object),
    )
    print(f"[SAVED] {out}: songs={len(song_keys)}")


if __name__ == "__main__":
    main()
