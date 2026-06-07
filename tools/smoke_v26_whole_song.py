#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU smoke tests for V26 phrase segmentation, planner and duration allocator."""
from __future__ import annotations

import numpy as np
import torch

from model.v26_whole_song_planner import V26WholeSongPlanner
from tools.v26_global_duration_alignment import allocate_whole_song_durations
from tools.v26_music_phrase_segmentation import segment_music_phrases


def main() -> None:
    rng = np.random.default_rng(20260626)
    features = rng.random((900, 12), dtype=np.float32)
    features[280:285, 10] = 1.0
    features[590:595, 10] = 1.0
    phrases, _ = segment_music_phrases(
        features,
        fps=30.0,
        min_phrase_seconds=2.5,
        max_phrase_seconds=7.5,
    )
    assert len(phrases) >= 4
    planner_features = torch.tensor(
        np.stack([np.asarray(p.planner_feature, dtype=np.float32) for p in phrases])[None]
    )
    model = V26WholeSongPlanner()
    output = model(planner_features)
    assert output["event_logits"].shape[:2] == planner_features.shape[:2]
    transition = [0] + [10] * (len(phrases) - 1)
    allocation = allocate_whole_song_durations(
        phrase_lengths=[p.length for p in phrases],
        natural_durations=[max(24, p.length * 0.8) for p in phrases],
        planner_durations=[max(24, p.length * 0.9) for p in phrases],
        event_types=["neutral_flow"] * len(phrases),
        music_events=[p.music_event for p in phrases],
        transition_lengths=transition,
        total_frames=len(features),
    )
    assert sum(allocation["phrase_total_lengths"]) == len(features)
    print(
        f"[PASS] V26 smoke test: frames={len(features)} "
        f"phrases={len(phrases)} exact={sum(allocation['phrase_total_lengths'])}"
    )


if __name__ == "__main__":
    main()
