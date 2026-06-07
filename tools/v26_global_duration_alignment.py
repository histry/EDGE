#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact whole-song integer duration allocation.

The allocator balances:
- music phrase lengths;
- event natural-duration priors;
- planner-predicted duration;
- event-specific elasticity;
while satisfying an exact global frame budget.

No hidden pad/trim operation is allowed after allocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def event_elasticity(event_type: str) -> float:
    """Larger values mean the event may absorb more timing adjustment."""
    table = {
        "pose_hold": 1.80,
        "calm_flow": 1.55,
        "neutral_flow": 1.25,
        "release": 1.20,
        "build_up": 1.00,
        "arm_flourish": 0.85,
        "support_shift": 0.75,
        "high_tension": 0.70,
    }
    return float(table.get(str(event_type), 1.0))


def event_importance(event_type: str, music_event: str) -> float:
    table = {
        "pose_hold": 0.70,
        "calm_flow": 0.85,
        "neutral_flow": 1.00,
        "release": 1.00,
        "build_up": 1.15,
        "arm_flourish": 1.30,
        "support_shift": 1.40,
        "high_tension": 1.50,
    }
    value = float(table.get(str(event_type), 1.0))
    if str(music_event) in {"climax", "accent", "section_change"}:
        value *= 1.15
    return value


def _continuous_projection(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    flexibility: np.ndarray,
    budget: float,
    iterations: int = 128,
) -> np.ndarray:
    x = np.clip(np.asarray(target, dtype=np.float64), lower, upper)
    flex = np.maximum(np.asarray(flexibility, dtype=np.float64), 1e-5)
    for _ in range(iterations):
        error = float(budget - x.sum())
        if abs(error) < 1e-7:
            break
        if error > 0:
            room = np.maximum(upper - x, 0.0)
        else:
            room = np.maximum(x - lower, 0.0)
        active = room > 1e-8
        if not np.any(active):
            break
        weight = room * flex * active
        if weight.sum() <= 1e-12:
            break
        delta = error * weight / weight.sum()
        if error > 0:
            delta = np.minimum(delta, room)
        else:
            delta = -np.minimum(-delta, room)
        x += delta
        x = np.clip(x, lower, upper)
    return x


def _integer_exact(
    continuous: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    budget: int,
    priority: np.ndarray,
) -> np.ndarray:
    base = np.floor(continuous).astype(np.int64)
    base = np.maximum(base, lower.astype(np.int64))
    base = np.minimum(base, upper.astype(np.int64))
    remainder = int(budget - int(base.sum()))
    fractional = continuous - np.floor(continuous)
    if remainder > 0:
        order = np.argsort(-(fractional + 1e-4 * priority))
        cursor = 0
        while remainder > 0:
            changed = False
            for index in order:
                if base[index] < int(upper[index]):
                    base[index] += 1
                    remainder -= 1
                    changed = True
                    if remainder == 0:
                        break
            cursor += 1
            if not changed or cursor > len(base) + 4:
                break
    elif remainder < 0:
        order = np.argsort(fractional + 1e-4 * priority)
        cursor = 0
        while remainder < 0:
            changed = False
            for index in order:
                if base[index] > int(lower[index]):
                    base[index] -= 1
                    remainder += 1
                    changed = True
                    if remainder == 0:
                        break
            cursor += 1
            if not changed or cursor > len(base) + 4:
                break
    if int(base.sum()) != int(budget):
        raise RuntimeError(
            f"Could not allocate exact frame budget: allocated={base.sum()} budget={budget}. "
            "Relax min/max warp or transition lengths."
        )
    return base.astype(np.int32)


def allocate_whole_song_durations(
    phrase_lengths: Sequence[int],
    natural_durations: Sequence[float],
    planner_durations: Sequence[float],
    event_types: Sequence[str],
    music_events: Sequence[str],
    transition_lengths: Sequence[int],
    total_frames: int,
    music_weight: float = 1.0,
    natural_weight: float = 1.25,
    planner_weight: float = 0.75,
    min_content_frames: int = 12,
    min_warp: float = 0.65,
    max_warp: float = 1.55,
) -> Dict[str, Any]:
    phrase = np.asarray(phrase_lengths, dtype=np.float64)
    natural = np.asarray(natural_durations, dtype=np.float64)
    planned = np.asarray(planner_durations, dtype=np.float64)
    transitions = np.asarray(transition_lengths, dtype=np.int32)
    n = len(phrase)
    if not (len(natural) == len(planned) == len(event_types) == len(music_events) == n):
        raise ValueError("All phrase/event sequences must have identical length")
    if len(transitions) != n:
        raise ValueError("transition_lengths must have one value per phrase; first must be zero")
    if n == 0:
        raise ValueError("At least one phrase is required")
    if int(transitions[0]) != 0:
        raise ValueError("The first transition length must be zero")

    content_budget = int(total_frames - int(transitions.sum()))
    if content_budget < n * int(min_content_frames):
        raise RuntimeError(
            f"Transitions consume too much of the song: content_budget={content_budget}, phrases={n}"
        )

    importance = np.asarray(
        [event_importance(e, m) for e, m in zip(event_types, music_events)],
        dtype=np.float64,
    )
    elasticity = np.asarray([event_elasticity(e) for e in event_types], dtype=np.float64)
    denominator = (
        music_weight
        + natural_weight * importance
        + planner_weight
    )
    target = (
        music_weight * np.maximum(phrase - transitions, min_content_frames)
        + natural_weight * importance * natural
        + planner_weight * planned
    ) / np.maximum(denominator, 1e-8)

    reference = np.maximum(natural, 1.0)
    lower = np.maximum(
        int(min_content_frames),
        np.floor(reference * float(min_warp)),
    )
    upper = np.maximum(
        lower,
        np.ceil(reference * float(max_warp)),
    )
    # Music phrase lengths remain a soft structural prior but should not create
    # impossible bounds for long/slow cultural actions.
    lower = np.minimum(lower, np.maximum(phrase - transitions, min_content_frames))
    upper = np.maximum(upper, np.maximum(phrase - transitions, min_content_frames))

    # High elasticity should receive more of the residual timing correction.
    flexibility = elasticity / np.maximum(importance, 1e-6)
    continuous = _continuous_projection(
        target,
        lower,
        upper,
        flexibility,
        float(content_budget),
    )
    allocation = _integer_exact(
        continuous,
        lower,
        upper,
        content_budget,
        priority=importance,
    )
    full_lengths = allocation + transitions
    if int(full_lengths.sum()) != int(total_frames):
        raise AssertionError("Whole-song duration allocation is not exact")

    boundaries = [0]
    for length in full_lengths:
        boundaries.append(boundaries[-1] + int(length))

    return {
        "total_frames": int(total_frames),
        "content_budget": int(content_budget),
        "content_lengths": allocation.tolist(),
        "transition_lengths": transitions.tolist(),
        "phrase_total_lengths": full_lengths.astype(int).tolist(),
        "output_boundaries": boundaries,
        "target_continuous": continuous.tolist(),
        "natural_durations": natural.tolist(),
        "planner_durations": planned.tolist(),
        "music_phrase_lengths": phrase.astype(int).tolist(),
        "warp_ratios": (allocation / np.maximum(natural, 1.0)).tolist(),
        "importance": importance.tolist(),
        "elasticity": elasticity.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = allocate_whole_song_durations(**payload)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.out_json}")


if __name__ == "__main__":
    main()
