#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect event-level strict warp feasibility for one locked music slot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v34_warp_aware_retrieval import _integer_content_interval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--slot_length", type=int, required=True)
    parser.add_argument("--min_content_frames", type=int, default=18)
    parser.add_argument("--transition_min_frames", type=int, default=16)
    parser.add_argument("--transition_max_frames", type=int, default=54)
    parser.add_argument("--min_warp", type=float, default=0.82)
    parser.add_argument("--max_warp", type=float, default=1.30)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--first_slot", type=int, default=0)
    args = parser.parse_args()

    arrays = np.load(Path(args.duration_index_npz), allow_pickle=True)
    natural = np.asarray(arrays["natural_duration"], np.float32)
    intervals = [
        _integer_content_interval(
            natural_duration=float(value),
            slot_length=args.slot_length,
            first_slot=bool(args.first_slot),
            min_content_frames=args.min_content_frames,
            transition_min_frames=args.transition_min_frames,
            transition_max_frames=args.transition_max_frames,
            minimum_warp=args.min_warp,
            maximum_warp=args.max_warp,
            tolerance=args.tolerance,
        )
        for value in natural
    ]
    feasible = np.asarray([value is not None for value in intervals], bool)
    examples = []
    for index in np.flatnonzero(feasible)[:20]:
        low, high = intervals[int(index)]
        examples.append({
            "event_index": int(index),
            "natural_duration": float(natural[index]),
            "content_interval": [int(low), int(high)],
            "transition_interval": [
                int(args.slot_length - high),
                int(args.slot_length - low),
            ],
        })
    result = {
        "slot_length": args.slot_length,
        "warp_bounds": [args.min_warp, args.max_warp],
        "natural_duration_min": float(natural.min()),
        "natural_duration_max": float(natural.max()),
        "natural_duration_quantiles": np.quantile(
            natural, [0, .05, .25, .5, .75, .95, 1]
        ).astype(float).tolist(),
        "num_events": int(len(natural)),
        "globally_feasible_events": int(feasible.sum()),
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not feasible.any():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
