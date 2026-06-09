#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build fixed-length training windows for V27 transition diffusion.

The available library is event-based, not paired with arbitrary true
transitions.  To learn a local motion manifold, we sample windows inside real
Dunhuang events: endpoints are conditions and the interior frames are the
in-between target.  This trains the model to redraw plausible transition motion
without changing retrieved event semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion


def _pad_window(x: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    k = len(x)
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    out[:k] = x
    mask[:k] = 1.0
    return out, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--samples_per_event", type=int, default=3)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    if args.max_events > 0:
        items = items[: args.max_events]

    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    music: List[np.ndarray] = []
    lengths: List[int] = []
    event_ids: List[str] = []

    for i, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(path).astype(np.float32)
        except Exception:
            continue
        if len(motion) < args.min_len + 2:
            continue
        for _ in range(max(1, args.samples_per_event)):
            k = int(rng.integers(args.min_len, min(args.max_len, len(motion) - 2) + 1))
            start_idx = int(rng.integers(0, len(motion) - k - 1))
            end_idx = start_idx + k + 1
            interior = motion[start_idx + 1 : end_idx]
            padded, mask = _pad_window(interior, args.max_len)
            targets.append(padded)
            masks.append(mask)
            starts.append(motion[start_idx])
            ends.append(motion[end_idx])
            # Training windows are motion-only.  Music conditioning is kept as
            # zeros here; inference supplies the real phrase query.
            music.append(np.zeros((12,), dtype=np.float32))
            lengths.append(k)
            event_ids.append(str(item.get("event_id", i)))

    if not targets:
        raise RuntimeError("No transition diffusion samples were built.")

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        target=np.stack(targets).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        start=np.stack(starts).astype(np.float32),
        end=np.stack(ends).astype(np.float32),
        music=np.stack(music).astype(np.float32),
        length=np.asarray(lengths, dtype=np.int32),
        event_id=np.asarray(event_ids, dtype=object),
        meta=np.asarray(
            json.dumps(
                {
                    "num_samples": len(targets),
                    "max_len": int(args.max_len),
                    "min_len": int(args.min_len),
                    "samples_per_event": int(args.samples_per_event),
                    "source_index_json": str(args.index_json),
                    "source_duration_index_npz": str(args.duration_index_npz),
                },
                ensure_ascii=False,
            ),
            dtype=object,
        ),
    )
    print(f"[SAVED] {out} samples={len(targets)} max_len={args.max_len}")


if __name__ == "__main__":
    main()
