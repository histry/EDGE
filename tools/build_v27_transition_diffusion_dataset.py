#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build conditional training windows for V27/V28 transition diffusion.

This builder deliberately mixes three sample families:

1. intra_event_real:
   real in-between windows sampled inside Dunhuang events.  These preserve the
   authentic local motion manifold.
2. source_adjacent_bridge:
   adjacent events from the same source sequence when source metadata exists.
   The target is a smooth endpoint-conditioned bridge, and the condition is the
   average event/music proxy of the two events.
3. pseudo_pair_bridge:
   random feasible event pairs used to expose the denoiser to cross-event entry
   and exit poses.

The important change from the first V27 builder is that ``music`` is no longer
all zeros.  It is a deterministic 12D event/music condition in the same coarse
space used by the hierarchical scheduler.  This makes the transition diffusion
training condition-aware instead of receiving music only at inference time.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import ROOT_X, ROOT_Z, load_motion


EVENT_GROUPS = {
    "pose_hold": 0,
    "calm_flow": 1,
    "release": 1,
    "neutral_flow": 2,
    "build_up": 3,
    "high_tension": 3,
    "arm_flourish": 4,
    "support_shift": 5,
}


def _array_names(arrays: Any) -> set[str]:
    return set(arrays.files) if hasattr(arrays, "files") else set(arrays.keys())


def _array_or(arrays: Any, name: str, default: np.ndarray) -> np.ndarray:
    return np.asarray(arrays[name]) if name in _array_names(arrays) else np.asarray(default)


def _normalize(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, dtype=np.float32).reshape(-1)
    return x / max(float(np.linalg.norm(x)), 1e-8)


def _normalize01_scalar(value: float, lo: float, hi: float) -> float:
    if hi <= lo + 1e-8:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _event_group(event_type: str) -> int:
    return int(EVENT_GROUPS.get(str(event_type), 2))


def _event_condition(item: Mapping[str, Any], arrays: Any, idx: int) -> np.ndarray:
    """Build a 12D condition vector compatible with phrase-level query space."""
    n = len(_array_or(arrays, "natural_duration", np.zeros((1,), dtype=np.float32)))
    natural = _array_or(arrays, "natural_duration", np.full((n,), 41.0, dtype=np.float32)).astype(np.float32)
    style = _array_or(arrays, "style_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    quality = _array_or(arrays, "quality_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    safety = _array_or(arrays, "safety_score", np.full((n,), 0.5, dtype=np.float32)).astype(np.float32)
    motion_desc = _array_or(arrays, "motion_desc", np.zeros((n, 4), dtype=np.float32)).astype(np.float32)
    if motion_desc.ndim == 1:
        motion_desc = motion_desc.reshape(n, 1)
    turn_peak = _array_or(arrays, "turn_peak_dps", np.zeros((n,), dtype=np.float32)).astype(np.float32)
    turn_angle = _array_or(arrays, "turn_angle_deg", np.zeros((n,), dtype=np.float32)).astype(np.float32)

    event_type = str(item.get("event_type", "neutral_flow"))
    coarse = np.zeros((6,), dtype=np.float32)
    coarse[np.clip(_event_group(event_type), 0, 5)] = 1.0
    activity = float(motion_desc[idx, 0]) if idx < len(motion_desc) and motion_desc.shape[1] else 0.0
    turn = 0.55 * _normalize01_scalar(float(turn_peak[idx]) if idx < len(turn_peak) else 0.0, 0.0, 720.0)
    turn += 0.45 * _normalize01_scalar(float(turn_angle[idx]) if idx < len(turn_angle) else 0.0, 0.0, 420.0)
    duration = _normalize01_scalar(float(natural[idx]) if idx < len(natural) else 41.0, 24.0, 96.0)
    continuous = np.asarray(
        [
            np.clip(activity, 0.0, 1.0),
            np.clip(turn, 0.0, 1.0),
            duration,
            np.clip(float(style[idx]) if idx < len(style) else 0.5, 0.0, 1.0),
            np.clip(float(quality[idx]) if idx < len(quality) else 0.5, 0.0, 1.0),
            np.clip(float(safety[idx]) if idx < len(safety) else 0.5, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return _normalize(np.concatenate([coarse, continuous], axis=0))


def _condition_dropout(x: np.ndarray, rng: np.random.Generator, dropout: float) -> np.ndarray:
    if dropout <= 0.0:
        return x.astype(np.float32)
    if rng.random() < float(dropout):
        return np.zeros_like(x, dtype=np.float32)
    return x.astype(np.float32)


def _pad_window(x: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    k = len(x)
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    out[:k] = x
    mask[:k] = 1.0
    return out, mask


def _smoothstep(u: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(u, dtype=np.float32), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bridge_target(prev: np.ndarray, nxt: np.ndarray, length: int) -> np.ndarray:
    """Create a stable pseudo target between two endpoint poses."""
    k = max(1, int(length))
    start = np.asarray(prev[-1], dtype=np.float32)
    end = np.asarray(nxt[0], dtype=np.float32)
    alpha = _smoothstep(np.linspace(1.0 / (k + 1), k / (k + 1), k, dtype=np.float32)).reshape(-1, 1)
    bridge = (1.0 - alpha) * start.reshape(1, -1) + alpha * end.reshape(1, -1)
    if len(prev) >= k and len(nxt) >= k:
        tail = np.asarray(prev[-k:], dtype=np.float32)
        head = np.asarray(nxt[:k], dtype=np.float32)
        bridge = 0.55 * bridge + 0.20 * tail + 0.25 * head
    bridge[:, ROOT_X] = 0.0
    bridge[:, ROOT_Z] = 0.0
    return bridge.astype(np.float32)


def _item_sort_key(item: Mapping[str, Any], fallback: int) -> Tuple[int, int]:
    start = item.get("source_start", item.get("start", item.get("begin", fallback)))
    end = item.get("source_end", item.get("end", fallback))
    try:
        return int(start), int(end)
    except Exception:
        return int(fallback), int(fallback)


def _source_key(item: Mapping[str, Any]) -> str:
    for name in ("source_id", "source", "source_name", "video_id", "music_id"):
        if name in item:
            return str(item.get(name))
    path = str(item.get("pkl", item.get("path", "")))
    return Path(path).stem.split("_")[0] if path else "unknown"


def _add_sample(
    targets: List[np.ndarray],
    masks: List[np.ndarray],
    starts: List[np.ndarray],
    ends: List[np.ndarray],
    music: List[np.ndarray],
    lengths: List[int],
    sample_weight: List[float],
    sample_kind: List[str],
    start_event_ids: List[str],
    end_event_ids: List[str],
    target: np.ndarray,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    cond: np.ndarray,
    max_len: int,
    kind: str,
    weight: float,
    start_event_id: str,
    end_event_id: str,
) -> None:
    padded, mask = _pad_window(target.astype(np.float32), max_len)
    targets.append(padded)
    masks.append(mask)
    starts.append(np.asarray(start_frame, dtype=np.float32))
    ends.append(np.asarray(end_frame, dtype=np.float32))
    music.append(np.asarray(cond, dtype=np.float32))
    lengths.append(int(len(target)))
    sample_weight.append(float(weight))
    sample_kind.append(str(kind))
    start_event_ids.append(str(start_event_id))
    end_event_ids.append(str(end_event_id))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--samples_per_event", type=int, default=4)
    parser.add_argument("--source_pairs_per_event", type=float, default=0.45)
    parser.add_argument("--pseudo_pairs_per_event", type=float, default=0.75)
    parser.add_argument("--condition_dropout", type=float, default=0.05)
    parser.add_argument("--max_source_gap", type=int, default=12)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, arrays, items = load_shared_index(Path(args.index_json), Path(args.duration_index_npz))
    if args.max_events > 0:
        items = items[: args.max_events]
    n = len(items)

    motions: List[np.ndarray | None] = []
    conditions: List[np.ndarray] = []
    for i, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motions.append(load_motion(path).astype(np.float32))
        except Exception:
            motions.append(None)
        conditions.append(_event_condition(item, arrays, i))

    targets: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    starts: List[np.ndarray] = []
    ends: List[np.ndarray] = []
    music: List[np.ndarray] = []
    lengths: List[int] = []
    sample_weight: List[float] = []
    sample_kind: List[str] = []
    start_event_ids: List[str] = []
    end_event_ids: List[str] = []

    for i, (item, motion) in enumerate(zip(items, motions)):
        if motion is None or len(motion) < args.min_len + 2:
            continue
        for _ in range(max(1, int(args.samples_per_event))):
            upper = min(int(args.max_len), len(motion) - 2)
            if upper < args.min_len:
                continue
            k = int(rng.integers(args.min_len, upper + 1))
            start_idx = int(rng.integers(0, len(motion) - k - 1))
            end_idx = start_idx + k + 1
            cond = _condition_dropout(conditions[i], rng, args.condition_dropout)
            _add_sample(
                targets,
                masks,
                starts,
                ends,
                music,
                lengths,
                sample_weight,
                sample_kind,
                start_event_ids,
                end_event_ids,
                motion[start_idx + 1 : end_idx],
                motion[start_idx],
                motion[end_idx],
                cond,
                args.max_len,
                "intra_event_real",
                1.0,
                str(item.get("event_id", i)),
                str(item.get("event_id", i)),
            )

    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(items):
        grouped[_source_key(item)].append(i)
    source_pairs: List[Tuple[int, int]] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda idx: _item_sort_key(items[idx], idx))
        for a, b in zip(ordered, ordered[1:]):
            a_start, a_end = _item_sort_key(items[a], a)
            b_start, _ = _item_sort_key(items[b], b)
            if a_end > 0 and b_start > 0 and b_start - a_end > int(args.max_source_gap):
                continue
            if motions[a] is not None and motions[b] is not None:
                source_pairs.append((a, b))
    rng.shuffle(source_pairs)
    max_source_pairs = int(round(max(0.0, float(args.source_pairs_per_event)) * max(n, 1)))
    for a, b in source_pairs[:max_source_pairs]:
        prev = motions[a]
        nxt = motions[b]
        if prev is None or nxt is None or len(prev) < 2 or len(nxt) < 2:
            continue
        k = int(rng.integers(args.min_len, args.max_len + 1))
        target = _bridge_target(prev, nxt, k)
        cond = _normalize(0.5 * conditions[a] + 0.5 * conditions[b])
        cond = _condition_dropout(cond, rng, args.condition_dropout)
        _add_sample(
            targets,
            masks,
            starts,
            ends,
            music,
            lengths,
            sample_weight,
            sample_kind,
            start_event_ids,
            end_event_ids,
            target,
            prev[-1],
            nxt[0],
            cond,
            args.max_len,
            "source_adjacent_bridge",
            1.25,
            str(items[a].get("event_id", a)),
            str(items[b].get("event_id", b)),
        )

    pseudo_count = int(round(max(0.0, float(args.pseudo_pairs_per_event)) * max(n, 1)))
    valid = [i for i, motion in enumerate(motions) if motion is not None and len(motion) >= 2]
    for _ in range(pseudo_count):
        if len(valid) < 2:
            break
        a, b = rng.choice(valid, size=2, replace=False)
        prev = motions[int(a)]
        nxt = motions[int(b)]
        if prev is None or nxt is None:
            continue
        k = int(rng.integers(args.min_len, args.max_len + 1))
        target = _bridge_target(prev, nxt, k)
        cond = _normalize(0.5 * conditions[int(a)] + 0.5 * conditions[int(b)])
        cond = _condition_dropout(cond, rng, args.condition_dropout)
        _add_sample(
            targets,
            masks,
            starts,
            ends,
            music,
            lengths,
            sample_weight,
            sample_kind,
            start_event_ids,
            end_event_ids,
            target,
            prev[-1],
            nxt[0],
            cond,
            args.max_len,
            "pseudo_pair_bridge",
            0.85,
            str(items[int(a)].get("event_id", int(a))),
            str(items[int(b)].get("event_id", int(b))),
        )

    if not targets:
        raise RuntimeError("No transition diffusion samples were built.")

    music_arr = np.stack(music).astype(np.float32)
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    kind_values, kind_counts = np.unique(np.asarray(sample_kind, dtype=object), return_counts=True)
    meta = {
        "version": "v28_conditioned_transition_diffusion_dataset",
        "num_samples": len(targets),
        "max_len": int(args.max_len),
        "min_len": int(args.min_len),
        "samples_per_event": int(args.samples_per_event),
        "source_pairs_per_event": float(args.source_pairs_per_event),
        "pseudo_pairs_per_event": float(args.pseudo_pairs_per_event),
        "condition_dropout": float(args.condition_dropout),
        "music_nonzero_rate": float(np.mean(np.linalg.norm(music_arr, axis=1) > 1e-6)),
        "sample_kind_counts": {str(k): int(v) for k, v in zip(kind_values, kind_counts)},
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
    }
    np.savez_compressed(
        out,
        target=np.stack(targets).astype(np.float32),
        mask=np.stack(masks).astype(np.float32),
        start=np.stack(starts).astype(np.float32),
        end=np.stack(ends).astype(np.float32),
        music=music_arr,
        length=np.asarray(lengths, dtype=np.int32),
        sample_weight=np.asarray(sample_weight, dtype=np.float32),
        sample_kind=np.asarray(sample_kind, dtype=object),
        start_event_id=np.asarray(start_event_ids, dtype=object),
        end_event_id=np.asarray(end_event_ids, dtype=object),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(
        f"[SAVED] {out} samples={len(targets)} max_len={args.max_len} "
        f"music_nonzero_rate={meta['music_nonzero_rate']:.3f} kinds={meta['sample_kind_counts']}"
    )


if __name__ == "__main__":
    main()
