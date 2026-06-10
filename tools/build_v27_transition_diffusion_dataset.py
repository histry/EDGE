#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the V29 research transition dataset.

The historical filename is retained for compatibility.  The V29 builder:
  1. samples real intra-event windows;
  2. extracts true source gaps when a full source motion can be resolved;
  3. otherwise creates SO(3) endpoint-velocity-aware adjacent bridges;
  4. only creates pseudo-pair bridges after strict kinematic filtering;
  5. stores start/end source groups so validation can be split without leakage.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion
from tools.v29_motion_geometry import (
    CONTACT,
    MOTION_DIM,
    ROOT_X,
    ROOT_Z,
    endpoint_metrics_np,
    make_so3_transition,
    resample_motion_so3_np,
)

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


def _normalize01(value: float, lo: float, hi: float) -> float:
    return float(np.clip((float(value) - lo) / max(hi - lo, 1e-8), 0.0, 1.0))


def _event_condition(item: Mapping[str, Any], arrays: Any, idx: int) -> np.ndarray:
    n = len(_array_or(arrays, "natural_duration", np.zeros((1,), np.float32)))
    natural = _array_or(arrays, "natural_duration", np.full((n,), 41.0, np.float32))
    style = _array_or(arrays, "style_score", np.full((n,), 0.5, np.float32))
    quality = _array_or(arrays, "quality_score", np.full((n,), 0.5, np.float32))
    safety = _array_or(arrays, "safety_score", np.full((n,), 0.5, np.float32))
    motion_desc = _array_or(arrays, "motion_desc", np.zeros((n, 4), np.float32))
    turn_peak = _array_or(arrays, "turn_peak_dps", np.zeros((n,), np.float32))
    turn_angle = _array_or(arrays, "turn_angle_deg", np.zeros((n,), np.float32))
    if motion_desc.ndim == 1:
        motion_desc = motion_desc.reshape(n, 1)

    event_type = str(item.get("event_type", "neutral_flow"))
    coarse = np.zeros((6,), dtype=np.float32)
    coarse[np.clip(EVENT_GROUPS.get(event_type, 2), 0, 5)] = 1.0
    activity = float(motion_desc[idx, 0]) if motion_desc.shape[1] else 0.0
    turn = 0.55 * _normalize01(float(turn_peak[idx]), 0.0, 720.0)
    turn += 0.45 * _normalize01(float(turn_angle[idx]), 0.0, 420.0)
    continuous = np.asarray(
        [
            np.clip(activity, 0.0, 1.0),
            np.clip(turn, 0.0, 1.0),
            _normalize01(float(natural[idx]), 24.0, 96.0),
            np.clip(float(style[idx]), 0.0, 1.0),
            np.clip(float(quality[idx]), 0.0, 1.0),
            np.clip(float(safety[idx]), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return _normalize(np.concatenate([coarse, continuous]))


def _source_group(item: Mapping[str, Any], fallback: int) -> str:
    for name in ("source_id", "video_id", "source_name", "music_id", "source"):
        value = item.get(name)
        if value not in (None, ""):
            return f"{name}:{value}"
    path = str(item.get("pkl", item.get("path", "")))
    return f"path:{Path(path).stem if path else fallback}"


def _event_bounds(item: Mapping[str, Any], fallback: int) -> Tuple[int, int]:
    start = item.get("source_start", item.get("start", item.get("begin", fallback)))
    end = item.get("source_end", item.get("end", fallback))
    try:
        return int(start), int(end)
    except Exception:
        return fallback, fallback


def _candidate_source_paths(item: Mapping[str, Any]) -> List[Path]:
    keys = (
        "source_motion_path", "source_pkl", "full_motion_path",
        "canonical_source_path", "video_motion_path", "source_path",
    )
    paths = []
    for key in keys:
        value = item.get(key)
        if value:
            paths.append(Path(str(value)))
    # Some indexes point pkl/path to the full source rather than the cropped event.
    for key in ("pkl", "path"):
        value = item.get(key)
        if value:
            paths.append(Path(str(value)))
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _load_full_source_for_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    end_a: int,
    start_b: int,
) -> np.ndarray | None:
    if end_a < 0 or start_b <= end_a:
        return None
    first_paths = {str(p): p for p in _candidate_source_paths(first)}
    second_paths = {str(p): p for p in _candidate_source_paths(second)}
    for key in sorted(set(first_paths).intersection(second_paths)):
        path = first_paths[key]
        if not path.is_file():
            continue
        try:
            motion = load_motion(path)
        except Exception:
            continue
        if len(motion) > start_b + 1:
            return motion.astype(np.float32)
    return None


def _condition_dropout(
    x: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> np.ndarray:
    return np.zeros_like(x) if rng.random() < probability else x.astype(np.float32)


def _pad_window(x: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) > max_len:
        raise ValueError(f"Window {len(x)} exceeds max_len={max_len}")
    out = np.zeros((max_len, x.shape[1]), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    out[: len(x)] = x
    mask[: len(x)] = 1.0
    return out, mask


def _resample_if_needed(target: np.ndarray, length: int) -> np.ndarray:
    if len(target) == length:
        return target.astype(np.float32)
    positions = np.linspace(0.0, len(target) - 1, length, dtype=np.float32)
    return resample_motion_so3_np(target, positions)


class SampleStore:
    def __init__(self, max_len: int) -> None:
        self.max_len = int(max_len)
        self.rows: Dict[str, List[Any]] = defaultdict(list)

    def add(
        self,
        target: np.ndarray,
        start_frame: np.ndarray,
        end_frame: np.ndarray,
        condition: np.ndarray,
        kind: str,
        weight: float,
        start_event_id: str,
        end_event_id: str,
        start_group: str,
        end_group: str,
        real_target: bool,
    ) -> None:
        padded, mask = _pad_window(np.asarray(target, np.float32), self.max_len)
        self.rows["target"].append(padded)
        self.rows["mask"].append(mask)
        self.rows["start"].append(np.asarray(start_frame, np.float32))
        self.rows["end"].append(np.asarray(end_frame, np.float32))
        self.rows["music"].append(np.asarray(condition, np.float32))
        self.rows["length"].append(int(len(target)))
        self.rows["sample_weight"].append(float(weight))
        self.rows["sample_kind"].append(str(kind))
        self.rows["start_event_id"].append(str(start_event_id))
        self.rows["end_event_id"].append(str(end_event_id))
        self.rows["start_group"].append(str(start_group))
        self.rows["end_group"].append(str(end_group))
        self.rows["real_target"].append(bool(real_target))

    def __len__(self) -> int:
        return len(self.rows["target"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--samples_per_event", type=int, default=6)
    parser.add_argument("--source_pairs_per_event", type=float, default=1.0)
    parser.add_argument("--pseudo_pairs_per_event", type=float, default=0.45)
    parser.add_argument("--condition_dropout", type=float, default=0.08)
    parser.add_argument("--max_source_gap", type=int, default=96)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--pseudo_max_pose_deg", type=float, default=42.0)
    parser.add_argument("--pseudo_max_velocity_deg_s", type=float, default=260.0)
    parser.add_argument("--pseudo_max_root_y", type=float, default=0.18)
    parser.add_argument("--pseudo_max_contact_jump", type=float, default=0.75)
    parser.add_argument("--pseudo_max_attempt_factor", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    if args.min_len < 2 or args.max_len < args.min_len:
        raise ValueError("Require 2 <= min_len <= max_len")

    rng = np.random.default_rng(args.seed)
    _, arrays, items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    if args.max_events > 0:
        items = items[: args.max_events]
    n = len(items)

    motions: List[np.ndarray | None] = []
    conditions: List[np.ndarray] = []
    groups: List[str] = []
    for i, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(path).astype(np.float32)
            motion[:, ROOT_X] = 0.0
            motion[:, ROOT_Z] = 0.0
            motions.append(motion)
        except Exception:
            motions.append(None)
        conditions.append(_event_condition(item, arrays, i))
        groups.append(_source_group(item, i))

    store = SampleStore(args.max_len)

    # 1. True local motion windows.
    for i, (item, motion) in enumerate(zip(items, motions)):
        if motion is None or len(motion) < args.min_len + 2:
            continue
        upper = min(args.max_len, len(motion) - 2)
        for _ in range(max(1, args.samples_per_event)):
            k = int(rng.integers(args.min_len, upper + 1))
            start_idx = int(rng.integers(0, len(motion) - k - 1))
            end_idx = start_idx + k + 1
            store.add(
                target=motion[start_idx + 1 : end_idx],
                start_frame=motion[start_idx],
                end_frame=motion[end_idx],
                condition=_condition_dropout(
                    conditions[i], rng, args.condition_dropout
                ),
                kind="intra_event_real",
                weight=1.35,
                start_event_id=str(item.get("event_id", i)),
                end_event_id=str(item.get("event_id", i)),
                start_group=groups[i],
                end_group=groups[i],
                real_target=True,
            )

    # 2. Adjacent events in the same source.
    grouped: Dict[str, List[int]] = defaultdict(list)
    for i, group in enumerate(groups):
        grouped[group].append(i)
    adjacent_pairs: List[Tuple[int, int]] = []
    for group_indices in grouped.values():
        ordered = sorted(group_indices, key=lambda i: _event_bounds(items[i], i))
        adjacent_pairs.extend(zip(ordered, ordered[1:]))
    rng.shuffle(adjacent_pairs)
    max_adjacent = int(round(args.source_pairs_per_event * max(n, 1)))
    actual_gap_count = 0
    adjacent_so3_count = 0
    for a, b in adjacent_pairs[:max_adjacent]:
        prev, nxt = motions[a], motions[b]
        if prev is None or nxt is None or len(prev) < 2 or len(nxt) < 2:
            continue
        start_a, end_a = _event_bounds(items[a], a)
        start_b, _ = _event_bounds(items[b], b)
        gap = start_b - end_a - 1
        if gap > args.max_source_gap:
            continue

        source = _load_full_source_for_pair(items[a], items[b], end_a, start_b)
        target = None
        kind = "source_adjacent_so3"
        real_target = False
        if source is not None and gap >= args.min_len:
            raw_target = source[end_a + 1 : start_b]
            if len(raw_target) >= args.min_len:
                k = min(len(raw_target), args.max_len)
                target = _resample_if_needed(raw_target, k)
                kind = "source_gap_real"
                real_target = True
                actual_gap_count += 1
        if target is None:
            k = int(rng.integers(args.min_len, args.max_len + 1))
            target = make_so3_transition(prev, nxt, k)
            adjacent_so3_count += 1

        condition = _normalize(0.5 * conditions[a] + 0.5 * conditions[b])
        store.add(
            target=target,
            start_frame=prev[-1],
            end_frame=nxt[0],
            condition=_condition_dropout(
                condition, rng, args.condition_dropout
            ),
            kind=kind,
            weight=1.50 if real_target else 0.90,
            start_event_id=str(items[a].get("event_id", a)),
            end_event_id=str(items[b].get("event_id", b)),
            start_group=groups[a],
            end_group=groups[b],
            real_target=real_target,
        )

    # 3. Strictly filtered pseudo pairs.
    requested_pseudo = int(round(args.pseudo_pairs_per_event * max(n, 1)))
    valid = [i for i, motion in enumerate(motions) if motion is not None and len(motion) >= 2]
    accepted = 0
    attempts = 0
    max_attempts = max(requested_pseudo * args.pseudo_max_attempt_factor, 1)
    while accepted < requested_pseudo and attempts < max_attempts and len(valid) >= 2:
        attempts += 1
        a, b = [int(v) for v in rng.choice(valid, size=2, replace=False)]
        if groups[a] == groups[b]:
            continue
        prev, nxt = motions[a], motions[b]
        if prev is None or nxt is None:
            continue
        metrics = endpoint_metrics_np(prev, nxt)
        if metrics["pose_jump_deg_rms"] > args.pseudo_max_pose_deg:
            continue
        if metrics["velocity_jump_deg_s_rms"] > args.pseudo_max_velocity_deg_s:
            continue
        if metrics["root_y_jump"] > args.pseudo_max_root_y:
            continue
        if metrics["contact_jump"] > args.pseudo_max_contact_jump:
            continue
        k = int(rng.integers(args.min_len, args.max_len + 1))
        target = make_so3_transition(prev, nxt, k)
        condition = _normalize(0.5 * conditions[a] + 0.5 * conditions[b])
        store.add(
            target=target,
            start_frame=prev[-1],
            end_frame=nxt[0],
            condition=_condition_dropout(
                condition, rng, args.condition_dropout
            ),
            kind="pseudo_pair_so3_filtered",
            weight=0.55,
            start_event_id=str(items[a].get("event_id", a)),
            end_event_id=str(items[b].get("event_id", b)),
            start_group=groups[a],
            end_group=groups[b],
            real_target=False,
        )
        accepted += 1

    if len(store) == 0:
        raise RuntimeError("No V29 transition samples were built")

    music_arr = np.stack(store.rows["music"]).astype(np.float32)
    kind_values, kind_counts = np.unique(
        np.asarray(store.rows["sample_kind"], dtype=object),
        return_counts=True,
    )
    meta = {
        "version": "v29_so3_group_split_transition_dataset",
        "num_samples": len(store),
        "num_events": n,
        "num_source_groups": len(set(groups)),
        "max_len": args.max_len,
        "min_len": args.min_len,
        "samples_per_event": args.samples_per_event,
        "source_pairs_per_event": args.source_pairs_per_event,
        "pseudo_pairs_per_event": args.pseudo_pairs_per_event,
        "pseudo_requested": requested_pseudo,
        "pseudo_accepted": accepted,
        "pseudo_attempts": attempts,
        "actual_source_gap_count": actual_gap_count,
        "adjacent_so3_count": adjacent_so3_count,
        "music_nonzero_rate": float(
            np.mean(np.linalg.norm(music_arr, axis=1) > 1e-6)
        ),
        "sample_kind_counts": {
            str(k): int(v) for k, v in zip(kind_values, kind_counts)
        },
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
    }

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        target=np.stack(store.rows["target"]).astype(np.float32),
        mask=np.stack(store.rows["mask"]).astype(np.float32),
        start=np.stack(store.rows["start"]).astype(np.float32),
        end=np.stack(store.rows["end"]).astype(np.float32),
        music=music_arr,
        length=np.asarray(store.rows["length"], dtype=np.int32),
        sample_weight=np.asarray(store.rows["sample_weight"], dtype=np.float32),
        sample_kind=np.asarray(store.rows["sample_kind"], dtype=object),
        start_event_id=np.asarray(store.rows["start_event_id"], dtype=object),
        end_event_id=np.asarray(store.rows["end_event_id"], dtype=object),
        start_group=np.asarray(store.rows["start_group"], dtype=object),
        end_group=np.asarray(store.rows["end_group"], dtype=object),
        real_target=np.asarray(store.rows["real_target"], dtype=np.bool_),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
