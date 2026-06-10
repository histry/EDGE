#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the V30 continuous-transition research dataset.

The historical filename is retained for command compatibility.  V30 separates
four supervision sources and records them explicitly:

1. ``intra_event_real``: real masked intervals inside an event;
2. ``source_gap_real``: real frames omitted between adjacent indexed events;
3. ``source_boundary_mask_real``: a real interval masked across an adjacent
   event boundary, available even when the two event crops are contiguous;
4. synthetic SO(3) bridges, retained only as lower-weight regularisation.

For publication runs, provide a source manifest/full-motion root and enable the
strict real-boundary gate.  The builder will fail rather than silently train a
transition model whose boundary supervision is entirely synthetic.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion
from tools.v29_motion_geometry import (
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


def _normalise(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _normalise01(value: float, low: float, high: float) -> float:
    return float(np.clip((float(value) - low) / max(high - low, 1e-8), 0.0, 1.0))


def _event_condition(item: Mapping[str, Any], arrays: Any, index: int) -> np.ndarray:
    n = len(_array_or(arrays, "natural_duration", np.zeros((1,), np.float32)))
    natural = _array_or(arrays, "natural_duration", np.full((n,), 41.0, np.float32))
    style = _array_or(arrays, "style_score", np.full((n,), 0.5, np.float32))
    quality = _array_or(arrays, "quality_score", np.full((n,), 0.5, np.float32))
    safety = _array_or(arrays, "safety_score", np.full((n,), 0.5, np.float32))
    descriptor = _array_or(arrays, "motion_desc", np.zeros((n, 12), np.float32))
    turn_peak = _array_or(arrays, "turn_peak_dps", np.zeros((n,), np.float32))
    turn_angle = _array_or(arrays, "turn_angle_deg", np.zeros((n,), np.float32))
    if descriptor.ndim == 1:
        descriptor = descriptor.reshape(n, 1)

    coarse = np.zeros((6,), dtype=np.float32)
    event_type = str(item.get("event_type", "neutral_flow"))
    coarse[np.clip(EVENT_GROUPS.get(event_type, 2), 0, 5)] = 1.0
    activity = float(descriptor[index, 0]) if descriptor.shape[1] else 0.0
    turn = (
        0.55 * _normalise01(float(turn_peak[index]), 0.0, 720.0)
        + 0.45 * _normalise01(float(turn_angle[index]), 0.0, 420.0)
    )
    continuous = np.asarray(
        [
            np.clip(activity, 0.0, 1.0),
            np.clip(turn, 0.0, 1.0),
            _normalise01(float(natural[index]), 24.0, 96.0),
            np.clip(float(style[index]), 0.0, 1.0),
            np.clip(float(quality[index]), 0.0, 1.0),
            np.clip(float(safety[index]), 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return _normalise(np.concatenate([coarse, continuous]))


def _source_group(item: Mapping[str, Any], fallback: int) -> str:
    for key in (
        "source_id", "video_id", "sequence_id", "source_name",
        "music_id", "source", "original_video",
    ):
        value = item.get(key)
        if value not in (None, ""):
            return f"{key}:{value}"
    path = str(item.get("pkl", item.get("path", "")))
    stem = Path(path).stem if path else str(fallback)
    # Strip common event/window suffixes while keeping the source identity.
    stem = re.sub(r"(?:_event|_unit|_clip|_window)[_-]?\d+.*$", "", stem)
    return f"path:{stem}"


def _explicit_bounds(item: Mapping[str, Any]) -> Tuple[int | None, int | None]:
    pairs = (
        ("source_start", "source_end"),
        ("start_frame", "end_frame"),
        ("frame_start", "frame_end"),
        ("begin", "end"),
        ("start", "end"),
    )
    for left, right in pairs:
        if item.get(left) is not None and item.get(right) is not None:
            try:
                return int(item[left]), int(item[right])
            except (TypeError, ValueError):
                pass
    text = " ".join(
        str(item.get(key, "")) for key in ("event_id", "pkl", "path")
    )
    patterns = (
        r"(?:^|[_-])s(?:tart)?[_-]?(\d+).*?(?:^|[_-])e(?:nd)?[_-]?(\d+)",
        r"(?:frame|frames)[_-]?(\d+)[_-](\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def _load_manifest(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {"sources": {}, "events": {}}
    manifest_path = Path(str(path))
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        events = {}
        sources = {}
        for row in data:
            if not isinstance(row, dict):
                continue
            event_id = str(row.get("event_id", ""))
            if event_id:
                events[event_id] = row
            source_id = str(row.get("source_id", row.get("source", "")))
            source_motion = row.get("source_motion", row.get("motion"))
            if source_id and source_motion:
                sources[source_id] = {"motion": source_motion}
        return {"sources": sources, "events": events}
    if not isinstance(data, dict):
        raise ValueError("Source manifest must be a JSON object or list")
    return {
        "sources": dict(data.get("sources", {})),
        "events": dict(data.get("events", {})),
    }


def _manifest_event(
    manifest: Mapping[str, Any], item: Mapping[str, Any]
) -> Mapping[str, Any]:
    event_id = str(item.get("event_id", ""))
    row = manifest.get("events", {}).get(event_id, {})
    return row if isinstance(row, dict) else {}


def _bounds(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Tuple[int | None, int | None]:
    row = _manifest_event(manifest, item)
    if row:
        start, end = _explicit_bounds(row)
        if start is not None and end is not None:
            return start, end
    return _explicit_bounds(item)


def _candidate_source_paths(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    group: str,
    full_motion_root: str | Path | None,
) -> List[Path]:
    candidates: List[Path] = []
    event_row = _manifest_event(manifest, item)
    for row in (event_row, item):
        for key in (
            "source_motion", "source_motion_path", "full_motion_path",
            "source_pkl", "canonical_source_path", "video_motion_path",
        ):
            value = row.get(key) if isinstance(row, Mapping) else None
            if value:
                candidates.append(Path(str(value)))

    source_keys = []
    for row in (event_row, item):
        if not isinstance(row, Mapping):
            continue
        for key in ("source_id", "video_id", "sequence_id", "source"):
            value = row.get(key)
            if value not in (None, ""):
                source_keys.append(str(value))
    source_keys.append(group.split(":", 1)[-1])
    for source_key in source_keys:
        source_row = manifest.get("sources", {}).get(source_key, {})
        if isinstance(source_row, str):
            candidates.append(Path(source_row))
        elif isinstance(source_row, Mapping):
            value = source_row.get("motion", source_row.get("path"))
            if value:
                candidates.append(Path(str(value)))

    if full_motion_root:
        root = Path(str(full_motion_root))
        for source_key in source_keys:
            safe = Path(source_key).stem
            for suffix in (".pkl", ".npy", ".npz"):
                candidates.append(root / f"{safe}{suffix}")
        # Fallback recursive stem lookup is performed only for exact stems.
        if root.is_dir():
            wanted = {Path(key).stem for key in source_keys if key}
            for candidate in root.rglob("*"):
                if candidate.is_file() and candidate.suffix.lower() in {".pkl", ".npy", ".npz"}:
                    if candidate.stem in wanted:
                        candidates.append(candidate)

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        value = str(candidate)
        if value not in seen:
            seen.add(value)
            unique.append(candidate)
    return unique


class SourceCache:
    def __init__(self) -> None:
        self.cache: Dict[str, np.ndarray | None] = {}

    def load(self, paths: Sequence[Path], minimum_length: int) -> Tuple[np.ndarray | None, str]:
        for path in paths:
            key = str(path)
            if key not in self.cache:
                try:
                    self.cache[key] = load_motion(path).astype(np.float32) if path.is_file() else None
                except Exception:
                    self.cache[key] = None
            motion = self.cache[key]
            if motion is not None and len(motion) >= int(minimum_length):
                motion = motion.copy()
                motion[:, ROOT_X] = 0.0
                motion[:, ROOT_Z] = 0.0
                return motion, key
        return None, ""


def _drop_condition(
    condition: np.ndarray,
    rng: np.random.Generator,
    probability: float,
) -> np.ndarray:
    return np.zeros_like(condition) if rng.random() < probability else condition.copy()


def _resample(target: np.ndarray, length: int) -> np.ndarray:
    if len(target) == length:
        return np.asarray(target, dtype=np.float32)
    positions = np.linspace(0.0, len(target) - 1, length, dtype=np.float32)
    return resample_motion_so3_np(target, positions)


class SampleStore:
    def __init__(self, max_len: int) -> None:
        self.max_len = int(max_len)
        self.rows: Dict[str, List[Any]] = defaultdict(list)

    def add(
        self,
        target: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        music: np.ndarray,
        kind: str,
        weight: float,
        start_event_id: str,
        end_event_id: str,
        start_group: str,
        end_group: str,
        real_target: bool,
        source_path: str = "",
        source_start_frame: int = -1,
        source_end_frame: int = -1,
    ) -> None:
        target = np.asarray(target, dtype=np.float32)
        if len(target) < 2 or len(target) > self.max_len:
            return
        padded = np.zeros((self.max_len, MOTION_DIM), dtype=np.float32)
        mask = np.zeros((self.max_len,), dtype=np.float32)
        padded[: len(target)] = target
        mask[: len(target)] = 1.0
        self.rows["target"].append(padded)
        self.rows["mask"].append(mask)
        self.rows["start"].append(np.asarray(start, dtype=np.float32))
        self.rows["end"].append(np.asarray(end, dtype=np.float32))
        self.rows["music"].append(np.asarray(music, dtype=np.float32))
        self.rows["length"].append(int(len(target)))
        self.rows["sample_weight"].append(float(weight))
        self.rows["sample_kind"].append(str(kind))
        self.rows["start_event_id"].append(str(start_event_id))
        self.rows["end_event_id"].append(str(end_event_id))
        self.rows["start_group"].append(str(start_group))
        self.rows["end_group"].append(str(end_group))
        self.rows["real_target"].append(bool(real_target))
        self.rows["source_path"].append(str(source_path))
        self.rows["source_start_frame"].append(int(source_start_frame))
        self.rows["source_end_frame"].append(int(source_end_frame))

    def __len__(self) -> int:
        return len(self.rows["target"])


def _load_external_prior(path: str | Path, store: SampleStore) -> int:
    if not path:
        return 0
    prior_path = Path(str(path))
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)
    data = np.load(prior_path, allow_pickle=True)
    required = {"target", "start", "end"}
    missing = required.difference(data.files)
    if missing:
        raise RuntimeError(f"External prior is missing arrays: {sorted(missing)}")
    target = np.asarray(data["target"], dtype=np.float32)
    start = np.asarray(data["start"], dtype=np.float32)
    end = np.asarray(data["end"], dtype=np.float32)
    music = np.asarray(
        data["music"] if "music" in data.files else np.zeros((len(target), 12)),
        dtype=np.float32,
    )
    length = np.asarray(
        data["length"] if "length" in data.files else np.full((len(target),), target.shape[1]),
        dtype=np.int32,
    )
    count = 0
    for index in range(len(target)):
        k = int(length[index])
        store.add(
            target=target[index, :k],
            start=start[index],
            end=end[index],
            music=music[index],
            kind="external_motion_prior",
            weight=0.35,
            start_event_id=f"external:{index}",
            end_event_id=f"external:{index}",
            start_group=f"external:{index}",
            end_group=f"external:{index}",
            real_target=True,
            source_path=str(prior_path),
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--source_manifest", default="")
    parser.add_argument("--full_motion_root", default="")
    parser.add_argument("--external_prior_npz", default="")
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--samples_per_event", type=int, default=6)
    parser.add_argument("--real_masks_per_boundary", type=int, default=3)
    parser.add_argument("--source_pairs_per_event", type=float, default=1.0)
    parser.add_argument("--pseudo_pairs_per_event", type=float, default=0.25)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--max_source_gap", type=int, default=120)
    parser.add_argument("--allow_synthetic_adjacent", type=int, default=1)
    parser.add_argument("--require_real_boundary_count", type=int, default=0)
    parser.add_argument("--require_real_boundary_ratio", type=float, default=0.0)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--pseudo_max_pose_deg", type=float, default=38.0)
    parser.add_argument("--pseudo_max_velocity_deg_s", type=float, default=220.0)
    parser.add_argument("--pseudo_max_root_y", type=float, default=0.15)
    parser.add_argument("--pseudo_max_contact_jump", type=float, default=0.50)
    parser.add_argument("--pseudo_max_attempt_factor", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    if args.min_len < 2 or args.max_len < args.min_len:
        raise ValueError("Require 2 <= min_len <= max_len")
    rng = np.random.default_rng(args.seed)
    manifest = _load_manifest(args.source_manifest)
    _, arrays, all_items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    items = all_items[: args.max_events] if args.max_events > 0 else all_items
    n = len(items)

    motions: List[np.ndarray | None] = []
    groups: List[str] = []
    conditions: List[np.ndarray] = []
    bounds: List[Tuple[int | None, int | None]] = []
    for index, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(path).astype(np.float32)
            motion[:, ROOT_X] = 0.0
            motion[:, ROOT_Z] = 0.0
        except Exception:
            motion = None
        motions.append(motion)
        groups.append(_source_group(item, index))
        conditions.append(_event_condition(item, arrays, index))
        bounds.append(_bounds(item, manifest))

    store = SampleStore(args.max_len)

    # Real masked windows within events.
    for index, (item, motion) in enumerate(zip(items, motions)):
        if motion is None or len(motion) < args.min_len + 2:
            continue
        upper = min(args.max_len, len(motion) - 2)
        for _ in range(max(1, args.samples_per_event)):
            k = int(rng.integers(args.min_len, upper + 1))
            left = int(rng.integers(0, len(motion) - k - 1))
            right = left + k + 1
            store.add(
                target=motion[left + 1 : right],
                start=motion[left],
                end=motion[right],
                music=_drop_condition(conditions[index], rng, args.condition_dropout),
                kind="intra_event_real",
                weight=1.20,
                start_event_id=str(item.get("event_id", index)),
                end_event_id=str(item.get("event_id", index)),
                start_group=groups[index],
                end_group=groups[index],
                real_target=True,
                source_path=str(item.get("pkl", item.get("path", ""))),
                source_start_frame=(bounds[index][0] + left + 1) if bounds[index][0] is not None else -1,
                source_end_frame=(bounds[index][0] + right) if bounds[index][0] is not None else -1,
            )

    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        grouped[group].append(index)
    adjacent_pairs: List[Tuple[int, int]] = []
    for indices in grouped.values():
        ordered = sorted(
            indices,
            key=lambda idx: (
                bounds[idx][0] if bounds[idx][0] is not None else idx,
                idx,
            ),
        )
        adjacent_pairs.extend(list(zip(ordered, ordered[1:])))
    rng.shuffle(adjacent_pairs)
    max_adjacent = int(round(args.source_pairs_per_event * max(n, 1)))
    source_cache = SourceCache()
    actual_gap_count = 0
    real_boundary_mask_count = 0
    synthetic_adjacent_count = 0

    for first, second in adjacent_pairs[:max_adjacent]:
        prev, nxt = motions[first], motions[second]
        if prev is None or nxt is None or len(prev) < 2 or len(nxt) < 2:
            continue
        start_a, end_a = bounds[first]
        start_b, end_b = bounds[second]
        minimum_source_length = max(
            (end_b or 0) + args.max_len + 4,
            (start_b or 0) + args.max_len + 4,
        )
        candidates = _candidate_source_paths(
            items[first],
            manifest,
            groups[first],
            args.full_motion_root,
        )
        full_source, source_path = source_cache.load(
            candidates, minimum_length=max(minimum_source_length, 8)
        )
        condition = _normalise(0.5 * conditions[first] + 0.5 * conditions[second])
        event_a = str(items[first].get("event_id", first))
        event_b = str(items[second].get("event_id", second))

        if (
            full_source is not None
            and end_a is not None
            and start_b is not None
            and 0 <= end_a < start_b < len(full_source)
        ):
            gap = full_source[end_a + 1 : start_b]
            if args.min_len <= len(gap) <= args.max_source_gap:
                k = min(len(gap), args.max_len)
                store.add(
                    target=_resample(gap, k),
                    start=full_source[end_a],
                    end=full_source[start_b],
                    music=_drop_condition(condition, rng, args.condition_dropout),
                    kind="source_gap_real",
                    weight=2.00,
                    start_event_id=event_a,
                    end_event_id=event_b,
                    start_group=groups[first],
                    end_group=groups[second],
                    real_target=True,
                    source_path=source_path,
                    source_start_frame=end_a + 1,
                    source_end_frame=start_b,
                )
                actual_gap_count += 1

            # Real masked intervals spanning the indexed event boundary.  This
            # remains available when the two event crops are contiguous.
            boundary = int(round(0.5 * (end_a + start_b)))
            for _ in range(max(0, args.real_masks_per_boundary)):
                k = int(rng.integers(args.min_len, args.max_len + 1))
                jitter = int(rng.integers(-max(1, k // 6), max(2, k // 6 + 1)))
                left = boundary - k // 2 - 1 + jitter
                right = left + k + 1
                if left < 0 or right >= len(full_source):
                    continue
                # Require the hidden interval to genuinely cross the event boundary.
                if not (left < end_a < right or left < start_b < right):
                    continue
                store.add(
                    target=full_source[left + 1 : right],
                    start=full_source[left],
                    end=full_source[right],
                    music=_drop_condition(condition, rng, args.condition_dropout),
                    kind="source_boundary_mask_real",
                    weight=2.20,
                    start_event_id=event_a,
                    end_event_id=event_b,
                    start_group=groups[first],
                    end_group=groups[second],
                    real_target=True,
                    source_path=source_path,
                    source_start_frame=left + 1,
                    source_end_frame=right,
                )
                real_boundary_mask_count += 1

        if bool(args.allow_synthetic_adjacent):
            k = int(rng.integers(args.min_len, args.max_len + 1))
            store.add(
                target=make_so3_transition(prev, nxt, k),
                start=prev[-1],
                end=nxt[0],
                music=_drop_condition(condition, rng, args.condition_dropout),
                kind="source_adjacent_so3_synthetic",
                weight=0.55,
                start_event_id=event_a,
                end_event_id=event_b,
                start_group=groups[first],
                end_group=groups[second],
                real_target=False,
            )
            synthetic_adjacent_count += 1

    # Strictly filtered cross-source pseudo pairs are only a weak regulariser.
    requested_pseudo = int(round(args.pseudo_pairs_per_event * max(n, 1)))
    valid = [index for index, motion in enumerate(motions) if motion is not None and len(motion) >= 2]
    accepted = 0
    attempts = 0
    maximum_attempts = max(requested_pseudo * args.pseudo_max_attempt_factor, 1)
    while accepted < requested_pseudo and attempts < maximum_attempts and len(valid) >= 2:
        attempts += 1
        first, second = [int(value) for value in rng.choice(valid, 2, replace=False)]
        if groups[first] == groups[second]:
            continue
        prev, nxt = motions[first], motions[second]
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
        condition = _normalise(0.5 * conditions[first] + 0.5 * conditions[second])
        store.add(
            target=make_so3_transition(prev, nxt, k),
            start=prev[-1],
            end=nxt[0],
            music=_drop_condition(condition, rng, args.condition_dropout),
            kind="pseudo_pair_so3_filtered",
            weight=0.25,
            start_event_id=str(items[first].get("event_id", first)),
            end_event_id=str(items[second].get("event_id", second)),
            start_group=groups[first],
            end_group=groups[second],
            real_target=False,
        )
        accepted += 1

    external_count = _load_external_prior(args.external_prior_npz, store)
    if len(store) == 0:
        raise RuntimeError("No transition samples were built")

    kinds, counts = np.unique(
        np.asarray(store.rows["sample_kind"], dtype=object), return_counts=True
    )
    real_boundary_total = actual_gap_count + real_boundary_mask_count
    real_boundary_ratio = real_boundary_total / max(len(store), 1)
    if real_boundary_total < int(args.require_real_boundary_count):
        raise RuntimeError(
            "Insufficient real event-boundary supervision: "
            f"found={real_boundary_total}, required={args.require_real_boundary_count}. "
            "Provide --source_manifest and/or --full_motion_root with full long sequences."
        )
    if real_boundary_ratio < float(args.require_real_boundary_ratio):
        raise RuntimeError(
            "Real boundary ratio below strict publication threshold: "
            f"ratio={real_boundary_ratio:.4f}, required={args.require_real_boundary_ratio:.4f}."
        )

    music = np.stack(store.rows["music"]).astype(np.float32)
    meta = {
        "version": "v30_continuous_inr_real_boundary_dataset",
        "num_samples": len(store),
        "num_events": n,
        "num_source_groups": len(set(groups)),
        "max_len": args.max_len,
        "min_len": args.min_len,
        "samples_per_event": args.samples_per_event,
        "real_masks_per_boundary": args.real_masks_per_boundary,
        "actual_source_gap_count": actual_gap_count,
        "real_boundary_mask_count": real_boundary_mask_count,
        "real_boundary_total": real_boundary_total,
        "real_boundary_ratio": real_boundary_ratio,
        "synthetic_adjacent_count": synthetic_adjacent_count,
        "pseudo_requested": requested_pseudo,
        "pseudo_accepted": accepted,
        "pseudo_attempts": attempts,
        "external_prior_count": external_count,
        "music_nonzero_rate": float(np.mean(np.linalg.norm(music, axis=1) > 1e-6)),
        "sample_kind_counts": {str(k): int(v) for k, v in zip(kinds, counts)},
        "source_manifest": str(args.source_manifest),
        "full_motion_root": str(args.full_motion_root),
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
    }

    output = Path(args.out_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target=np.stack(store.rows["target"]).astype(np.float32),
        mask=np.stack(store.rows["mask"]).astype(np.float32),
        start=np.stack(store.rows["start"]).astype(np.float32),
        end=np.stack(store.rows["end"]).astype(np.float32),
        music=music,
        length=np.asarray(store.rows["length"], dtype=np.int32),
        sample_weight=np.asarray(store.rows["sample_weight"], dtype=np.float32),
        sample_kind=np.asarray(store.rows["sample_kind"], dtype=object),
        start_event_id=np.asarray(store.rows["start_event_id"], dtype=object),
        end_event_id=np.asarray(store.rows["end_event_id"], dtype=object),
        start_group=np.asarray(store.rows["start_group"], dtype=object),
        end_group=np.asarray(store.rows["end_group"], dtype=object),
        real_target=np.asarray(store.rows["real_target"], dtype=np.bool_),
        source_path=np.asarray(store.rows["source_path"], dtype=object),
        source_start_frame=np.asarray(store.rows["source_start_frame"], dtype=np.int32),
        source_end_frame=np.asarray(store.rows["source_end_frame"], dtype=np.int32),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
