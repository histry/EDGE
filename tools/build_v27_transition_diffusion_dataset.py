#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the V31 risk-controlled band-limited transition dataset.

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
    CONTACT,
    MOTION_DIM,
    ROOT_X,
    ROOT_Z,
    endpoint_metrics_np,
    make_so3_transition,
    resample_motion_so3_np,
)
from tools.v33_event_contacts import (
    LABEL_SOURCE,
    EventContactCache,
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
    """Cache complete source sequences and infer contacts once per source."""

    def __init__(self, event_contact_cache: EventContactCache) -> None:
        self.event_contact_cache = event_contact_cache
        self.cache: Dict[str, Tuple[np.ndarray, np.ndarray] | None] = {}

    def load(
        self,
        paths: Sequence[Path],
        minimum_length: int,
    ) -> Tuple[np.ndarray | None, np.ndarray | None, str]:
        for path in paths:
            key = str(path)
            if key not in self.cache:
                try:
                    if not path.is_file():
                        self.cache[key] = None
                    else:
                        original = load_motion(path).astype(np.float32)
                        hard, confidence = self.event_contact_cache.infer_sequence(original)
                        original = original.copy()
                        original[:, CONTACT] = hard
                        original[:, ROOT_X] = 0.0
                        original[:, ROOT_Z] = 0.0
                        self.cache[key] = (original, confidence.astype(np.float32))
                except Exception:
                    self.cache[key] = None
            row = self.cache[key]
            if row is not None and len(row[0]) >= int(minimum_length):
                return row[0].copy(), row[1].copy(), key
        return None, None, ""


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


def _resample_contact_confidence(confidence: np.ndarray, length: int) -> np.ndarray:
    value = np.asarray(confidence, dtype=np.float32)
    if len(value) == length:
        return value.copy()
    if len(value) == 1:
        return np.repeat(value, length, axis=0)
    old = np.linspace(0.0, 1.0, len(value), dtype=np.float32)
    new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    result = np.empty((length, 4), dtype=np.float32)
    for channel in range(4):
        result[:, channel] = np.interp(new, old, value[:, channel])
    return np.clip(result, 0.0, 1.0).astype(np.float32)


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
        contact_confidence: np.ndarray | None = None,
        start_contact_confidence: np.ndarray | None = None,
        end_contact_confidence: np.ndarray | None = None,
        contact_label_source: str = "none",
        contact_origin_id: str = "",
        contact_target_start: int = -1,
        contact_target_end_exclusive: int = -1,
    ) -> None:
        target = np.asarray(target, dtype=np.float32)
        if len(target) < 2 or len(target) > self.max_len:
            return
        padded = np.zeros((self.max_len, MOTION_DIM), dtype=np.float32)
        mask = np.zeros((self.max_len,), dtype=np.float32)
        padded[: len(target)] = target
        mask[: len(target)] = 1.0

        if contact_confidence is None:
            confidence = np.zeros((len(target), 4), dtype=np.float32)
        else:
            confidence = np.asarray(contact_confidence, dtype=np.float32)
            if confidence.shape != (len(target), 4):
                raise ValueError(
                    f"contact_confidence shape={confidence.shape}, "
                    f"expected={(len(target), 4)}"
                )
        confidence_padded = np.zeros((self.max_len, 4), dtype=np.float32)
        confidence_padded[: len(target)] = np.clip(confidence, 0.0, 1.0)
        start_conf = np.zeros((4,), dtype=np.float32) if start_contact_confidence is None else np.asarray(start_contact_confidence, dtype=np.float32).reshape(4)
        end_conf = np.zeros((4,), dtype=np.float32) if end_contact_confidence is None else np.asarray(end_contact_confidence, dtype=np.float32).reshape(4)

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
        self.rows["contact_confidence"].append(confidence_padded)
        self.rows["start_contact_confidence"].append(np.clip(start_conf, 0.0, 1.0))
        self.rows["end_contact_confidence"].append(np.clip(end_conf, 0.0, 1.0))
        self.rows["contact_label_source"].append(str(contact_label_source))
        self.rows["contact_origin_id"].append(str(contact_origin_id))
        self.rows["contact_target_start"].append(int(contact_target_start))
        self.rows["contact_target_end_exclusive"].append(int(contact_target_end_exclusive))

    def __len__(self) -> int:
        return len(self.rows["target"])


def assert_synchronised_contact_slices(store: SampleStore) -> Dict[str, int]:
    """Prove that overlapping real samples use identical source-frame labels."""
    seen: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    comparisons = 0
    overlaps = 0
    for row in range(len(store)):
        origin = str(store.rows["contact_origin_id"][row])
        start = int(store.rows["contact_target_start"][row])
        end = int(store.rows["contact_target_end_exclusive"][row])
        if not origin or start < 0 or end <= start:
            continue
        length = int(store.rows["length"][row])
        if end - start != length:
            raise AssertionError(
                f"Contact origin interval length mismatch at sample {row}: "
                f"[{start},{end}) vs length={length}"
            )
        target = np.asarray(store.rows["target"][row], np.float32)[:length, CONTACT]
        confidence = np.asarray(
            store.rows["contact_confidence"][row], np.float32
        )[:length]
        for local in range(length):
            key = (origin, start + local)
            value = (target[local], confidence[local])
            comparisons += 1
            if key in seen:
                overlaps += 1
                old_contact, old_confidence = seen[key]
                if not np.array_equal(old_contact, value[0]):
                    raise AssertionError(
                        f"Conflicting contact labels for {key}: "
                        f"{old_contact.tolist()} vs {value[0].tolist()}"
                    )
                if not np.allclose(old_confidence, value[1], atol=1e-7, rtol=0.0):
                    raise AssertionError(
                        f"Conflicting contact confidence for {key}: "
                        f"{old_confidence.tolist()} vs {value[1].tolist()}"
                    )
            else:
                seen[key] = (value[0].copy(), value[1].copy())
    return {
        "unique_origin_frames": len(seen),
        "frame_comparisons": comparisons,
        "overlap_comparisons": overlaps,
        "conflicts": 0,
    }

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
            real_target=False,
            source_path=str(prior_path),
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--event_contact_cache", required=True)
    parser.add_argument("--require_event_contacts", type=int, default=1)
    parser.add_argument("--assert_contact_consistency", type=int, default=1)
    parser.add_argument("--source_manifest", default="")
    parser.add_argument("--full_motion_root", default="")
    parser.add_argument("--external_prior_npz", default="")
    parser.add_argument("--max_len", type=int, default=120)
    parser.add_argument("--min_len", type=int, default=8)
    parser.add_argument("--samples_per_event", type=int, default=6)
    parser.add_argument("--real_masks_per_boundary", type=int, default=3)
    parser.add_argument("--source_pairs_per_event", type=float, default=1.0)
    parser.add_argument("--pseudo_pairs_per_event", type=float, default=0.0)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--max_source_gap", type=int, default=120)
    parser.add_argument("--allow_synthetic_adjacent", type=int, default=0)
    parser.add_argument("--require_real_boundary_count", type=int, default=0)
    parser.add_argument("--require_real_boundary_ratio", type=float, default=0.0)
    parser.add_argument("--require_unique_real_boundary_count", type=int, default=0)
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
    event_contact_cache = EventContactCache(args.event_contact_cache)
    if len(event_contact_cache) != len(all_items):
        raise RuntimeError(
            f"Event contact cache has {len(event_contact_cache)} entries, "
            f"index has {len(all_items)}"
        )

    motions: List[np.ndarray | None] = []
    event_contact_confidence: List[np.ndarray | None] = []
    event_contact_origin: List[str] = []
    groups: List[str] = []
    conditions: List[np.ndarray] = []
    bounds: List[Tuple[int | None, int | None]] = []
    for index, item in enumerate(items):
        path = Path(str(item.get("pkl", item.get("path", ""))))
        try:
            motion = load_motion(path).astype(np.float32)
            hard, confidence, origin = event_contact_cache.get(
                index, item, motion, strict=bool(args.require_event_contacts)
            )
            motion = motion.copy()
            motion[:, CONTACT] = hard
            motion[:, ROOT_X] = 0.0
            motion[:, ROOT_Z] = 0.0
        except Exception:
            if bool(args.require_event_contacts):
                raise
            motion = None
            confidence = None
            origin = ""
        motions.append(motion)
        event_contact_confidence.append(confidence)
        event_contact_origin.append(origin)
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
                contact_confidence=event_contact_confidence[index][left + 1 : right],
                start_contact_confidence=event_contact_confidence[index][left],
                end_contact_confidence=event_contact_confidence[index][right],
                contact_label_source=LABEL_SOURCE,
                contact_origin_id=event_contact_origin[index],
                contact_target_start=left + 1,
                contact_target_end_exclusive=right,
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
    source_cache = SourceCache(event_contact_cache)
    actual_gap_count = 0
    real_boundary_mask_count = 0
    synthetic_adjacent_count = 0
    unique_real_boundary_pairs: set[str] = set()

    for first, second in adjacent_pairs[:max_adjacent]:
        prev, nxt = motions[first], motions[second]
        if prev is None or nxt is None or len(prev) < 2 or len(nxt) < 2:
            continue
        start_a, end_a = bounds[first]
        start_b, end_b = bounds[second]
        # Only require the indexed boundary to exist. The V30 builder added
        # max_len to this check, which rejected valid full sequences near the
        # sequence tail and silently reduced real-boundary supervision.
        maximum_bound = max(
            value for value in (end_a, start_b, end_b)
            if value is not None
        ) if any(value is not None for value in (end_a, start_b, end_b)) else 0
        minimum_source_length = maximum_bound + 2
        candidates = _candidate_source_paths(
            items[first],
            manifest,
            groups[first],
            args.full_motion_root,
        )
        full_source, full_source_confidence, source_path = source_cache.load(
            candidates, minimum_length=max(minimum_source_length, 8)
        )
        condition = _normalise(0.5 * conditions[first] + 0.5 * conditions[second])
        event_a = str(items[first].get("event_id", first))
        event_b = str(items[second].get("event_id", second))

        if (
            full_source is not None
            and end_a is not None
            and start_b is not None
            and 0 <= end_a < len(full_source)
            and 0 <= start_b < len(full_source)
        ):
            pair_key = f"{groups[first]}|{event_a}|{event_b}|{end_a}|{start_b}"

            # A genuine omitted source interval exists only when there is at
            # least one frame strictly between the two indexed events.
            if end_a + 1 < start_b:
                gap = full_source[end_a + 1 : start_b]
                if args.min_len <= len(gap) <= args.max_source_gap:
                    k = min(len(gap), args.max_len)
                    store.add(
                        target=_resample(gap, k),
                        start=full_source[end_a],
                        end=full_source[start_b],
                        music=_drop_condition(
                            condition, rng, args.condition_dropout
                        ),
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
                        contact_confidence=_resample_contact_confidence(
                            full_source_confidence[end_a + 1 : start_b], k
                        ),
                        start_contact_confidence=full_source_confidence[end_a],
                        end_contact_confidence=full_source_confidence[start_b],
                        contact_label_source=LABEL_SOURCE,
                        contact_origin_id=(
                            f"source:{source_path}" if k == len(gap) else ""
                        ),
                        contact_target_start=(end_a + 1 if k == len(gap) else -1),
                        contact_target_end_exclusive=(start_b if k == len(gap) else -1),
                    )
                    actual_gap_count += 1
                    unique_real_boundary_pairs.add(pair_key)

            # Mask a real interval around the annotated boundary. This remains
            # valid for contiguous or mildly overlapping event crops.
            boundary = int(round(0.5 * (end_a + start_b)))
            for _ in range(max(0, args.real_masks_per_boundary)):
                k = int(rng.integers(args.min_len, args.max_len + 1))
                jitter = int(rng.integers(
                    -max(1, k // 8), max(2, k // 8 + 1)
                ))
                left = boundary - k // 2 - 1 + jitter
                right = left + k + 1
                if left < 0 or right >= len(full_source):
                    continue
                lower_boundary = min(end_a, start_b)
                upper_boundary = max(end_a, start_b)
                if not (
                    left < lower_boundary < right
                    or left < upper_boundary < right
                ):
                    continue
                target = full_source[left + 1 : right]
                if len(target) != k:
                    continue
                store.add(
                    target=target,
                    start=full_source[left],
                    end=full_source[right],
                    music=_drop_condition(
                        condition, rng, args.condition_dropout
                    ),
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
                    contact_confidence=full_source_confidence[left + 1 : right],
                    start_contact_confidence=full_source_confidence[left],
                    end_contact_confidence=full_source_confidence[right],
                    contact_label_source=LABEL_SOURCE,
                    contact_origin_id=f"source:{source_path}",
                    contact_target_start=left + 1,
                    contact_target_end_exclusive=right,
                )
                real_boundary_mask_count += 1
                unique_real_boundary_pairs.add(pair_key)

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

    contact_consistency = (
        assert_synchronised_contact_slices(store)
        if bool(args.assert_contact_consistency)
        else {
            "unique_origin_frames": 0,
            "frame_comparisons": 0,
            "overlap_comparisons": 0,
            "conflicts": -1,
        }
    )

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
    if len(unique_real_boundary_pairs) < int(
        args.require_unique_real_boundary_count
    ):
        raise RuntimeError(
            "Insufficient unique real boundaries: "
            f"found={len(unique_real_boundary_pairs)}, "
            f"required={args.require_unique_real_boundary_count}. "
            "Multiple masks from one boundary do not count as independent evidence."
        )

    music = np.stack(store.rows["music"]).astype(np.float32)
    meta = {
        "version": "v33_event_level_contact_transition_dataset",
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
        "unique_real_boundary_count": len(unique_real_boundary_pairs),
        "synthetic_adjacent_count": synthetic_adjacent_count,
        "pseudo_requested": requested_pseudo,
        "pseudo_accepted": accepted,
        "pseudo_attempts": attempts,
        "external_prior_count": external_count,
        "external_prior_is_real_target": False,
        "bounds_convention": "source_start inclusive; source_end treated as indexed endpoint; real gaps use end+1:start",
        "music_nonzero_rate": float(np.mean(np.linalg.norm(music, axis=1) > 1e-6)),
        "sample_kind_counts": {str(k): int(v) for k, v in zip(kinds, counts)},
        "source_manifest": str(args.source_manifest),
        "full_motion_root": str(args.full_motion_root),
        "source_index_json": str(args.index_json),
        "source_duration_index_npz": str(args.duration_index_npz),
        "event_contact_cache": str(args.event_contact_cache),
        "contact_pipeline": {
            "level": "complete_event_before_window_sampling",
            "label_source": LABEL_SOURCE,
            "synchronised_slicing": True,
            "window_level_relabeling": False,
            "consistency_assertion": contact_consistency,
            "cache_meta": event_contact_cache.meta,
        },
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
        contact_confidence=np.stack(store.rows["contact_confidence"]).astype(np.float32),
        start_contact_confidence=np.stack(store.rows["start_contact_confidence"]).astype(np.float32),
        end_contact_confidence=np.stack(store.rows["end_contact_confidence"]).astype(np.float32),
        contact_label_source=np.asarray(store.rows["contact_label_source"], dtype=object),
        contact_origin_id=np.asarray(store.rows["contact_origin_id"], dtype=object),
        contact_target_start=np.asarray(store.rows["contact_target_start"], dtype=np.int32),
        contact_target_end_exclusive=np.asarray(
            store.rows["contact_target_end_exclusive"], dtype=np.int32
        ),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
