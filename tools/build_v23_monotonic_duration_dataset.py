#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V23-v2.1 natural-duration supervision.

This builder fixes three failure modes observed in V23-v2:
1. short root-yaw cores were clamped to one minimum duration;
2. an online equal-quota collector discarded most usable records when long bins
   were sparse;
3. fixed bins did not adapt to the actual Dunhuang turn-duration distribution.

The new pipeline is two-pass:
- pass 1 detects complete full-body turn phrases and builds adaptive duration bins;
- pass 2 allocates exact per-bin sample counts and generates diverse corruption /
  identity variants without dropping the dominant short-duration population.
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from tools.v21_common import load_json_items, load_motion
from tools.v22_turn_utils import yaw_speed_dps_np
from tools.v23_duration_utils import (
    NaturalTurnEvent,
    build_v23_condition,
    detect_natural_turn_events,
    duration_bin_ids,
    extract_window_with_event,
    inverse_time_map,
    make_fast_turn_corruption_v2,
    make_soft_event_mask,
    parse_duration_bins,
)


@dataclass(frozen=True)
class EventSpec:
    source_id: int
    source_path: Path
    event_index: int
    event: NaturalTurnEvent
    duration_bin: int


def collect_sources(motion_globs: Sequence[str], event_db: str) -> List[Path]:
    paths: Set[Path] = set()
    for pattern in motion_globs:
        for value in glob.glob(pattern, recursive=True):
            path = Path(value)
            if path.is_file() and path.suffix.lower() in {".npy", ".npz", ".pkl"}:
                paths.add(path.resolve())
    if event_db:
        _, items = load_json_items(event_db)
        for item in items:
            for key in ("source_file", "source_path", "source", "pkl", "path"):
                text = str(item.get(key, "")).strip()
                if not text:
                    continue
                path = Path(text)
                if path.is_file():
                    paths.add(path.resolve())
                    break
    return sorted(paths)


def auto_duration_edges(durations: np.ndarray, requested_bins: int) -> np.ndarray:
    """Create value-balanced edges from unique observed natural durations."""
    unique = np.unique(np.asarray(durations, dtype=np.int32))
    if len(unique) < 2:
        raise RuntimeError(f"Natural-duration labels collapsed to one value: {unique.tolist()}")
    num_bins = int(np.clip(requested_bins, 2, len(unique)))
    boundaries: List[int] = []
    for i in range(num_bins):
        index = int(np.floor(i * len(unique) / num_bins))
        boundaries.append(int(unique[min(index, len(unique) - 1)]))
    boundaries = sorted(set(boundaries))
    boundaries.append(int(unique[-1]) + 1)
    edges = np.asarray(boundaries, dtype=np.int32)
    if len(edges) < 3:
        edges = np.asarray([int(unique[0]), int(unique[-1]), int(unique[-1]) + 1], dtype=np.int32)
    return edges


def parse_or_make_edges(specification: str, durations: np.ndarray) -> np.ndarray:
    text = str(specification).strip().lower()
    if text.startswith("auto"):
        pieces = text.split(":", 1)
        num_bins = int(pieces[1]) if len(pieces) == 2 and pieces[1] else 6
        return auto_duration_edges(durations, num_bins)
    return parse_duration_bins(specification)


def capped_distribution(weights: np.ndarray, maximum_fraction: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(weights > 0.0, weights, 0.0)
    if weights.sum() <= 0.0:
        raise RuntimeError("No non-empty duration bins")
    weights = weights / weights.sum()
    cap = float(np.clip(maximum_fraction, 0.20, 0.90))
    for _ in range(32):
        over = weights > cap + 1e-12
        if not np.any(over):
            break
        excess = float(np.sum(weights[over] - cap))
        weights[over] = cap
        under = (~over) & (weights > 0.0)
        if not np.any(under):
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    return weights / weights.sum()


def allocate_integer_counts(total: int, weights: np.ndarray) -> np.ndarray:
    raw = np.asarray(weights, dtype=np.float64) * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        for index in order[:remainder]:
            counts[int(index)] += 1
    return counts


def choose_factor(
    rng: np.random.Generator,
    ordinal: int,
    minimum: float,
    maximum: float,
) -> float:
    anchors = np.asarray([1.15, 1.25, 1.40, 1.60, 1.90, 2.20, 2.60, 3.00], dtype=np.float32)
    anchors = np.clip(anchors, minimum, maximum)
    if ordinal % 3 == 0:
        return float(anchors[(ordinal // 3) % len(anchors)])
    return float(np.exp(rng.uniform(np.log(minimum), np.log(maximum))))


def make_record(
    *,
    target: np.ndarray,
    local_start: int,
    local_end: int,
    source_id: int,
    source_path: Path,
    event_index: int,
    window_start: int,
    factor: float,
    is_identity: bool,
    duration_bin: int,
    fps: float,
    mask_context: int,
    min_corrupted_duration: int,
    max_effective_factor: float,
) -> Dict[str, Any]:
    window_len = len(target)
    target_duration = int(local_end - local_start + 1)

    if is_identity:
        corrupted = target.copy()
        source_positions = np.arange(window_len, dtype=np.float32)
        corrupted_start = int(local_start)
        corrupted_end = int(local_end)
        corrupted_duration = int(target_duration)
        effective_factor = 1.0
        edit_mask = make_soft_event_mask(window_len, local_start, local_end, context=mask_context)
    else:
        corrupted, edit_mask, info = make_fast_turn_corruption_v2(
            target,
            local_start,
            local_end,
            speed_factor=float(factor),
            min_context_frames=max(4, mask_context // 2),
            min_corrupted_duration=min_corrupted_duration,
            max_effective_factor=max_effective_factor,
        )
        source_positions = np.asarray(info["source_positions"], dtype=np.float32)
        corrupted_start = int(info["corrupted_turn_start"])
        corrupted_end = int(info["corrupted_turn_end"])
        corrupted_duration = int(info["corrupted_turn_duration_frames"])
        effective_factor = float(info["effective_speed_factor"])
        edit_mask = make_soft_event_mask(window_len, corrupted_start, corrupted_end, context=mask_context)

    target_tau = inverse_time_map(source_positions)
    condition = build_v23_condition(corrupted, corrupted_start, corrupted_end, fps=fps)
    target_speed = yaw_speed_dps_np(target, fps=fps, smooth_window=5)
    corrupted_speed = yaw_speed_dps_np(corrupted, fps=fps, smooth_window=5)

    return {
        "corrupted": corrupted.astype(np.float32),
        "target": target.astype(np.float32),
        "edit_mask": edit_mask.astype(np.float32),
        "condition": condition.astype(np.float32),
        "target_tau": target_tau.astype(np.float32),
        "source_id": int(source_id),
        "target_duration_frames": float(target_duration),
        "corrupted_duration_frames": float(corrupted_duration),
        "turn_input_start": float(corrupted_start / max(window_len - 1, 1)),
        "turn_input_end": float(corrupted_end / max(window_len - 1, 1)),
        "target_peak_dps": float(target_speed.max()) if len(target_speed) else 0.0,
        "corrupted_peak_dps": float(corrupted_speed.max()) if len(corrupted_speed) else 0.0,
        "speed_factor": float(effective_factor),
        "is_identity": float(is_identity),
        "duration_bin": int(duration_bin),
        "metadata": {
            "source": str(source_path),
            "source_id": int(source_id),
            "event_index": int(event_index),
            "window_start": int(window_start),
            "target_turn_start": int(local_start),
            "target_turn_end": int(local_end),
            "corrupted_turn_start": int(corrupted_start),
            "corrupted_turn_end": int(corrupted_end),
            "target_duration_frames": int(target_duration),
            "corrupted_duration_frames": int(corrupted_duration),
            "target_peak_dps": float(target_speed.max()) if len(target_speed) else 0.0,
            "corrupted_peak_dps": float(corrupted_speed.max()) if len(corrupted_speed) else 0.0,
            "requested_speed_factor": float(factor),
            "effective_speed_factor": float(effective_factor),
            "is_identity": bool(is_identity),
            "duration_bin": int(duration_bin),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_glob", action="append", default=[])
    parser.add_argument("--event_db", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window_len", type=int, default=72)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min_peak_dps", type=float, default=32.0)
    parser.add_argument("--min_turn_angle_deg", type=float, default=10.0)
    parser.add_argument("--min_gap", type=int, default=18)
    parser.add_argument("--min_target_duration", type=int, default=10)
    parser.add_argument("--max_target_duration", type=int, default=56)
    parser.add_argument("--turn_threshold_ratio", type=float, default=0.12)
    parser.add_argument("--activity_threshold_ratio", type=float, default=0.18)
    parser.add_argument("--boundary_yaw_ratio", type=float, default=0.06)
    parser.add_argument("--quiet_run", type=int, default=4)
    parser.add_argument("--opposite_run", type=int, default=3)
    parser.add_argument("--phrase_margin", type=int, default=3)
    parser.add_argument("--slow_pose_span", type=int, default=10)
    parser.add_argument("--slow_angle_window", type=int, default=24)
    parser.add_argument("--search_duration_multiplier", type=float, default=1.80)
    parser.add_argument("--split_valley_radius", type=int, default=3)
    parser.add_argument("--reversal_angle_deg", type=float, default=7.0)
    parser.add_argument("--secondary_peak_ratio", type=float, default=0.48)
    parser.add_argument("--split_score_threshold", type=float, default=0.68)
    parser.add_argument("--long_split_score_threshold", type=float, default=0.42)
    parser.add_argument("--min_direction_consistency", type=float, default=0.18)
    parser.add_argument("--cumulative_low", type=float, default=0.02)
    parser.add_argument("--cumulative_high", type=float, default=0.98)
    parser.add_argument("--max_events_per_source", type=int, default=200)
    parser.add_argument("--augmentations_per_event", type=int, default=16, help="Compatibility / diversity hint")
    parser.add_argument("--min_speed_factor", type=float, default=1.15)
    parser.add_argument("--max_speed_factor", type=float, default=3.0)
    parser.add_argument("--identity_fraction", type=float, default=0.25)
    parser.add_argument("--center_jitter", type=int, default=8)
    parser.add_argument("--mask_context", type=int, default=6)
    parser.add_argument("--min_corrupted_duration", type=int, default=4)
    parser.add_argument("--duration_bins", default="auto:6")
    parser.add_argument("--balance_power", type=float, default=0.35)
    parser.add_argument("--max_bin_fraction", type=float, default=0.45)
    parser.add_argument("--max_samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    if not (0.0 < args.identity_fraction < 0.8):
        raise ValueError("identity_fraction must be in (0,0.8)")
    if args.max_target_duration >= args.window_len - 4:
        raise ValueError("max_target_duration must leave context inside the fixed window")
    if args.min_target_duration >= args.max_target_duration:
        raise ValueError("min_target_duration must be smaller than max_target_duration")
    if args.max_speed_factor <= 1.0 or args.min_speed_factor <= 1.0:
        raise ValueError("speed factors must be > 1 for non-identity corruption")

    sources = collect_sources(args.motion_glob, args.event_db)
    if not sources:
        raise RuntimeError("No source motions found")
    print("source motions:", len(sources), flush=True)

    # Pass 1: detect complete natural events and retain only metadata.
    raw_specs: List[Tuple[int, Path, int, NaturalTurnEvent]] = []
    failed_sources = 0
    source_event_counts: Dict[str, int] = {}
    for source_id, path in enumerate(sources):
        try:
            motion = load_motion(path)
        except Exception as exc:
            failed_sources += 1
            print(f"[SKIP] {path}: {exc}", flush=True)
            continue
        if len(motion) < args.window_len:
            source_event_counts[path.name] = 0
            continue
        events = detect_natural_turn_events(
            motion,
            fps=args.fps,
            min_peak_dps=args.min_peak_dps,
            min_turn_angle_deg=args.min_turn_angle_deg,
            min_gap=args.min_gap,
            min_duration=args.min_target_duration,
            max_duration=args.max_target_duration,
            threshold_ratio=args.turn_threshold_ratio,
            cumulative_low=args.cumulative_low,
            cumulative_high=args.cumulative_high,
            max_events=args.max_events_per_source,
            activity_threshold_ratio=args.activity_threshold_ratio,
            boundary_yaw_ratio=args.boundary_yaw_ratio,
            quiet_run=args.quiet_run,
            opposite_run=args.opposite_run,
            phrase_margin=args.phrase_margin,
            slow_pose_span=args.slow_pose_span,
            slow_angle_window=args.slow_angle_window,
            search_duration_multiplier=args.search_duration_multiplier,
            split_valley_radius=args.split_valley_radius,
            reversal_angle_deg=args.reversal_angle_deg,
            secondary_peak_ratio=args.secondary_peak_ratio,
            split_score_threshold=args.split_score_threshold,
            long_split_score_threshold=args.long_split_score_threshold,
            min_direction_consistency=args.min_direction_consistency,
        )
        source_event_counts[path.name] = len(events)
        for event_index, event in enumerate(events):
            raw_specs.append((source_id, path, event_index, event))
        print(f"[{source_id + 1}/{len(sources)}] {path.name}: events={len(events)}", flush=True)

    if not raw_specs:
        raise RuntimeError("No natural turn events detected")
    raw_durations = np.asarray([item[3].duration_frames for item in raw_specs], dtype=np.int32)
    duration_edges = parse_or_make_edges(args.duration_bins, raw_durations)
    raw_bin_ids = duration_bin_ids(raw_durations, duration_edges)
    specs: List[EventSpec] = []
    for item, bin_id in zip(raw_specs, raw_bin_ids.tolist()):
        specs.append(EventSpec(item[0], item[1], item[2], item[3], int(bin_id)))

    num_bins = len(duration_edges) - 1
    event_counts = np.bincount(raw_bin_ids, minlength=num_bins).astype(np.int64)
    nonempty = event_counts > 0
    if int(nonempty.sum()) < 4:
        raise RuntimeError(
            f"Only {int(nonempty.sum())} non-empty duration bins after full-body detection: "
            f"edges={duration_edges.tolist()} counts={event_counts.tolist()}"
        )

    weights = np.zeros((num_bins,), dtype=np.float64)
    weights[nonempty] = np.power(event_counts[nonempty].astype(np.float64), float(args.balance_power))
    weights = capped_distribution(weights, args.max_bin_fraction)
    total_identity = int(round(args.max_samples * args.identity_fraction))
    total_nonidentity = int(args.max_samples - total_identity)
    identity_counts = allocate_integer_counts(total_identity, weights)
    nonidentity_counts = allocate_integer_counts(total_nonidentity, weights)

    print("=" * 80)
    print("raw events:", len(specs))
    print("raw natural duration p0/p10/p25/p50/p75/p90/p100:", np.percentile(raw_durations, [0,10,25,50,75,90,100]))
    print("adaptive duration edges:", duration_edges.tolist())
    print("raw event bin counts:", event_counts.tolist())
    print("raw events at max duration:", int(np.sum(raw_durations == int(args.max_target_duration))))
    print("raw max-duration fraction:", float(np.mean(raw_durations == int(args.max_target_duration))))
    print("selected non-identity counts:", nonidentity_counts.tolist())
    print("selected identity counts:", identity_counts.tolist())
    print("=" * 80)

    specs_by_bin: Dict[int, List[EventSpec]] = {bin_id: [] for bin_id in range(num_bins)}
    for spec in specs:
        specs_by_bin[spec.duration_bin].append(spec)

    rng = np.random.default_rng(args.seed)
    assignments_by_source: Dict[int, List[Tuple[EventSpec, bool, int]]] = {}
    for bin_id in range(num_bins):
        candidates = specs_by_bin[bin_id]
        if not candidates:
            continue
        for is_identity, count in ((False, int(nonidentity_counts[bin_id])), (True, int(identity_counts[bin_id]))):
            order = rng.permutation(len(candidates))
            shuffled = [candidates[int(index)] for index in order]
            for ordinal in range(count):
                spec = shuffled[ordinal % len(shuffled)]
                assignments_by_source.setdefault(spec.source_id, []).append((spec, is_identity, ordinal))

    # Pass 2: load each source once and materialize the exact selected dataset.
    records: List[Dict[str, Any]] = []
    for source_id, assignments in sorted(assignments_by_source.items()):
        path = sources[source_id]
        motion = load_motion(path)
        rng.shuffle(assignments)
        for spec, is_identity, ordinal in assignments:
            jitter = int(rng.integers(-args.center_jitter, args.center_jitter + 1))
            target, window_start, local_start, local_end = extract_window_with_event(
                motion,
                spec.event,
                args.window_len,
                center_jitter=jitter,
            )
            factor = 1.0 if is_identity else choose_factor(
                rng,
                ordinal,
                minimum=args.min_speed_factor,
                maximum=args.max_speed_factor,
            )
            record = make_record(
                target=target,
                local_start=local_start,
                local_end=local_end,
                source_id=source_id,
                source_path=path,
                event_index=spec.event_index,
                window_start=window_start,
                factor=factor,
                is_identity=is_identity,
                duration_bin=spec.duration_bin,
                fps=args.fps,
                mask_context=args.mask_context,
                min_corrupted_duration=args.min_corrupted_duration,
                max_effective_factor=args.max_speed_factor,
            )
            records.append(record)
        print(f"materialized {path.name}: total={len(records)}", flush=True)

    if len(records) != int(args.max_samples):
        raise RuntimeError(f"Expected {args.max_samples} records, built {len(records)}")
    rng.shuffle(records)

    arrays: Dict[str, np.ndarray] = {}
    stack_keys = ("corrupted", "target", "edit_mask", "condition", "target_tau")
    scalar_keys = (
        "source_id",
        "target_duration_frames",
        "corrupted_duration_frames",
        "turn_input_start",
        "turn_input_end",
        "target_peak_dps",
        "corrupted_peak_dps",
        "speed_factor",
        "is_identity",
        "duration_bin",
    )
    for key in stack_keys:
        arrays[key] = np.stack([record[key] for record in records]).astype(np.float32)
    for key in scalar_keys:
        dtype = np.int32 if key in {"source_id", "duration_bin"} else np.float32
        arrays[key] = np.asarray([record[key] for record in records], dtype=dtype)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays["duration_edges"] = duration_edges.astype(np.float32)
    np.savez_compressed(out, **arrays)

    target_duration = arrays["target_duration_frames"]
    corrupted_duration = arrays["corrupted_duration_frames"]
    speed_factor = arrays["speed_factor"]
    identity = arrays["is_identity"] > 0.5
    selected_bin_counts = np.bincount(arrays["duration_bin"], minlength=num_bins)
    metadata = {
        "version": "v23_v2_3_slow_aware_two_stage",
        "num_samples": len(records),
        "num_sources": int(len(np.unique(arrays["source_id"]))),
        "failed_sources": int(failed_sources),
        "detected_events": int(len(specs)),
        "window_len": int(args.window_len),
        "condition_dim": int(arrays["condition"].shape[1]),
        "duration_edges": duration_edges.tolist(),
        "raw_event_duration_percentiles": np.percentile(raw_durations, [0,10,25,50,75,90,100]).tolist(),
        "raw_event_bin_counts": event_counts.tolist(),
        "raw_events_at_max_duration": int(np.sum(raw_durations == int(args.max_target_duration))),
        "raw_max_duration_fraction": float(np.mean(raw_durations == int(args.max_target_duration))),
        "selected_duration_bin_counts": selected_bin_counts.tolist(),
        "identity_fraction": float(identity.mean()),
        "target_duration_percentiles": np.percentile(target_duration, [0,10,25,50,75,90,100]).tolist(),
        "corrupted_duration_percentiles": np.percentile(corrupted_duration, [0,10,25,50,75,90,100]).tolist(),
        "speed_factor_percentiles": np.percentile(speed_factor, [0,10,25,50,75,90,100]).tolist(),
        "source_event_counts": source_event_counts,
        "config": vars(args),
        "samples": [record["metadata"] for record in records],
    }
    out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=" * 80)
    print("saved:", out)
    print("samples:", len(records))
    print("sources:", len(np.unique(arrays["source_id"])))
    print("identity ratio:", float(identity.mean()))
    print("duration edges:", duration_edges.tolist())
    print("duration bin counts:", selected_bin_counts.tolist())
    print("target duration p0/p10/p25/p50/p75/p90/p100:", np.percentile(target_duration, [0,10,25,50,75,90,100]))
    print("corrupt duration p0/p10/p25/p50/p75/p90/p100:", np.percentile(corrupted_duration, [0,10,25,50,75,90,100]))
    print("factor p0/p10/p25/p50/p75/p90/p100:", np.percentile(speed_factor, [0,10,25,50,75,90,100]))


if __name__ == "__main__":
    main()
