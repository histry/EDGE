#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a source-aware dynamic Event-RAG database for low-resource Dunhuang.

Compared with the older V20 splitter, this builder moves source control into
the segmentation stage instead of applying it only after the index has already
been built.  It addresses three failure modes seen in the current database:

1. source skew: one dancer/repeat/category can contribute many more snippets;
2. static-hold bias: fast first second followed by long pose holding;
3. near-duplicate leakage: repeated takes of the same category produce almost
   identical segments that pass as different retrieval candidates.

The output keeps the existing Event-RAG pkl/json contract so downstream V21/V26
index builders and V34 schedulers can consume it without model-code changes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from tools.v20_motion_utils import (
    ROT,
    canonical_resample_motion,
    compute_motion_curves,
    describe_motion_event,
    ensure_dir,
    iter_motion_files,
    json_safe,
    load_motion_any,
    local_minima,
    localize_root,
    moving_average,
    save_pkl,
    velocity_at,
    write_json,
    zero_crossings,
)
from tools.v34_source_aware_rag import (
    SOURCE_RE,
    enrich_item_with_source_identity,
    source_aware_select,
    source_distribution,
    source_quality_score,
)


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(x, np.float32), axis=axis)


def _percentile01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32).reshape(-1)
    if len(values) == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    return np.clip((values - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0)


def _pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, np.float32).reshape(-1)
    if len(values) >= length:
        return values[:length]
    if len(values) == 0:
        return np.zeros((length,), np.float32)
    return np.pad(values, (0, length - len(values)), mode="edge")


def motion_energy_curve(motion: np.ndarray, smooth: int) -> np.ndarray:
    if len(motion) < 2:
        return np.zeros((len(motion),), np.float32)
    vel = np.diff(motion[:, ROT], axis=0)
    energy = _safe_norm(vel, axis=1)
    return _pad_to_length(moving_average(energy, max(1, int(smooth))), len(motion))


def motion_jerk_curve(motion: np.ndarray, smooth: int) -> np.ndarray:
    if len(motion) < 4:
        return np.zeros((len(motion),), np.float32)
    vel = np.diff(motion[:, ROT], axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    curve = _safe_norm(jerk, axis=1)
    return _pad_to_length(moving_average(curve, max(1, int(smooth))), len(motion))


def source_aware_boundary_candidates(
    motion: np.ndarray,
    *,
    min_len: int,
    max_len: int,
    smooth: int,
    min_gap: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Return variable-length event boundaries and diagnostic candidate stats."""
    T = int(len(motion))
    if T <= min_len:
        return [0, T], {"reason": "short_source", "num_candidates": 2}

    curves = compute_motion_curves(motion, smooth=smooth)
    energy = motion_energy_curve(motion, smooth)
    jerk = motion_jerk_curve(motion, smooth)
    energy_n = _percentile01(energy)
    jerk_n = _percentile01(jerk)
    cand = {0, T}
    reason_counter: Counter[str] = Counter()

    for i in local_minima(energy_n, radius=max(2, smooth // 2)):
        cand.add(int(i))
        reason_counter["energy_valley"] += 1

    for i in local_minima(jerk_n, radius=max(2, smooth // 2)):
        # A boundary is safer when both energy and jerk are locally low.
        if energy_n[int(i)] <= np.percentile(energy_n, 65):
            cand.add(int(i))
            reason_counter["jerk_valley"] += 1

    for i in zero_crossings(moving_average(curves["root_y_vel"], smooth)):
        cand.add(int(i))
        reason_counter["root_y_zero_crossing"] += 1

    contact = np.asarray(curves.get("contact_switch", np.zeros((T,), np.float32)), np.float32)
    if len(contact):
        threshold = float(np.percentile(contact, 82))
        for i in np.where(contact >= threshold)[0].tolist():
            lo = max(1, int(i) - 6)
            hi = min(T - 1, int(i) + 7)
            if hi > lo:
                local_cost = 0.70 * energy_n[lo:hi] + 0.30 * jerk_n[lo:hi]
                j = lo + int(np.argmin(local_cost))
                cand.add(j)
                reason_counter["contact_switch_near_valley"] += 1

    raw = sorted(c for c in cand if 0 <= c <= T)
    filtered = [0]
    for c in raw:
        if c in (0, T):
            continue
        if c < min_len or T - c < min_len:
            continue
        if c - filtered[-1] >= min_gap:
            filtered.append(c)
            continue
        prev = filtered[-1]
        if prev != 0:
            c_cost = 0.70 * energy_n[c] + 0.30 * jerk_n[c]
            p_cost = 0.70 * energy_n[prev] + 0.30 * jerk_n[prev]
            if c_cost < p_cost:
                filtered[-1] = c
    if filtered[-1] != T:
        filtered.append(T)

    changed = True
    while changed:
        changed = False
        out = [filtered[0]]
        for a, b in zip(filtered[:-1], filtered[1:]):
            if b - a > max_len:
                target = a + (b - a) // 2
                lo = max(a + min_len, target - 14)
                hi = min(b - min_len, target + 15)
                if hi > lo:
                    cost = 0.72 * energy_n[lo:hi] + 0.28 * jerk_n[lo:hi]
                    cut = lo + int(np.argmin(cost))
                else:
                    cut = target
                out.extend([int(cut), b])
                changed = True
            else:
                out.append(b)
        filtered = sorted(set(out))

    merged = [filtered[0]]
    for b in filtered[1:]:
        if b - merged[-1] < min_len and b != T:
            continue
        merged.append(b)
    if merged[-1] != T:
        merged.append(T)

    return merged, {
        "num_candidates_raw": int(len(raw)),
        "num_boundaries": int(len(merged)),
        "candidate_reasons": dict(reason_counter),
        "energy_percentiles": np.percentile(energy, [5, 25, 50, 75, 95]).round(8).tolist(),
        "jerk_percentiles": np.percentile(jerk, [5, 25, 50, 75, 95]).round(8).tolist(),
    }


def segment_motion_stats(
    motion: np.ndarray,
    a: int,
    b: int,
    *,
    ideal_len: int,
    fps: float,
    min_motion_density: float,
) -> Dict[str, float]:
    seg = motion[int(a):int(b)]
    L = int(len(seg))
    if L < 2:
        return {
            "mean_activity": 0.0,
            "tail_mean_activity": 0.0,
            "first1s_energy_ratio": 1.0,
            "motion_density_score": 0.0,
            "static_hold_score": 1.0,
            "length_score": 0.0,
            "segment_quality": 0.0,
        }

    energy = _safe_norm(np.diff(seg[:, ROT], axis=0), axis=1)
    mean_activity = float(np.mean(energy)) if len(energy) else 0.0
    tail_start = max(0, len(energy) // 2)
    tail_mean = float(np.mean(energy[tail_start:])) if len(energy[tail_start:]) else mean_activity
    first_n = max(1, min(len(energy), int(round(float(fps)))))
    total_energy = float(np.sum(energy) + 1e-8)
    first_ratio = float(np.sum(energy[:first_n]) / total_energy)
    density_score = float(np.clip(mean_activity / max(float(min_motion_density) * 3.0, 1e-8), 0.0, 1.0))

    frontload = max(0.0, first_ratio - 0.65)
    tail_deficit = max(0.0, float(min_motion_density) * 2.5 - tail_mean)
    static_hold_score = float(frontload * (tail_deficit / max(float(min_motion_density) * 2.5, 1e-8)))
    length_score = float(np.exp(-abs(float(L) - float(ideal_len)) / max(float(ideal_len), 1.0)))
    segment_quality = float(
        0.34 * density_score
        + 0.26 * length_score
        + 0.18 * np.clip(tail_mean / max(float(min_motion_density) * 3.0, 1e-8), 0.0, 1.0)
        + 0.12 * (1.0 - np.clip(first_ratio, 0.0, 1.0))
        + 0.10 * (1.0 - np.clip(static_hold_score, 0.0, 1.0))
    )
    return {
        "mean_activity": mean_activity,
        "tail_mean_activity": tail_mean,
        "first1s_energy_ratio": first_ratio,
        "motion_density_score": density_score,
        "static_hold_score": static_hold_score,
        "length_score": length_score,
        "segment_quality": segment_quality,
    }


def boundary_quality_score(motion: np.ndarray, a: int, b: int, smooth: int) -> float:
    energy = _percentile01(motion_energy_curve(motion, smooth))
    jerk = _percentile01(motion_jerk_curve(motion, smooth))

    def one(index: int) -> float:
        idx = int(np.clip(index, 0, max(len(energy) - 1, 0)))
        risk = 0.68 * float(energy[idx]) + 0.32 * float(jerk[idx])
        return float(1.0 - np.clip(risk, 0.0, 1.0))

    left = one(a)
    right = one(max(a, b - 1))
    return float(0.5 * (left + right))


def interval_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    left = max(int(a[0]), int(b[0]))
    right = min(int(a[1]), int(b[1]))
    inter = max(0, right - left)
    union = max(int(a[1]), int(b[1])) - min(int(a[0]), int(b[0]))
    return float(inter / max(union, 1))


def suppress_near_duplicates(
    events: Sequence[Dict[str, Any]],
    *,
    overlap_iou: float,
    max_per_source_before_global_cap: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in events:
        grouped[str(item.get("source_uid", item.get("source_id", "unknown")))].append(dict(item))

    kept: List[Dict[str, Any]] = []
    suppressed: List[Dict[str, Any]] = []
    for source_uid, rows in grouped.items():
        rows.sort(key=source_quality_score, reverse=True)
        source_kept: List[Dict[str, Any]] = []
        for row in rows:
            interval = (int(row.get("source_start", 0)), int(row.get("source_end", 0)))
            duplicate = False
            for prev in source_kept:
                prev_interval = (int(prev.get("source_start", 0)), int(prev.get("source_end", 0)))
                if interval_iou(interval, prev_interval) >= overlap_iou:
                    duplicate = True
                    break
            if duplicate:
                suppressed.append(
                    {
                        "event_id": str(row.get("event_id", "")),
                        "source_uid": source_uid,
                        "reason": "overlap_iou",
                    }
                )
                continue
            source_kept.append(row)
            if max_per_source_before_global_cap > 0 and len(source_kept) >= max_per_source_before_global_cap:
                break
        kept.extend(source_kept)

    return kept, {
        "near_duplicate_overlap_iou": float(overlap_iou),
        "max_per_source_before_global_cap": int(max_per_source_before_global_cap),
        "suppressed_count": int(len(suppressed)),
        "suppressed_examples": suppressed[:50],
    }


def build_events_for_motion(
    motion: np.ndarray,
    source_path: Path,
    out_event_dir: Path,
    *,
    min_len: int,
    max_len: int,
    ideal_len: int,
    smooth: int,
    min_gap: int,
    canonical_len: int,
    source_id: int,
    fps: float,
    min_motion_density: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    motion = localize_root(motion.astype(np.float32))
    boundaries, boundary_report = source_aware_boundary_candidates(
        motion,
        min_len=min_len,
        max_len=max_len,
        smooth=smooth,
        min_gap=min_gap,
    )
    source_stem = source_path.stem.replace(" ", "_")
    source_probe = enrich_item_with_source_identity({"source_file": str(source_path), "event_id": source_stem})
    events: List[Dict[str, Any]] = []
    rejected: Counter[str] = Counter()

    for event_idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        L = int(b - a)
        if L < min_len:
            rejected["too_short"] += 1
            continue
        if L > max_len + 4:
            rejected["too_long"] += 1
            continue
        seg = motion[a:b].copy()
        if len(seg) < min_len:
            rejected["empty_after_crop"] += 1
            continue

        desc = describe_motion_event(seg)
        stats = segment_motion_stats(
            motion,
            a,
            b,
            ideal_len=ideal_len,
            fps=fps,
            min_motion_density=min_motion_density,
        )
        bq = boundary_quality_score(motion, a, b, smooth=smooth)
        base_quality = float(desc.get("quality_score", 0.0))
        safety = float(desc.get("safety_score", base_quality))
        event_type = str(desc.get("event_type", "neutral_flow"))
        if event_type == "pose_hold" and stats["static_hold_score"] > 0.35:
            safety *= 0.86
        quality = float(
            0.40 * base_quality
            + 0.24 * stats["segment_quality"]
            + 0.20 * bq
            + 0.10 * stats["length_score"]
            + 0.06 * safety
        )

        desc.update(stats)
        desc["boundary_quality"] = float(bq)
        desc["quality_score"] = float(quality)
        desc["safety_score"] = float(safety)
        desc["source_aware_quality_score"] = float(quality)

        event_id = f"src{source_id:04d}_ev{event_idx:04d}_{source_stem}_{a:06d}_{b:06d}"
        pkl_path = out_event_dir / f"{event_id}.pkl"
        common = {
            "event_id": event_id,
            "pkl": str(pkl_path),
            "path": str(pkl_path),
            "source_id": int(source_id),
            "source_file": str(source_path),
            "source_start": int(a),
            "source_end": int(b),
            "length": int(L),
            "event_type": event_type,
            "quality_score": float(quality),
            "visual_score": float(quality),
            "safety_score": float(safety),
            "descriptor": desc,
            "boundary_quality": float(bq),
            "segment_quality": float(stats["segment_quality"]),
            "mean_activity": float(stats["mean_activity"]),
            "tail_mean_activity": float(stats["tail_mean_activity"]),
            "first1s_energy_ratio": float(stats["first1s_energy_ratio"]),
            "static_hold_score": float(stats["static_hold_score"]),
        }
        common.update(
            {
                key: source_probe[key]
                for key in (
                    "dancer_id",
                    "repeat_id",
                    "category_id",
                    "source_uid",
                    "dancer_category_group",
                    "category_repeat_group",
                    "source_identity_parsed",
                )
            }
        )
        obj = dict(common)
        obj.update(
            {
                "motion": seg.astype(np.float32),
                "canonical_motion": canonical_resample_motion(seg, canonical_len).astype(np.float32),
                "entry_pose": seg[0].astype(np.float32),
                "center_pose": seg[len(seg) // 2].astype(np.float32),
                "exit_pose": seg[-1].astype(np.float32),
                "entry_velocity": velocity_at(seg, at_exit=False),
                "exit_velocity": velocity_at(seg, at_exit=True),
            }
        )
        save_pkl(obj, pkl_path)
        events.append(common)

    report = dict(boundary_report)
    report["rejected"] = dict(rejected)
    report["num_events"] = int(len(events))
    return events, report


def _distribution_by_key(items: Sequence[Mapping[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(x.get(key, "unknown")) for x in items))


def is_original_source_motion(path: Path) -> bool:
    """Reject already-cut event files such as src0003_ev0371_...pkl."""
    stem = path.stem
    if stem.startswith("src") and "_ev" in stem:
        return False
    if any(part.lower() == "events" for part in path.parts):
        return False
    return SOURCE_RE.search(str(path)) is not None


def build_database(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    event_dir = ensure_dir(out_dir / "events")
    discovered = iter_motion_files(args.input_dir)
    source_files = [path for path in discovered if is_original_source_motion(path)]
    if source_files:
        files = source_files
    elif int(args.allow_event_files):
        files = discovered
    else:
        raise RuntimeError(
            "No original source motion files were found. V35 expects files named "
            "like dyl_002_Take_003.pkl/.npy/.npz, not already-cut srcXXXX_evXXXX events. "
            "Set --allow_event_files 1 only for debugging."
        )
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No motion files found under {args.input_dir}")

    all_events: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    source_reports: List[Dict[str, Any]] = []
    for source_id, path in enumerate(files):
        try:
            motion, meta = load_motion_any(path)
            events, report = build_events_for_motion(
                motion=motion,
                source_path=path,
                out_event_dir=event_dir,
                min_len=args.min_len,
                max_len=args.max_len,
                ideal_len=args.ideal_len,
                smooth=args.energy_smooth,
                min_gap=args.boundary_min_gap,
                canonical_len=args.save_canonical_len,
                source_id=source_id,
                fps=args.fps,
                min_motion_density=args.min_motion_density,
            )
            all_events.extend(events)
            source_reports.append(
                {
                    "source_id": int(source_id),
                    "source_file": str(path),
                    "num_events": int(len(events)),
                    "report": report,
                }
            )
            print(f"[OK] {path} -> {len(events)} events", flush=True)
        except Exception as exc:
            failed.append({"file": str(path), "error": repr(exc)})
            print(f"[FAIL] {path}: {exc}", flush=True)

    before_dedup = list(all_events)
    deduped, dedup_report = suppress_near_duplicates(
        all_events,
        overlap_iou=float(args.near_duplicate_iou),
        max_per_source_before_global_cap=int(args.max_per_source_before_global_cap),
    )
    selected, source_report = source_aware_select(
        deduped,
        cap_per_source_uid=int(args.cap_per_source_uid),
        category_cap_factor=float(args.category_cap_factor),
        repeat_cap_factor=float(args.repeat_cap_factor),
        dancer_cap_factor=float(args.dancer_cap_factor),
        max_events=int(args.max_events),
    )
    selected.sort(key=source_quality_score, reverse=True)

    if args.quality_top_k > 0:
        selected = selected[: int(args.quality_top_k)]

    by_type = _distribution_by_key(selected, "event_type")
    lengths = [int(x.get("length", 0)) for x in selected]
    index = {
        "version": "v35_source_aware_dynamic_event_rag",
        "input_dir": str(args.input_dir),
        "out_dir": str(out_dir),
        "event_dir": str(event_dir),
        "num_discovered_motion_files": int(len(discovered)),
        "num_original_source_files": int(len(source_files)),
        "num_sources": int(len(files)),
        "num_events_raw": int(len(before_dedup)),
        "num_events_after_dedup": int(len(deduped)),
        "num_events": int(len(selected)),
        "params": vars(args),
        "event_type_counts": by_type,
        "length_stats": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "source_distribution_raw": source_distribution(before_dedup),
        "source_distribution_selected": source_distribution(selected),
        "source_aware_selection": source_report,
        "near_duplicate_suppression": dedup_report,
        "source_reports": source_reports,
        "items": selected,
        "failed": failed,
    }

    json_path = out_dir / "index_dynamic_event_source_aware.json"
    write_json(json_safe(index), json_path)
    print("=" * 72)
    print(f"saved_json: {json_path}")
    print(f"events_raw: {len(before_dedup)}")
    print(f"events_after_dedup: {len(deduped)}")
    print(f"events_selected: {len(selected)}")
    print(f"event_types: {by_type}")
    return index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing .npy/.npz/.pkl 151D motion files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--min_len", type=int, default=24)
    ap.add_argument("--ideal_len", type=int, default=48)
    ap.add_argument("--max_len", type=int, default=72)
    ap.add_argument("--boundary_min_gap", type=int, default=18)
    ap.add_argument("--energy_smooth", type=int, default=7)
    ap.add_argument("--save_canonical_len", type=int, default=48)
    ap.add_argument("--limit_files", type=int, default=0)
    ap.add_argument("--quality_top_k", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_motion_density", type=float, default=0.0045)
    ap.add_argument("--near_duplicate_iou", type=float, default=0.72)
    ap.add_argument("--max_per_source_before_global_cap", type=int, default=160)
    ap.add_argument("--cap_per_source_uid", type=int, default=96)
    ap.add_argument("--category_cap_factor", type=float, default=1.35)
    ap.add_argument("--repeat_cap_factor", type=float, default=1.55)
    ap.add_argument("--dancer_cap_factor", type=float, default=1.45)
    ap.add_argument("--max_events", type=int, default=0)
    ap.add_argument("--allow_event_files", type=int, default=0)
    args = ap.parse_args()

    index = build_database(args)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(json_safe(index), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_report: {report_path}")


if __name__ == "__main__":
    main()
