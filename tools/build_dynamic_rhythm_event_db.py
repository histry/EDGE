#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build V20 Dynamic Rhythm Event-RAG database.

It replaces fixed 45-frame sliding windows with variable-length rhythm-event units.
Boundaries are inferred from motion energy valleys, root-y zero crossings, contact
switches and optional music/beat proximity.  The output is a directory of event
.pkl files plus index_dynamic_event.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from tools.v20_motion_utils import (
    MOTION_DIM,
    ROOT_X,
    ROOT_Z,
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


def boundary_candidates(motion: np.ndarray, min_len: int, max_len: int, smooth: int, min_gap: int) -> List[int]:
    T = len(motion)
    if T <= min_len:
        return [0, T]
    curves = compute_motion_curves(motion, smooth=smooth)
    energy = moving_average(curves["full"], smooth)
    energy_n = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    cand = set([0, T])

    # Motion breathing often pauses around energy valleys.
    for i in local_minima(energy_n, radius=max(2, smooth // 2)):
        cand.add(int(i))

    # Pelvis/root-y cycles are a useful proxy for phrase breathing.
    for i in zero_crossings(moving_average(curves["root_y_vel"], smooth)):
        cand.add(int(i))

    # Support/contact changes should be near boundaries, but avoid placing the boundary exactly
    # on an unstable frame; move to a nearby low-energy frame when possible.
    contact = curves["contact_switch"]
    if len(contact):
        th = float(np.percentile(contact, 80))
        for i in np.where(contact >= th)[0].tolist():
            lo = max(1, int(i) - 5)
            hi = min(T - 1, int(i) + 6)
            if hi > lo:
                j = lo + int(np.argmin(energy_n[lo:hi]))
                cand.add(j)

    # Remove candidates too close to endpoints unless needed.
    raw = sorted([c for c in cand if 0 <= c <= T])
    filtered = [0]
    for c in raw:
        if c in (0, T):
            continue
        if c < min_len or T - c < min_len:
            continue
        if c - filtered[-1] >= min_gap:
            filtered.append(c)
        elif c - filtered[-1] > 0:
            # Keep lower-energy candidate within a crowded neighborhood.
            prev = filtered[-1]
            if prev != 0 and energy_n[c] < energy_n[prev]:
                filtered[-1] = c
    if filtered[-1] != T:
        filtered.append(T)

    # Split long intervals; prefer local low energy points inside them.
    changed = True
    while changed:
        changed = False
        out = [filtered[0]]
        for a, b in zip(filtered[:-1], filtered[1:]):
            if b - a > max_len:
                target = a + (b - a) // 2
                lo = max(a + min_len, target - 10)
                hi = min(b - min_len, target + 11)
                if hi > lo:
                    c = lo + int(np.argmin(energy_n[lo:hi]))
                else:
                    c = target
                out.extend([int(c), b])
                changed = True
            else:
                out.append(b)
        filtered = sorted(set(out))

    # Merge too-short intervals.
    merged = [filtered[0]]
    for b in filtered[1:]:
        if b - merged[-1] < min_len and b != T:
            continue
        merged.append(b)
    if merged[-1] != T:
        merged.append(T)
    return merged


def build_events_for_motion(
    motion: np.ndarray,
    source_path: Path,
    out_event_dir: Path,
    min_len: int,
    max_len: int,
    ideal_len: int,
    smooth: int,
    min_gap: int,
    canonical_len: int,
    source_id: int,
) -> List[Dict]:
    motion = localize_root(motion)
    boundaries = boundary_candidates(motion, min_len=min_len, max_len=max_len, smooth=smooth, min_gap=min_gap)
    events: List[Dict] = []
    source_stem = source_path.stem.replace(" ", "_")
    for event_idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        L = int(b - a)
        if L < min_len or L > max_len + 4:
            continue
        seg = motion[a:b].copy()
        if len(seg) < min_len:
            continue
        desc = describe_motion_event(seg)
        # Length prior encourages natural mid-length units without forcing a fixed 45 frames.
        length_score = float(np.exp(-abs(L - ideal_len) / max(ideal_len, 1)))
        desc["length_score"] = length_score
        desc["quality_score"] = float(0.70 * float(desc.get("quality_score", 0.0)) + 0.30 * length_score)
        event_id = f"src{source_id:04d}_ev{event_idx:04d}_{source_stem}_{a:06d}_{b:06d}"
        pkl_path = out_event_dir / f"{event_id}.pkl"
        obj = {
            "motion": seg.astype(np.float32),
            "canonical_motion": canonical_resample_motion(seg, canonical_len).astype(np.float32),
            "length": int(L),
            "event_id": event_id,
            "source_id": int(source_id),
            "source_file": str(source_path),
            "source_start": int(a),
            "source_end": int(b),
            "entry_pose": seg[0].astype(np.float32),
            "center_pose": seg[len(seg) // 2].astype(np.float32),
            "exit_pose": seg[-1].astype(np.float32),
            "entry_velocity": velocity_at(seg, at_exit=False),
            "exit_velocity": velocity_at(seg, at_exit=True),
            "descriptor": desc,
            "event_type": desc.get("event_type", "neutral_flow"),
            "quality_score": float(desc.get("quality_score", 0.0)),
            "visual_score": float(desc.get("quality_score", 0.0)),
            "safety_score": float(desc.get("safety_score", 0.0)),
        }
        save_pkl(obj, pkl_path)
        item = {
            "event_id": event_id,
            "pkl": str(pkl_path),
            "source_id": int(source_id),
            "source_file": str(source_path),
            "source_start": int(a),
            "source_end": int(b),
            "length": int(L),
            "event_type": obj["event_type"],
            "quality_score": obj["quality_score"],
            "visual_score": obj["visual_score"],
            "safety_score": obj["safety_score"],
            "descriptor": desc,
        }
        events.append(item)
    return events


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
    ap.add_argument("--quality_top_k", type=int, default=0, help="Keep top-k by quality_score; 0 keeps all")
    args = ap.parse_args()

    out_dir = ensure_dir(args.out_dir)
    event_dir = ensure_dir(out_dir / "events")
    files = iter_motion_files(args.input_dir)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    if not files:
        raise RuntimeError(f"No motion files found under {args.input_dir}")

    all_events: List[Dict] = []
    failed: List[Dict] = []
    for source_id, f in enumerate(files):
        try:
            motion, meta = load_motion_any(f)
            events = build_events_for_motion(
                motion=motion,
                source_path=f,
                out_event_dir=event_dir,
                min_len=args.min_len,
                max_len=args.max_len,
                ideal_len=args.ideal_len,
                smooth=args.energy_smooth,
                min_gap=args.boundary_min_gap,
                canonical_len=args.save_canonical_len,
                source_id=source_id,
            )
            all_events.extend(events)
            print(f"[OK] {f} -> {len(events)} events")
        except Exception as exc:
            failed.append({"file": str(f), "error": str(exc)})
            print(f"[FAIL] {f}: {exc}")

    all_events = sorted(all_events, key=lambda x: float(x.get("quality_score", 0.0)), reverse=True)
    if args.quality_top_k > 0:
        all_events = all_events[: args.quality_top_k]

    by_type: Dict[str, int] = {}
    lengths: List[int] = []
    for e in all_events:
        by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
        lengths.append(int(e["length"]))

    index = {
        "version": "v20_dynamic_rhythm_event_rag",
        "input_dir": str(args.input_dir),
        "out_dir": str(out_dir),
        "event_dir": str(event_dir),
        "num_sources": len(files),
        "num_events": len(all_events),
        "params": vars(args),
        "event_type_counts": by_type,
        "length_stats": {
            "min": int(min(lengths)) if lengths else 0,
            "max": int(max(lengths)) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
        },
        "items": all_events,
        "failed": failed,
    }
    write_json(index, out_dir / "index_dynamic_event.json")
    if args.report:
        write_json(index, args.report)
    print("============================================================")
    print(f"saved: {out_dir / 'index_dynamic_event.json'}")
    print(f"events: {len(all_events)}")
    print(f"event_type_counts: {by_type}")
    print("============================================================")


if __name__ == "__main__":
    main()
