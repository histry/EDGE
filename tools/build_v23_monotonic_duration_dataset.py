#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build V23 supervision for explicit duration and monotonic time mapping."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from tools.v21_common import load_json_items, load_motion
from tools.v22_turn_utils import (
    detect_turn_events,
    make_fast_turn_corruption,
    motion_query_from_dynamics,
    yaw_speed_dps_np,
)


def collect_sources(motion_globs: Sequence[str], event_db: str) -> List[Path]:
    paths: Set[Path] = set()
    for pattern in motion_globs:
        for value in glob.glob(pattern, recursive=True):
            p = Path(value)
            if p.is_file() and p.suffix.lower() in {".npy", ".npz", ".pkl"}:
                paths.add(p.resolve())
    if event_db:
        _, items = load_json_items(event_db)
        for item in items:
            for key in ("source_file", "source_path", "source", "pkl", "path"):
                text = str(item.get(key, "")).strip()
                if text:
                    p = Path(text)
                    if p.is_file():
                        paths.add(p.resolve())
                        break
    return sorted(paths)


def extract_window(motion: np.ndarray, center: int, window_len: int) -> Tuple[np.ndarray, int]:
    half = window_len // 2
    start = int(center - half)
    end = start + window_len
    if start < 0:
        start, end = 0, window_len
    if end > len(motion):
        end = len(motion)
        start = max(0, end - window_len)
    window = motion[start:end]
    if len(window) < window_len:
        if len(window) == 0:
            raise ValueError("Empty motion window")
        window = np.concatenate(
            [window, np.repeat(window[-1:], window_len - len(window), axis=0)], axis=0
        )
    return window.astype(np.float32), int(start)


def inverse_time_map(source_positions: np.ndarray) -> np.ndarray:
    """Invert corrupted-output -> natural-source map into natural-output -> corrupted-input."""
    p = np.asarray(source_positions, dtype=np.float32)
    p = np.maximum.accumulate(p)
    n = len(p)
    output_frames = np.arange(n, dtype=np.float32)
    natural_frames = np.arange(n, dtype=np.float32)
    inverse = np.interp(natural_frames, p, output_frames).astype(np.float32)
    inverse[0] = 0.0
    inverse[-1] = float(n - 1)
    return np.clip(inverse / max(n - 1, 1), 0.0, 1.0).astype(np.float32)


def build_condition(
    target: np.ndarray,
    corrupted_peak_dps: float,
    target_peak_dps: float,
    turn_angle_deg: float,
    corrupted_span: int,
    turn_center_phase: float,
) -> np.ndarray:
    query = motion_query_from_dynamics(target)
    extra = np.asarray(
        [
            np.log1p(max(corrupted_peak_dps, 0.0)) / np.log1p(1600.0),
            np.log1p(max(target_peak_dps, 0.0)) / np.log1p(250.0),
            np.clip(turn_angle_deg / 180.0, 0.0, 2.0),
            np.clip(corrupted_span / max(len(target) - 1, 1), 0.0, 1.0),
            np.clip(turn_center_phase, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([query, extra], axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion_glob", action="append", default=[])
    ap.add_argument("--event_db", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window_len", type=int, default=72)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_peak_dps", type=float, default=32.0)
    ap.add_argument("--min_turn_angle_deg", type=float, default=14.0)
    ap.add_argument("--min_gap", type=int, default=20)
    ap.add_argument("--max_events_per_source", type=int, default=200)
    ap.add_argument("--augmentations_per_event", type=int, default=10)
    ap.add_argument("--min_speed_factor", type=float, default=1.15)
    ap.add_argument("--max_speed_factor", type=float, default=8.0)
    ap.add_argument("--identity_ratio", type=float, default=0.10)
    ap.add_argument("--center_jitter", type=int, default=6)
    ap.add_argument("--max_samples", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    sources = collect_sources(args.motion_glob, args.event_db)
    if not sources:
        raise RuntimeError("No source motions found")
    print("source motions:", len(sources), flush=True)

    rng = np.random.default_rng(args.seed)
    corrupted_list: List[np.ndarray] = []
    target_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    condition_list: List[np.ndarray] = []
    tau_list: List[np.ndarray] = []
    source_id_list: List[int] = []
    target_duration_list: List[float] = []
    corrupted_duration_list: List[float] = []
    turn_input_start_list: List[float] = []
    turn_input_end_list: List[float] = []
    target_peak_list: List[float] = []
    corrupted_peak_list: List[float] = []
    factor_list: List[float] = []
    metadata: List[Dict[str, object]] = []
    failed = 0

    for source_id, path in enumerate(sources):
        try:
            motion = load_motion(path)
        except Exception as exc:
            failed += 1
            print(f"[SKIP] {path}: {exc}", flush=True)
            continue
        if len(motion) < args.window_len:
            continue

        events = detect_turn_events(
            motion,
            fps=args.fps,
            min_peak_dps=args.min_peak_dps,
            threshold_ratio=0.27,
            min_gap=args.min_gap,
            min_duration=3,
            max_events=args.max_events_per_source,
        )
        events = [e for e in events if e.path_angle_deg >= args.min_turn_angle_deg]

        for event_index, event in enumerate(events):
            factor_grid = np.exp(
                rng.uniform(
                    np.log(args.min_speed_factor),
                    np.log(args.max_speed_factor),
                    size=max(1, args.augmentations_per_event),
                )
            )
            factors = factor_grid.tolist()
            if rng.random() < args.identity_ratio:
                factors.append(1.0)

            for factor in factors:
                jitter = int(rng.integers(-args.center_jitter, args.center_jitter + 1))
                target, window_start = extract_window(
                    motion, event.peak_index + jitter, args.window_len
                )
                local_start = int(np.clip(event.start - window_start, 1, args.window_len - 4))
                local_end = int(np.clip(event.end - window_start, local_start + 2, args.window_len - 2))
                target_span = int(local_end - local_start)
                target_speed = yaw_speed_dps_np(target, fps=args.fps)
                target_peak = float(target_speed.max()) if len(target_speed) else float(event.peak_speed_dps)
                center_phase = float(0.5 * (local_start + local_end) / max(args.window_len - 1, 1))

                if factor <= 1.001:
                    corrupted = target.copy()
                    source_positions = np.arange(args.window_len, dtype=np.float32)
                    corrupted_start, corrupted_end = local_start, local_end
                    corrupted_span = target_span
                    effective_factor = 1.0
                    phase = np.linspace(0.0, 1.0, args.window_len, dtype=np.float32)
                    center = 0.5 * (local_start + local_end)
                    radius = max(7.0, 0.5 * target_span + 6.0)
                    mask = np.exp(-0.5 * ((np.arange(args.window_len) - center) / radius) ** 2)
                    mask = (mask / max(mask.max(), 1e-8)).astype(np.float32)
                else:
                    corrupted, mask, info = make_fast_turn_corruption(
                        target,
                        local_start,
                        local_end,
                        speed_factor=float(factor),
                        min_context_frames=4,
                    )
                    source_positions = np.asarray(info["source_positions"], dtype=np.float32)
                    corrupted_start = int(info["corrupted_turn_start"])
                    corrupted_end = int(info["corrupted_turn_end"])
                    corrupted_span = int(info["corrupted_turn_span"])
                    effective_factor = float(info["effective_speed_factor"])

                target_tau = inverse_time_map(source_positions)
                corrupted_speed = yaw_speed_dps_np(corrupted, fps=args.fps)
                corrupted_peak = float(corrupted_speed.max()) if len(corrupted_speed) else target_peak
                condition = build_condition(
                    target,
                    corrupted_peak,
                    target_peak,
                    float(event.path_angle_deg),
                    corrupted_span,
                    center_phase,
                )

                corrupted_list.append(corrupted.astype(np.float32))
                target_list.append(target.astype(np.float32))
                mask_list.append(mask.astype(np.float32))
                condition_list.append(condition)
                tau_list.append(target_tau)
                source_id_list.append(source_id)
                target_duration_list.append(float(target_span + 1))
                corrupted_duration_list.append(float(corrupted_span + 1))
                turn_input_start_list.append(float(corrupted_start / max(args.window_len - 1, 1)))
                turn_input_end_list.append(float(corrupted_end / max(args.window_len - 1, 1)))
                target_peak_list.append(target_peak)
                corrupted_peak_list.append(corrupted_peak)
                factor_list.append(effective_factor)
                metadata.append(
                    {
                        "source": str(path),
                        "source_id": source_id,
                        "event_index": event_index,
                        "window_start": window_start,
                        "target_turn_start": local_start,
                        "target_turn_end": local_end,
                        "corrupted_turn_start": corrupted_start,
                        "corrupted_turn_end": corrupted_end,
                        "target_duration_frames": target_span + 1,
                        "corrupted_duration_frames": corrupted_span + 1,
                        "target_peak_dps": target_peak,
                        "corrupted_peak_dps": corrupted_peak,
                        "speed_factor": effective_factor,
                    }
                )
                if len(corrupted_list) >= args.max_samples:
                    break
            if len(corrupted_list) >= args.max_samples:
                break
        print(
            f"[{source_id + 1}/{len(sources)}] {path.name}: turns={len(events)} samples={len(corrupted_list)}",
            flush=True,
        )
        if len(corrupted_list) >= args.max_samples:
            break

    if not corrupted_list:
        raise RuntimeError("No V23 samples built")

    order = rng.permutation(len(corrupted_list))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        corrupted=np.stack(corrupted_list)[order].astype(np.float32),
        target=np.stack(target_list)[order].astype(np.float32),
        edit_mask=np.stack(mask_list)[order].astype(np.float32),
        condition=np.stack(condition_list)[order].astype(np.float32),
        target_tau=np.stack(tau_list)[order].astype(np.float32),
        source_id=np.asarray(source_id_list, dtype=np.int32)[order],
        target_duration_frames=np.asarray(target_duration_list, dtype=np.float32)[order],
        corrupted_duration_frames=np.asarray(corrupted_duration_list, dtype=np.float32)[order],
        turn_input_start=np.asarray(turn_input_start_list, dtype=np.float32)[order],
        turn_input_end=np.asarray(turn_input_end_list, dtype=np.float32)[order],
        target_peak_dps=np.asarray(target_peak_list, dtype=np.float32)[order],
        corrupted_peak_dps=np.asarray(corrupted_peak_list, dtype=np.float32)[order],
        speed_factor=np.asarray(factor_list, dtype=np.float32)[order],
    )
    out.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "version": "v23_monotonic_duration_dataset",
                "num_samples": len(corrupted_list),
                "num_sources": len(set(source_id_list)),
                "failed_sources": failed,
                "window_len": args.window_len,
                "condition_dim": int(condition_list[0].shape[0]),
                "samples": [metadata[i] for i in order.tolist()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=" * 72)
    print("saved:", out)
    print("samples:", len(corrupted_list))
    print("sources:", len(set(source_id_list)))
    print("duration target p10/p50/p90:", np.percentile(target_duration_list, [10, 50, 90]))
    print("duration corrupt p10/p50/p90:", np.percentile(corrupted_duration_list, [10, 50, 90]))
    print("speed factor p10/p50/p90/max:", np.percentile(factor_list, [10, 50, 90, 100]))


if __name__ == "__main__":
    main()
