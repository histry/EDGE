#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely apply V23-v2.4 to V21/EDGE 151D motions."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from model.v23_monotonic_duration import load_v23_checkpoint, warp_motion_so3
from tools.v22_turn_utils import yaw_speed_dps_np
from tools.v23_duration_utils import (
    blend_motion_so3_np,
    build_v23_condition,
    cosine_window,
    detect_natural_turn_events,
    extract_window_with_event,
    make_soft_event_mask,
    max_rotation_jump_np,
    rotation_activity_np,
    rotation_range_np,
)


def load_motion_file(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.lib.npyio.NpzFile):
        for key in ("motion", "motion_151", "poses", "arr_0"):
            if key in value.files:
                value = value[key]
                break
        else:
            raise ValueError(f"No motion array in {path}")
    motion = np.asarray(value, dtype=np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape} from {path}")
    return motion


def contact_debounce(motion: np.ndarray, threshold: float = 0.5, min_run: int = 3) -> np.ndarray:
    output = np.asarray(motion, dtype=np.float32).copy()
    binary = output[:, :4] >= float(threshold)
    for channel in range(4):
        values = binary[:, channel].copy()
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            if end - start < int(min_run):
                left = values[start - 1] if start > 0 else None
                right = values[end] if end < len(values) else None
                if left is not None and right is not None and left == right:
                    values[start:end] = left
                elif left is not None:
                    values[start:end] = left
                elif right is not None:
                    values[start:end] = right
            start = end
        binary[:, channel] = values
    output[:, :4] = binary.astype(np.float32)
    return output


def percentile_peak(motion: np.ndarray, percentile: float = 95.0) -> float:
    speed = yaw_speed_dps_np(motion, fps=30.0, smooth_window=5)
    return float(np.percentile(np.abs(speed), percentile)) if len(speed) else 0.0


def apply_one(
    motion: np.ndarray,
    bundle: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    output = np.asarray(motion, dtype=np.float32).copy()
    model = bundle["model"]
    config = bundle["config"]
    window_len = int(config.get("window_len", 120))
    max_natural_duration = int(round(float(config.get("duration_edges", [12, 89])[-1]) - 1.0))

    detected = detect_natural_turn_events(
        output,
        fps=30.0,
        min_peak_dps=args.detect_peak_dps,
        min_turn_angle_deg=args.min_turn_angle_deg,
        min_gap=args.min_gap,
        min_duration=args.detect_min_duration,
        max_duration=min(args.detect_max_duration or max_natural_duration, window_len - 8),
        threshold_ratio=args.detect_threshold_ratio,
        cumulative_low=args.cumulative_low,
        cumulative_high=args.cumulative_high,
        max_events=args.max_events,
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
    detected = sorted(detected, key=lambda event: event.peak_speed_dps, reverse=True)
    occupied: List[Tuple[int, int]] = []
    reports: List[Dict[str, Any]] = []

    for event in detected:
        if event.peak_speed_dps < args.apply_peak_dps:
            continue
        window, window_start, local_start, local_end = extract_window_with_event(
            output, event, window_len, center_jitter=0
        )
        window_end = window_start + window_len - 1
        if any(max(window_start, lo) <= min(window_end, hi) for lo, hi in occupied):
            reports.append({"event": event.to_dict(), "accepted": False, "reasons": ["overlap"]})
            continue

        mask = make_soft_event_mask(window_len, local_start, local_end, context=args.mask_context)
        condition = build_v23_condition(window, local_start, local_end, fps=30.0)
        x = torch.from_numpy(window[None]).to(device)
        m = torch.from_numpy(mask[None]).to(device)
        c = torch.from_numpy(condition[None]).to(device)
        with torch.no_grad():
            result = model(x, m, c, use_hard_duration=False)
            candidate = warp_motion_so3(x, result["tau"])[0].cpu().numpy().astype(np.float32)

        predicted_duration = float(result["duration_frames"][0].item())
        edit_probability = float(result["edit_probability"][0].item())
        bin_confidence = float(result["duration_bin_confidence"][0].item())
        predicted_bin = int(result["duration_bin_index"][0].item())
        observed_duration = float(event.duration_frames)
        expansion_ratio = predicted_duration / max(observed_duration, 1.0)

        original_peak = percentile_peak(window)
        candidate_peak = percentile_peak(candidate)
        original_activity = rotation_activity_np(window)
        candidate_activity = rotation_activity_np(candidate)
        original_range = rotation_range_np(window)
        candidate_range = rotation_range_np(candidate)
        original_jump = max_rotation_jump_np(window)
        candidate_jump = max_rotation_jump_np(candidate)
        activity_ratio = candidate_activity / max(original_activity, 1e-8)
        range_ratio = candidate_range / max(original_range, 1e-8)
        jump_ratio = candidate_jump / max(original_jump, 1e-8)

        reasons: List[str] = []
        if edit_probability < args.min_edit_probability:
            reasons.append("low_edit_probability")
        if bin_confidence < args.min_duration_bin_confidence:
            reasons.append("low_duration_bin_confidence")
        if expansion_ratio < args.min_expansion_ratio:
            reasons.append("insufficient_expansion")
        if activity_ratio < args.min_activity_ratio:
            reasons.append("activity_loss")
        if range_ratio < args.min_pose_range_ratio:
            reasons.append("pose_range_loss")
        if jump_ratio > args.max_jump_ratio:
            reasons.append("jump_increase")
        peak_improved = candidate_peak <= original_peak * args.max_peak_ratio
        peak_safe = candidate_peak <= args.allowed_peak_dps * args.allowed_peak_slack
        if not (peak_improved or peak_safe):
            reasons.append("yaw_peak_not_improved")

        accepted = not reasons
        if accepted:
            alpha = np.maximum(
                cosine_window(window_len, edge=args.blend_edge),
                mask * args.event_blend_floor,
            )
            output[window_start : window_start + window_len] = blend_motion_so3_np(
                window, candidate, alpha
            )
            occupied.append((window_start, window_end))

        reports.append({
            "event": event.to_dict(),
            "window_start": int(window_start),
            "window_end": int(window_end),
            "observed_duration": observed_duration,
            "predicted_duration": predicted_duration,
            "predicted_duration_bin": predicted_bin,
            "duration_bin_confidence": bin_confidence,
            "expansion_ratio": expansion_ratio,
            "edit_probability": edit_probability,
            "original_peak_yaw_dps": original_peak,
            "candidate_peak_yaw_dps": candidate_peak,
            "activity_ratio": activity_ratio,
            "pose_range_ratio": range_ratio,
            "jump_ratio": jump_ratio,
            "accepted": accepted,
            "reasons": reasons,
        })

    output = contact_debounce(output, min_run=args.contact_min_run)
    return output.astype(np.float32), {
        "checkpoint": args.checkpoint,
        "detected_events": len(detected),
        "accepted_events": sum(int(item["accepted"]) for item in reports),
        "events": reports,
        "input_peak_yaw_dps": percentile_peak(motion),
        "output_peak_yaw_dps": percentile_peak(output),
        "input_activity": rotation_activity_np(motion),
        "output_activity": rotation_activity_np(output),
        "input_pose_range": rotation_range_np(motion),
        "output_pose_range": rotation_range_np(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", action="append", default=[])
    parser.add_argument("--motion_glob", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--suffix", default="_v23v23")
    parser.add_argument("--detect_peak_dps", type=float, default=14.0)
    parser.add_argument("--apply_peak_dps", type=float, default=90.0)
    parser.add_argument("--allowed_peak_dps", type=float, default=130.0)
    parser.add_argument("--allowed_peak_slack", type=float, default=1.10)
    parser.add_argument("--max_peak_ratio", type=float, default=0.94)
    parser.add_argument("--min_turn_angle_deg", type=float, default=10.0)
    parser.add_argument("--min_gap", type=int, default=16)
    parser.add_argument("--detect_min_duration", type=int, default=12)
    parser.add_argument("--detect_max_duration", type=int, default=0)
    parser.add_argument("--detect_threshold_ratio", type=float, default=0.12)
    parser.add_argument("--max_events", type=int, default=4)
    parser.add_argument("--activity_threshold_ratio", type=float, default=0.22)
    parser.add_argument("--boundary_yaw_ratio", type=float, default=0.04)
    parser.add_argument("--quiet_run", type=int, default=8)
    parser.add_argument("--opposite_run", type=int, default=4)
    parser.add_argument("--phrase_margin", type=int, default=3)
    parser.add_argument("--slow_pose_span", type=int, default=10)
    parser.add_argument("--slow_angle_window", type=int, default=24)
    parser.add_argument("--search_duration_multiplier", type=float, default=1.80)
    parser.add_argument("--split_valley_radius", type=int, default=3)
    parser.add_argument("--reversal_angle_deg", type=float, default=7.0)
    parser.add_argument("--secondary_peak_ratio", type=float, default=0.48)
    parser.add_argument("--split_score_threshold", type=float, default=0.68)
    parser.add_argument("--long_split_score_threshold", type=float, default=0.30)
    parser.add_argument("--min_direction_consistency", type=float, default=0.18)
    parser.add_argument("--cumulative_low", type=float, default=0.03)
    parser.add_argument("--cumulative_high", type=float, default=0.97)
    parser.add_argument("--mask_context", type=int, default=6)
    parser.add_argument("--blend_edge", type=int, default=12)
    parser.add_argument("--event_blend_floor", type=float, default=0.90)
    parser.add_argument("--min_edit_probability", type=float, default=0.70)
    parser.add_argument("--min_duration_bin_confidence", type=float, default=0.30)
    parser.add_argument("--min_expansion_ratio", type=float, default=1.06)
    parser.add_argument("--min_activity_ratio", type=float, default=0.75)
    parser.add_argument("--min_pose_range_ratio", type=float, default=0.93)
    parser.add_argument("--max_jump_ratio", type=float, default=1.15)
    parser.add_argument("--contact_min_run", type=int, default=3)
    args = parser.parse_args()

    paths = [Path(value) for value in args.motion]
    if args.motion_glob:
        paths.extend(Path(value) for value in glob.glob(args.motion_glob, recursive=True))
    paths = sorted({path.resolve() for path in paths if path.is_file()})
    if not paths:
        raise RuntimeError("No input motion files")
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = load_v23_checkpoint(args.checkpoint, device=device)

    summary = []
    for path in paths:
        motion = load_motion_file(path)
        corrected, report = apply_one(motion, bundle, args, device)
        output_path = output_dir / f"{path.stem}{args.suffix}.npy"
        report_path = output_dir / f"{path.stem}{args.suffix}.json"
        np.save(output_path, corrected)
        report["input"] = str(path)
        report["output"] = str(output_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary.append(report)
        print(
            f"{path.name}: detected={report['detected_events']} accepted={report['accepted_events']} "
            f"peak={report['input_peak_yaw_dps']:.2f}->{report['output_peak_yaw_dps']:.2f} "
            f"activity={report['input_activity']:.5f}->{report['output_activity']:.5f}",
            flush=True,
        )
    (output_dir / "V23_V2_4_RUNTIME_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
