#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 whole-song motion evaluation with two-sided transition checks.

The historical filename is retained for compatibility.  Unlike the previous
evaluator, this version evaluates both:
  content -> transition
and
  transition -> next content,
and reports global FK acceleration/jerk statistics and worst joints/frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v29_motion_geometry import (
    CONTACT,
    ROOT_Y,
    endpoint_metrics_np,
    jitter_statistics_np,
    root_yaw_np,
)


def load_motion(path: str | Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    return x


def join_metrics(
    motion: np.ndarray,
    boundary: int,
    side: str,
    fps: float,
) -> Dict[str, float | int | str]:
    boundary = int(boundary)
    if boundary <= 0 or boundary >= len(motion):
        return {
            "boundary": boundary,
            "side": side,
            "valid": False,
        }
    left_start = max(0, boundary - 3)
    right_end = min(len(motion), boundary + 3)
    metrics = endpoint_metrics_np(
        motion[left_start:boundary],
        motion[boundary:right_end],
        fps=fps,
    )
    return {
        "boundary": boundary,
        "side": side,
        "valid": True,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    motion = load_motion(args.motion)
    report = json.loads(
        Path(args.schedule_report).read_text(encoding="utf-8")
    )
    allocation = report["allocation"]
    boundaries = [int(v) for v in allocation["output_boundaries"]]
    transition_lengths = [int(v) for v in allocation["transition_lengths"]]

    joins: List[Dict[str, object]] = []
    for slot in range(1, len(transition_lengths)):
        entry = boundaries[slot]
        exit_ = entry + transition_lengths[slot]
        joins.append(join_metrics(motion, entry, "transition_entry", args.fps))
        joins.append(join_metrics(motion, exit_, "transition_exit", args.fps))

    valid_joins = [row for row in joins if row.get("valid")]
    yaw = root_yaw_np(motion)
    yaw_speed = (
        np.abs(np.diff(yaw)) * args.fps * 180.0 / np.pi
        if len(yaw) >= 2 else np.zeros((0,), dtype=np.float32)
    )
    event_ids = [row["event_id"] for row in report.get("schedule", [])]
    families = [row["family_id"] for row in report.get("schedule", [])]
    jitter = jitter_statistics_np(motion, fps=args.fps)

    def mean_key(key: str) -> float:
        values = [float(row[key]) for row in valid_joins if key in row]
        return float(np.mean(values)) if values else 0.0

    transition_frames = int(sum(transition_lengths))
    result = {
        "version": "v29_two_sided_long_dance_evaluation",
        "motion": str(args.motion),
        "frames": int(len(motion)),
        "seconds": float(len(motion) / args.fps),
        "phrases": len(report.get("schedule", [])),
        "transition_frames": transition_frames,
        "transition_frame_ratio": float(transition_frames / max(len(motion), 1)),
        "yaw_p95_dps": float(np.percentile(yaw_speed, 95)) if len(yaw_speed) else 0.0,
        "yaw_max_dps": float(yaw_speed.max()) if len(yaw_speed) else 0.0,
        "event_unique_ratio": len(set(event_ids)) / max(len(event_ids), 1),
        "family_unique_ratio": len(set(families)) / max(len(families), 1),
        "boundary_metrics": valid_joins,
        "boundary_pose_jump_mean": mean_key("pose_jump"),
        "boundary_velocity_jump_mean": mean_key("velocity_jump"),
        "boundary_acceleration_jump_mean": mean_key("acceleration_jump"),
        "boundary_pose_jump_deg_rms_mean": mean_key("pose_jump_deg_rms"),
        "boundary_velocity_jump_deg_s_rms_mean": mean_key("velocity_jump_deg_s_rms"),
        "boundary_acceleration_jump_deg_s2_rms_mean": mean_key("acceleration_jump_deg_s2_rms"),
        "boundary_contact_jump_mean": mean_key("contact_jump"),
        "boundary_root_y_jump_mean": mean_key("root_y_jump"),
        "jitter": jitter,
        "exact_length_match": int(len(motion)) == int(allocation["total_frames"]),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
