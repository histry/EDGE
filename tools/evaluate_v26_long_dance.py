#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate V26 whole-song choreography and phrase-boundary continuity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tools.v21_common import CONTACT, ROT
from tools.v22_turn_utils import yaw_speed_dps_np
from tools.v23_duration_utils import rotation_activity_np, rotation_range_np


def load_motion(path: str | Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {x.shape}")
    return x


def local_boundary_metrics(motion: np.ndarray, boundary: int, radius: int = 3) -> Dict[str, float]:
    boundary = int(boundary)
    if boundary <= 1 or boundary >= len(motion) - 2:
        return {"boundary": boundary, "pose_jump": 0.0, "velocity_jump": 0.0, "acceleration_jump": 0.0}
    pose_jump = float(np.linalg.norm(motion[boundary, ROT] - motion[boundary - 1, ROT]) / np.sqrt(144.0))
    left_v = motion[boundary - 1, ROT] - motion[boundary - 2, ROT]
    right_v = motion[boundary + 1, ROT] - motion[boundary, ROT]
    velocity_jump = float(np.linalg.norm(left_v - right_v) / np.sqrt(144.0))
    left_a = motion[boundary - 1, ROT] - 2 * motion[boundary - 2, ROT] + motion[boundary - 3, ROT]
    right_a = motion[boundary + 2, ROT] - 2 * motion[boundary + 1, ROT] + motion[boundary, ROT]
    acceleration_jump = float(np.linalg.norm(left_a - right_a) / np.sqrt(144.0))
    return {
        "boundary": boundary,
        "pose_jump": pose_jump,
        "velocity_jump": velocity_jump,
        "acceleration_jump": acceleration_jump,
        "contact_jump": float(np.abs(motion[boundary, CONTACT] - motion[boundary - 1, CONTACT]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    motion = load_motion(args.motion)
    report = json.loads(Path(args.schedule_report).read_text(encoding="utf-8"))
    boundaries = report["allocation"]["output_boundaries"][1:-1]
    boundary_rows = [local_boundary_metrics(motion, boundary) for boundary in boundaries]
    yaw = np.abs(yaw_speed_dps_np(motion, fps=args.fps, smooth_window=5))
    event_ids = [row["event_id"] for row in report["schedule"]]
    families = [row["family_id"] for row in report["schedule"]]
    result = {
        "version": "v26_long_dance_evaluation",
        "motion": str(args.motion),
        "frames": int(len(motion)),
        "seconds": float(len(motion) / args.fps),
        "phrases": len(report["schedule"]),
        "activity": rotation_activity_np(motion),
        "pose_range": rotation_range_np(motion),
        "yaw_p95_dps": float(np.percentile(yaw, 95)) if len(yaw) else 0.0,
        "yaw_max_dps": float(yaw.max()) if len(yaw) else 0.0,
        "event_unique_ratio": len(set(event_ids)) / max(len(event_ids), 1),
        "family_unique_ratio": len(set(families)) / max(len(families), 1),
        "boundary_metrics": boundary_rows,
        "boundary_pose_jump_mean": float(np.mean([x["pose_jump"] for x in boundary_rows])) if boundary_rows else 0.0,
        "boundary_velocity_jump_mean": float(np.mean([x["velocity_jump"] for x in boundary_rows])) if boundary_rows else 0.0,
        "boundary_acceleration_jump_mean": float(np.mean([x["acceleration_jump"] for x in boundary_rows])) if boundary_rows else 0.0,
        "exact_length_match": int(len(motion)) == int(report["allocation"]["total_frames"]),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
