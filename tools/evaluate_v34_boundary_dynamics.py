#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate absolute cross-boundary dynamics for a generated V34 whole song."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v32_transition_quality import transition_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max_boundary_jerk", type=float, default=5000.0)
    parser.add_argument("--max_exit_rotation_step_rad", type=float, default=0.12)
    parser.add_argument("--max_exit_fk_jump", type=float, default=0.040)
    args = parser.parse_args()

    motion = np.load(args.motion, allow_pickle=True)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    motion = np.asarray(motion, np.float32)
    report = json.loads(
        Path(args.schedule_report).read_text(encoding="utf-8")
    )
    allocation = report.get("allocation", {})
    contents = [int(x) for x in allocation.get("content_lengths", [])]
    transitions = [int(x) for x in allocation.get("transition_lengths", [])]
    if len(contents) != len(transitions):
        raise RuntimeError("content/transition length mismatch")

    cursor = 0
    content_ranges = []
    transition_ranges = []
    for slot, content_length in enumerate(contents):
        transition_length = transitions[slot]
        if slot == 0:
            transition_ranges.append((cursor, cursor))
        else:
            transition_ranges.append((cursor, cursor + transition_length))
            cursor += transition_length
        content_ranges.append((cursor, cursor + content_length))
        cursor += content_length
    if cursor != len(motion):
        raise RuntimeError(
            f"Schedule reconstructs {cursor} frames, motion has {len(motion)}"
        )

    rows = []
    for slot in range(1, len(contents)):
        previous_start, previous_end = content_ranges[slot - 1]
        transition_start, transition_end = transition_ranges[slot]
        content_start, content_end = content_ranges[slot]
        previous = motion[max(previous_start, previous_end - 4):previous_end]
        transition = motion[transition_start:transition_end]
        following = motion[content_start:min(content_end, content_start + 4)]
        risk = transition_risk(previous, transition, following, fps=args.fps)
        violations = {
            "boundary_jerk": (
                risk["boundary_joint_jerk_max"] > args.max_boundary_jerk
            ),
            "exit_rotation": (
                risk["exit_rotation_step_rad"]
                > args.max_exit_rotation_step_rad
            ),
            "exit_fk": risk["exit_fk_jump"] > args.max_exit_fk_jump,
        }
        rows.append({
            "slot": slot,
            "transition_start": transition_start,
            "transition_end": transition_end,
            "content_start": content_start,
            "risk": risk,
            "violations": violations,
            "safe": not any(violations.values()),
        })

    keys = (
        "entry_boundary_jerk", "exit_boundary_jerk",
        "boundary_joint_jerk_max", "boundary_angular_jerk_max",
        "entry_rotation_step_rad", "exit_rotation_step_rad",
        "entry_fk_jump", "exit_fk_jump", "foot_slip",
    )
    summary = {}
    for key in keys:
        values = [float(row["risk"][key]) for row in rows]
        summary[key] = {
            "mean": float(np.mean(values)) if values else 0.0,
            "p95": float(np.percentile(values, 95)) if values else 0.0,
            "max": float(np.max(values)) if values else 0.0,
        }
    result = {
        "version": "v34_absolute_cross_boundary_dynamics",
        "motion": args.motion,
        "frames": len(motion),
        "num_boundaries": len(rows),
        "safe_boundaries": sum(bool(row["safe"]) for row in rows),
        "unsafe_boundaries": sum(not bool(row["safe"]) for row in rows),
        "thresholds": {
            "max_boundary_jerk": args.max_boundary_jerk,
            "max_exit_rotation_step_rad": args.max_exit_rotation_step_rad,
            "max_exit_fk_jump": args.max_exit_fk_jump,
        },
        "summary": summary,
        "boundaries": rows,
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "num_boundaries": len(rows),
        "safe_boundaries": result["safe_boundaries"],
        "unsafe_boundaries": result["unsafe_boundaries"],
        "exit_boundary_jerk_max": summary.get(
            "exit_boundary_jerk", {}
        ).get("max", 0.0),
        "exit_rotation_step_max": summary.get(
            "exit_rotation_step_rad", {}
        ).get("max", 0.0),
    }, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output}")


if __name__ == "__main__":
    main()
