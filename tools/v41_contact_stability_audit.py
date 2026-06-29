#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit helper for V39 contact stability.

This is optional and does not change generation.  It compares two [T,151] motions
or reads a V39 postprocess summary and prints the key deltas that matter for the
current project: foot skate, penetration, jerk and collision risk.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Any, Dict

import numpy as np

from tools.v34_motion_quality_postprocess import _load_motion, _quality_audit


def _audit(path: str, args: argparse.Namespace) -> Dict[str, Any]:
    motion = _load_motion(Path(path))
    return _quality_audit(
        motion,
        contact_threshold=args.contact_threshold,
        min_contact_frames=args.min_contact_frames,
        floor_margin=args.floor_margin,
        collision_radius=args.collision_radius,
        denoise=True,
        median_size=args.contact_median_size,
        close_holes=args.contact_close_holes,
        open_spikes=args.contact_open_spikes,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--before", default="")
    p.add_argument("--after", default="")
    p.add_argument("--summary_json", default="")
    p.add_argument("--out_json", default="")
    p.add_argument("--contact_threshold", type=float, default=0.65)
    p.add_argument("--min_contact_frames", type=int, default=8)
    p.add_argument("--floor_margin", type=float, default=0.006)
    p.add_argument("--collision_radius", type=float, default=0.16)
    p.add_argument("--contact_median_size", type=int, default=5)
    p.add_argument("--contact_close_holes", type=int, default=7)
    p.add_argument("--contact_open_spikes", type=int, default=4)
    args = p.parse_args()

    if args.summary_json:
        data = json.loads(Path(args.summary_json).read_text(encoding="utf-8"))
        result = {
            "source": args.summary_json,
            "version": data.get("version"),
            "pre_audit": data.get("pre_audit"),
            "post_audit": data.get("post_audit"),
            "audit_improvement": data.get("audit_improvement"),
            "planner_feedback": data.get("planner_feedback"),
        }
    else:
        if not args.before or not args.after:
            raise SystemExit("Provide --summary_json or both --before and --after")
        before = _audit(args.before, args)
        after = _audit(args.after, args)
        result = {
            "before": args.before,
            "after": args.after,
            "pre_audit": before,
            "post_audit": after,
            "delta": {
                "foot_skate_mean_delta": float(before["foot_skate_mean_mpf"] - after["foot_skate_mean_mpf"]),
                "foot_skate_p95_delta": float(before.get("foot_skate_p95_mpf", 0.0) - after.get("foot_skate_p95_mpf", 0.0)),
                "foot_penetration_min_delta": float(after["foot_penetration_min_m"] - before["foot_penetration_min_m"]),
                "jerk_p95_delta": float(before["mean_joint_jerk_p95"] - after["mean_joint_jerk_p95"]),
                "hand_jerk_p95_delta": float(before.get("hand_joint_jerk_p95", 0.0) - after.get("hand_joint_jerk_p95", 0.0)),
                "collision_risk_delta": float(before["collision_risk"] - after["collision_risk"]),
            },
        }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
