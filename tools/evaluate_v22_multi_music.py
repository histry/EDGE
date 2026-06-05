#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate V22 multi-music schedules, diversity, boundary and turn speed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from tools.v21_common import ROOT_X, ROOT_Z, ROT, json_safe, motion_mmr_embedding
from tools.v22_turn_utils import detect_turn_events, yaw_speed_dps_np


def motion_metrics(x: np.ndarray, boundaries: list[int]) -> Dict[str, float]:
    if x.ndim == 3:
        x = x[0]
    rot = x[:, ROT]
    frame_velocity = np.mean(np.abs(np.diff(rot, axis=0)), axis=1) if len(x) > 1 else np.zeros((0,), dtype=np.float32)
    frame_acc = np.mean(np.abs(np.diff(rot, n=2, axis=0)), axis=1) if len(x) > 2 else np.zeros((0,), dtype=np.float32)
    tail = frame_velocity[-30:] if len(frame_velocity) else frame_velocity
    yaw_speed = yaw_speed_dps_np(x, fps=30.0)
    turns = detect_turn_events(x, fps=30.0, min_peak_dps=35.0, min_gap=12, min_duration=3)

    boundary_velocity = []
    velocity_jumps = []
    pose_jumps = []
    for boundary in boundaries:
        b = int(boundary)
        if not (2 <= b <= len(x) - 2):
            continue
        lo = max(0, b - 5)
        hi = min(len(frame_velocity), b + 5)
        boundary_velocity.extend(frame_velocity[lo:hi].tolist())
        vb = rot[b - 1] - rot[b - 2]
        va = rot[b + 1] - rot[b]
        velocity_jumps.append(float(np.mean(np.abs(va - vb))))
        pose_jumps.append(float(np.mean(np.abs(rot[b] - rot[b - 1]))))

    return {
        "mean_activity": float(frame_velocity.mean()) if len(frame_velocity) else 0.0,
        "tail_activity": float(tail.mean()) if len(tail) else 0.0,
        "activity_p90": float(np.percentile(frame_velocity, 90)) if len(frame_velocity) else 0.0,
        "mean_acceleration": float(frame_acc.mean()) if len(frame_acc) else 0.0,
        "p95_acceleration": float(np.percentile(frame_acc, 95)) if len(frame_acc) else 0.0,
        "root_radius": float(np.sqrt(x[:, ROOT_X] ** 2 + x[:, ROOT_Z] ** 2).max()) if len(x) else 0.0,
        "contact_switch": float(np.abs(np.diff(x[:, :4], axis=0)).sum()) if len(x) > 1 else 0.0,
        "mean_yaw_speed_dps": float(yaw_speed.mean()) if len(yaw_speed) else 0.0,
        "p95_yaw_speed_dps": float(np.percentile(yaw_speed, 95)) if len(yaw_speed) else 0.0,
        "max_yaw_speed_dps": float(yaw_speed.max()) if len(yaw_speed) else 0.0,
        "turn_count": int(len(turns)),
        "max_turn_path_angle_deg": float(max((event.path_angle_deg for event in turns), default=0.0)),
        "boundary_mean_velocity": float(np.mean(boundary_velocity)) if boundary_velocity else 0.0,
        "boundary_mean_velocity_jump": float(np.mean(velocity_jumps)) if velocity_jumps else 0.0,
        "boundary_max_velocity_jump": float(np.max(velocity_jumps)) if velocity_jumps else 0.0,
        "boundary_mean_pose_jump": float(np.mean(pose_jumps)) if pose_jumps else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    motions: Dict[str, np.ndarray] = {}
    reports: Dict[str, dict] = {}
    for path in sorted(run_dir.glob("*_v22.npy")):
        key = path.name.replace("_v22.npy", "")
        motions[key] = np.load(path, allow_pickle=True).astype(np.float32)
        report_path = run_dir / f"{key}_v22.schedule_report.json"
        if report_path.is_file():
            reports[key] = json.loads(report_path.read_text(encoding="utf-8"))

    result = {"version": "v22_evaluation", "run_dir": str(run_dir), "motions": {}, "pairwise": []}
    embeddings = {}
    event_ids = {}
    families = {}
    for key, x in motions.items():
        xx = x[0] if x.ndim == 3 else x
        report = reports.get(key, {})
        boundaries = [int(value) for value in report.get("boundaries", []) if 0 < int(value) < len(xx)]
        result["motions"][key] = motion_metrics(xx, boundaries)
        embeddings[key] = motion_mmr_embedding(xx)
        schedule = report.get("schedule", [])
        event_ids[key] = {str(item.get("event_id", "")) for item in schedule}
        families[key] = {str(item.get("family_id", item.get("motion_family_id", ""))) for item in schedule}
        style_scores = []
        for item in schedule:
            slot = item.get("v22_slot", {})
            style_scores.append(float(slot.get("style", item.get("v21_style_score", item.get("dunhuang_style_score_v20f3", 0.0)))))
        result["motions"][key]["mean_style_score"] = float(np.mean(style_scores)) if style_scores else 0.0
        refinement = report.get("turn_refinement", {})
        result["motions"][key]["turn_events_refined"] = int(refinement.get("events_refined", 0))
        if refinement.get("events"):
            before = [float(row.get("peak_before_dps", 0.0)) for row in refinement["events"]]
            after = [float(row.get("peak_after_dps", 0.0)) for row in refinement["events"]]
            result["motions"][key]["refined_peak_before_dps"] = float(np.mean(before))
            result["motions"][key]["refined_peak_after_dps"] = float(np.mean(after))

    keys = sorted(motions)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            result["pairwise"].append(
                {
                    "a": a,
                    "b": b,
                    "motion_cosine": float(embeddings[a] @ embeddings[b]),
                    "event_overlap": len(event_ids[a] & event_ids[b]),
                    "family_overlap": len(families[a] & families[b]),
                }
            )

    out = Path(args.out) if args.out else run_dir / "V22_EVALUATION.json"
    out.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
