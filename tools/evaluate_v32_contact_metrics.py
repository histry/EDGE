#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate contact calibration and foot physics for V32 outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v29_motion_geometry import (
    CONTACT,
    motion_to_joint_positions_np,
)

FOOT_JOINTS = (7, 8, 10, 11)


def load_motion(path: str | Path) -> np.ndarray:
    motion = np.asarray(np.load(path, allow_pickle=True), np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape}")
    return motion


def transition_ranges(report_path: str, total: int) -> List[tuple[int, int]]:
    if not report_path or not Path(report_path).is_file():
        return []
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    allocation = report.get("allocation", {})
    boundaries = [int(x) for x in allocation.get("output_boundaries", [])]
    lengths = [int(x) for x in allocation.get("transition_lengths", [])]
    rows = []
    for slot in range(1, min(len(boundaries), len(lengths))):
        start = boundaries[slot]
        end = min(total, start + lengths[slot])
        if end > start:
            rows.append((start, end))
    return rows


def metrics(motion: np.ndarray, fps: float) -> Dict[str, object]:
    positions = motion_to_joint_positions_np(motion)
    feet = positions[:, FOOT_JOINTS]
    velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    horizontal_speed = np.linalg.norm(
        velocity[..., (0, 2)], axis=-1
    )
    contact_prob = motion[:, CONTACT].clip(0.0, 1.0)
    hard_contact = contact_prob >= 0.5
    ground = float(np.percentile(feet[..., 1], 5))
    relative_height = feet[..., 1] - ground
    proxy = (
        (horizontal_speed < 0.08)
        & (relative_height < 0.035)
    )
    tp = np.logical_and(hard_contact, proxy).sum(axis=0)
    fp = np.logical_and(hard_contact, ~proxy).sum(axis=0)
    fn = np.logical_and(~hard_contact, proxy).sum(axis=0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(
        precision + recall, 1e-8
    )
    slip = (
        (horizontal_speed * contact_prob).sum(axis=0)
        / np.maximum(contact_prob.sum(axis=0), 1e-6)
    )
    penetration = np.maximum(
        ground - feet[..., 1] - 0.008, 0.0
    )
    return {
        "frames": int(len(motion)),
        "soft_contact_rate": contact_prob.mean(axis=0).astype(float).tolist(),
        "hard_contact_rate": hard_contact.mean(axis=0).astype(float).tolist(),
        "kinematic_proxy_rate": proxy.mean(axis=0).astype(float).tolist(),
        "precision": precision.astype(float).tolist(),
        "recall": recall.astype(float).tolist(),
        "f1": f1.astype(float).tolist(),
        "contact_slip_mean": slip.astype(float).tolist(),
        "contact_slip_mean_all": float(np.mean(slip)),
        "penetration_mean": float(np.mean(penetration**2)),
        "penetration_p95": float(np.percentile(penetration, 95)),
        "contact_switch_rate": (
            np.abs(np.diff(contact_prob, axis=0)).mean(axis=0)
            .astype(float).tolist()
            if len(contact_prob) > 1 else [0.0] * 4
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--schedule_report", default="")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    motion = load_motion(args.motion)
    result = {
        "version": "v32_contact_physics_metrics",
        "whole_song": metrics(motion, args.fps),
        "transitions": [],
    }
    for start, end in transition_ranges(
        args.schedule_report, len(motion)
    ):
        row = metrics(motion[start:end], args.fps)
        row.update({"start": start, "end": end})
        result["transitions"].append(row)
    if result["transitions"]:
        result["transition_mean_slip"] = float(np.mean([
            row["contact_slip_mean_all"]
            for row in result["transitions"]
        ]))
        result["transition_mean_penetration"] = float(np.mean([
            row["penetration_mean"]
            for row in result["transitions"]
        ]))
    else:
        result["transition_mean_slip"] = 0.0
        result["transition_mean_penetration"] = 0.0

    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
