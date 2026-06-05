#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_motion(path: Path) -> np.ndarray:
    motion = np.load(path, allow_pickle=True).astype(np.float32)

    if motion.ndim == 3:
        motion = motion[0]

    if motion.ndim != 2 or motion.shape[1] < 151:
        raise ValueError(f"Expected [T,151], got {motion.shape}")

    return motion


def evaluate_motion(
    motion: np.ndarray,
    boundaries: list[int],
    boundary_radius: int = 4,
) -> dict:
    pose = motion[:, 7:151]

    velocity = np.mean(
        np.abs(np.diff(pose, axis=0)),
        axis=1,
    )

    acceleration = np.mean(
        np.abs(np.diff(pose, n=2, axis=0)),
        axis=1,
    )

    boundary_velocity = []
    interior_velocity = []

    boundary_mask = np.zeros(
        len(velocity),
        dtype=bool,
    )

    for boundary in boundaries:
        start = max(0, boundary - boundary_radius)
        end = min(len(velocity), boundary + boundary_radius)

        boundary_mask[start:end] = True

        if start < end:
            boundary_velocity.extend(
                velocity[start:end].tolist()
            )

    interior_velocity = velocity[~boundary_mask]

    return {
        "mean_velocity": float(np.mean(velocity)),
        "p90_velocity": float(np.percentile(velocity, 90)),
        "p95_velocity": float(np.percentile(velocity, 95)),
        "max_velocity": float(np.max(velocity)),
        "mean_acceleration": float(np.mean(acceleration)),
        "p95_acceleration": float(np.percentile(acceleration, 95)),
        "boundary_mean_velocity": (
            float(np.mean(boundary_velocity))
            if boundary_velocity else 0.0
        ),
        "boundary_p95_velocity": (
            float(np.percentile(boundary_velocity, 95))
            if boundary_velocity else 0.0
        ),
        "interior_mean_velocity": (
            float(np.mean(interior_velocity))
            if len(interior_velocity) else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    results = {}

    for report in sorted(
        run_dir.glob("*_v21.schedule_report.json")
    ):
        name = report.name.replace(
            "_v21.schedule_report.json",
            "",
        )

        motion_path = run_dir / f"{name}_v21.npy"

        if not motion_path.is_file():
            continue

        report_data = json.loads(
            report.read_text(encoding="utf-8")
        )

        boundaries = [
            int(x)
            for x in report_data.get("boundaries", [])
            if 0 < int(x) < 150
        ]

        motion = load_motion(motion_path)

        results[name] = evaluate_motion(
            motion,
            boundaries,
        )

    output_path = (
        Path(args.out)
        if args.out
        else run_dir / "V21_SPEED_EVALUATION.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for name, values in results.items():
        print(
            name,
            "mean_vel=", round(values["mean_velocity"], 6),
            "p95_vel=", round(values["p95_velocity"], 6),
            "boundary_vel=", round(
                values["boundary_mean_velocity"], 6
            ),
            "interior_vel=", round(
                values["interior_mean_velocity"], 6
            ),
            "p95_acc=", round(
                values["p95_acceleration"], 6
            ),
        )

    print("saved:", output_path)


if __name__ == "__main__":
    main()
