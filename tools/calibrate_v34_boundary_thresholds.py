#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibrate V34 absolute boundary gates from natural intra-event motion.

The generated thresholds are empirical engineering limits, not physical SI
constants.  They are estimated from unedited, contact-back-injected events so a
candidate must stay near the high percentile of naturally occurring Dunhuang
motion rather than merely improve over a bad synthetic baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.schedule_v21_multi_music import load_shared_index
from tools.v21_common import load_motion
from tools.v32_transition_quality import transition_risk


METRICS = {
    "V34_MAX_BOUNDARY_JERK": ("boundary_joint_jerk_max", 500.0),
    "V34_MAX_BOUNDARY_ANGULAR_JERK": (
        "boundary_angular_jerk_max", 500.0
    ),
    "V34_MAX_ENTRY_ROTATION_STEP_RAD": (
        "entry_rotation_step_rad", 0.06
    ),
    "V34_MAX_EXIT_ROTATION_STEP_RAD": (
        "exit_rotation_step_rad", 0.06
    ),
    "V34_MAX_ENTRY_FK_JUMP": ("entry_fk_jump", 0.015),
    "V34_MAX_EXIT_FK_JUMP": ("exit_fk_jump", 0.015),
    "V34_MAX_EXIT_ACCELERATION": ("exit_acceleration", 2.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--duration_index_npz", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_env", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--samples_per_event", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--minimum_samples", type=int, default=1000)
    args = parser.parse_args()

    _, _, items = load_shared_index(
        Path(args.index_json), Path(args.duration_index_npz)
    )
    rows: Dict[str, List[float]] = {
        metric: [] for metric, _ in METRICS.values()
    }
    events_used = 0
    samples = 0
    for item in items:
        path = Path(str(item.get("pkl", item.get("path", ""))))
        motion = load_motion(path)
        if len(motion) < 10:
            continue
        lo = 4
        hi = len(motion) - 5
        count = min(max(1, args.samples_per_event), max(1, hi - lo + 1))
        boundaries = np.unique(
            np.linspace(lo, hi, count, dtype=np.int64)
        )
        used = False
        for boundary in boundaries:
            b = int(boundary)
            previous = motion[b - 4:b]
            transition = motion[b:b + 1]
            following = motion[b + 1:b + 5]
            if not len(previous) or not len(following):
                continue
            risk = transition_risk(
                previous, transition, following, fps=args.fps
            )
            for metric, _ in METRICS.values():
                value = float(risk[metric])
                if np.isfinite(value):
                    rows[metric].append(value)
            samples += 1
            used = True
        events_used += int(used)

    if samples < int(args.minimum_samples):
        raise RuntimeError(
            f"Only {samples} natural boundary samples; expected "
            f">={int(args.minimum_samples)}"
        )

    thresholds: Dict[str, float] = {}
    distributions = {}
    q = float(np.clip(args.quantile, 0.90, 0.9999))
    multiplier = max(float(args.multiplier), 1.0)
    for env_name, (metric, floor) in METRICS.items():
        values = np.asarray(rows[metric], np.float64)
        percentile = float(np.quantile(values, q))
        threshold = max(float(floor), percentile * multiplier)
        thresholds[env_name] = threshold
        distributions[metric] = {
            "count": int(len(values)),
            "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "calibration_quantile": percentile,
            "maximum": float(values.max()),
        }

    result = {
        "version": "v34_natural_boundary_threshold_calibration",
        "source_index": args.index_json,
        "events_used": events_used,
        "natural_boundary_samples": samples,
        "fps": args.fps,
        "quantile": q,
        "multiplier": multiplier,
        "thresholds": thresholds,
        "distributions": distributions,
        "interpretation": (
            "Empirical gates calibrated from natural intra-event boundaries; "
            "not universal physical constants."
        ),
    }
    output_json = Path(args.out_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_env = Path(args.out_env)
    output_env.write_text(
        "\n".join(
            f"export V34_CALIBRATED_{name[4:]}={value:.12g}"
            for name, value in thresholds.items()
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "events_used": events_used,
        "natural_boundary_samples": samples,
        "thresholds": thresholds,
    }, ensure_ascii=False, indent=2))
    print(f"[SAVED] {output_json}")
    print(f"[SAVED] {output_env}")


if __name__ == "__main__":
    main()
