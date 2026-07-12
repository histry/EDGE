#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Independent root-yaw/mechanical-spin audit.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tools.v46_49_gravity_contract import rot6d_to_matrix_np


def moving_average(x, n):
    x = np.asarray(x, np.float32)
    n = max(1, int(n))
    if n % 2 == 0:
        n += 1
    if n <= 1:
        return x
    p = n // 2
    xp = np.pad(x, (p, p), mode="edge")
    return np.convolve(
        xp,
        np.ones(n, np.float32) / n,
        mode="valid",
    )


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(
        np.concatenate([[0], m.astype(np.int8), [0]])
    )
    return list(
        zip(
            np.where(d == 1)[0].tolist(),
            np.where(d == -1)[0].tolist(),
        )
    )


def audit(motion, fps=30.0, min_speed=7.0, window_s=3.0):
    x = np.asarray(motion, np.float32)
    if x.ndim == 3:
        x = x[0]

    R = rot6d_to_matrix_np(
        x[:, 7:13].reshape(-1, 1, 6)
    )[:, 0]
    forward = R[:, :, 2]
    yaw = np.unwrap(
        np.arctan2(forward[:, 0], forward[:, 2])
    )
    speed = np.degrees(np.gradient(yaw) * fps)
    smooth = moving_average(speed, int(round(fps)))
    n = max(3, int(round(window_s * fps)))
    local_mean = moving_average(smooth, n)
    local_dev = moving_average(
        np.abs(smooth - local_mean), n
    )

    mechanical = (
        (np.abs(local_mean) >= min_speed)
        & (
            local_dev
            <= np.maximum(4.0, 0.25 * np.abs(local_mean))
        )
        & (np.sign(smooth) == np.sign(local_mean))
    )
    longest = max(
        (b - a for a, b in runs(mechanical)),
        default=0,
    )
    return {
        "frames": int(len(x)),
        "duration_seconds": float(len(x) / fps),
        "net_turns": float(
            (yaw[-1] - yaw[0]) / (2 * np.pi)
        ),
        "absolute_turns": float(
            np.abs(np.diff(yaw)).sum() / (2 * np.pi)
        ),
        "yaw_speed_deg_s_p50": float(
            np.percentile(np.abs(speed), 50)
        ),
        "yaw_speed_deg_s_p95": float(
            np.percentile(np.abs(speed), 95)
        ),
        "yaw_speed_deg_s_max": float(
            np.max(np.abs(speed))
        ),
        "mechanical_spin_ratio": float(
            mechanical.mean()
        ),
        "longest_mechanical_spin_seconds": float(
            longest / fps
        ),
        "mechanical_spin_fail": bool(
            longest / fps > 3.0
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--allow_failed", action="store_true")
    args = ap.parse_args()

    report = audit(
        np.load(args.input, allow_pickle=True),
        fps=args.fps,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if args.allow_failed or not report[
        "mechanical_spin_fail"
    ] else 2


if __name__ == "__main__":
    raise SystemExit(main())
