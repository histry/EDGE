#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locate exact mechanical-spin intervals in an EDGE151D motion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

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
    return np.convolve(xp, np.ones(n, np.float32) / n, mode="valid")


def runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_speed_deg_s", type=float, default=7.0)
    ap.add_argument("--window_s", type=float, default=3.0)
    args = ap.parse_args()

    x = np.load(args.input, allow_pickle=True).astype(np.float32)
    if x.ndim == 3:
        x = x[0]

    R = rot6d_to_matrix_np(x[:, 7:13].reshape(-1, 1, 6))[:, 0]
    forward = R[:, :, 2]
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    speed = np.degrees(np.gradient(yaw) * args.fps)

    smooth = moving_average(speed, int(round(args.fps)))
    n = max(3, int(round(args.window_s * args.fps)))
    local_mean = moving_average(smooth, n)
    local_dev = moving_average(np.abs(smooth - local_mean), n)

    mechanical = (
        (np.abs(local_mean) >= args.min_speed_deg_s)
        & (local_dev <= np.maximum(4.0, 0.25 * np.abs(local_mean)))
        & (np.sign(smooth) == np.sign(local_mean))
    )

    rows = []
    ordered = sorted(runs(mechanical), key=lambda ab: ab[1] - ab[0], reverse=True)
    for rank, (a, b) in enumerate(ordered, 1):
        if b <= a:
            continue
        rows.append({
            "rank": rank,
            "start_frame": int(a),
            "end_frame": int(b),
            "start_seconds": float(a / args.fps),
            "end_seconds": float(b / args.fps),
            "duration_seconds": float((b - a) / args.fps),
            "mean_yaw_speed_deg_s": float(np.mean(speed[a:b])),
            "p95_abs_yaw_speed_deg_s": float(np.percentile(np.abs(speed[a:b]), 95)),
            "yaw_change_degrees": float(np.degrees(yaw[b - 1] - yaw[a])),
        })

    Path(args.out_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) if rows else [
            "rank", "start_frame", "end_frame", "start_seconds", "end_seconds",
            "duration_seconds", "mean_yaw_speed_deg_s",
            "p95_abs_yaw_speed_deg_s", "yaw_change_degrees",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows[:10], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
