#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motion utilities for root-locked / body-centered evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6


def load_motion(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x)
    while x.ndim > 2:
        x = x[0]
    return x.astype(np.float32)


def freeze_root_xz(x: np.ndarray, freeze_y: bool = False) -> np.ndarray:
    y = np.array(x, copy=True)
    if y.ndim != 2:
        raise ValueError(f"Expected [T,D], got {y.shape}")
    y[:, ROOT_X_IDX] = y[0, ROOT_X_IDX]
    y[:, ROOT_Z_IDX] = y[0, ROOT_Z_IDX]
    if freeze_y and y.shape[-1] > ROOT_Y_IDX:
        y[:, ROOT_Y_IDX] = y[0, ROOT_Y_IDX]
    return y


def metrics(x: np.ndarray) -> Dict[str, float]:
    if x.ndim != 2:
        raise ValueError(f"Expected [T,D], got {x.shape}")

    root_xz = x[:, [ROOT_X_IDX, ROOT_Z_IDX]] if x.shape[-1] > ROOT_Z_IDX else np.zeros((len(x), 2))
    root_path = float(np.linalg.norm(np.diff(root_xz, axis=0), axis=-1).sum()) if len(x) > 1 else 0.0
    root_disp = float(np.linalg.norm(root_xz[-1] - root_xz[0])) if len(x) > 1 else 0.0

    pose = x[:, 7:] if x.shape[-1] >= 151 else x
    dpose = np.diff(pose, axis=0)
    frame_energy = np.linalg.norm(dpose, axis=-1) if len(dpose) else np.zeros((1,))

    if x.shape[-1] >= 151:
        # Proxy groups.
        lower = x[:, 7:79]
        torso = x[:, 31:79]
        upper = x[:, 79:151]
        lower_activity = float(np.linalg.norm(np.diff(lower, axis=0), axis=-1).mean())
        torso_activity = float(np.linalg.norm(np.diff(torso, axis=0), axis=-1).mean())
        upper_activity = float(np.linalg.norm(np.diff(upper, axis=0), axis=-1).mean())
    else:
        lower_activity = torso_activity = upper_activity = float("nan")

    jerk = float(np.linalg.norm(np.diff(x, n=3, axis=0), axis=-1).mean()) if len(x) > 4 else 0.0

    return {
        "motion_energy_pose_only": float(frame_energy.mean()),
        "motion_energy_pose_p95": float(np.percentile(frame_energy, 95)),
        "upper_activity_proxy": upper_activity,
        "torso_activity_proxy": torso_activity,
        "lower_activity_proxy": lower_activity,
        "root_path": root_path,
        "root_disp": root_disp,
        "jerk_proxy": jerk,
        "freezing_rate_proxy": float(np.mean(frame_energy < 1e-3)),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    fz = sub.add_parser("freeze-root")
    fz.add_argument("--input", required=True)
    fz.add_argument("--output", required=True)
    fz.add_argument("--freeze_y", action="store_true")

    ev = sub.add_parser("eval")
    ev.add_argument("--input", required=True)
    ev.add_argument("--output_json", default=None)

    batch = sub.add_parser("batch-eval")
    batch.add_argument("--glob", required=True)
    batch.add_argument("--output_csv", required=True)

    args = ap.parse_args()

    if args.cmd == "freeze-root":
        x = load_motion(args.input)
        y = freeze_root_xz(x, freeze_y=args.freeze_y)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, y)
        print(f"saved {args.output}, shape={y.shape}")

    elif args.cmd == "eval":
        x = load_motion(args.input)
        m_raw = metrics(x)
        x_lock = freeze_root_xz(x)
        m_lock = metrics(x_lock)
        out = {"input": args.input, "raw": m_raw, "root_locked": m_lock}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2))

    elif args.cmd == "batch-eval":
        rows = []
        for p in sorted(Path(".").glob(args.glob)):
            if p.name.endswith("_raw.npy") or "_mid" in p.name or "_unit" in p.name:
                continue
            try:
                x = load_motion(str(p))
                r = metrics(x)
                rl = metrics(freeze_root_xz(x))
                row = {"case": p.stem, "path": str(p)}
                for k, v in r.items():
                    row[f"raw_{k}"] = v
                for k, v in rl.items():
                    row[f"rootlock_{k}"] = v
                rows.append(row)
            except Exception as e:
                rows.append({"case": p.stem, "path": str(p), "error": repr(e)})

        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r.keys()})
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"saved {args.output_csv}")


if __name__ == "__main__":
    main()
