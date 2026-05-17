#!/usr/bin/env python3
"""Lightweight V3 unit-reconstruction diagnostics.

This script does not render. It computes temporal continuity metrics from
generated .npy motions, optionally comparing against GT motions.

Examples:
  python tools/eval_v3_unit_recon_metrics.py --pred output/foo.npy
  python tools/eval_v3_unit_recon_metrics.py --pred output/foo.npy --gt data/unit.npy
  python tools/eval_v3_unit_recon_metrics.py --pred_dir output/v3_eval --out_csv output/v3_eval/metrics.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def _load_motion(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.lib.npyio.NpzFile):
        for key in ("motion", "motion_151", "sample", "arr_0"):
            if key in arr:
                arr = arr[key]
                break
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # If batch exists, evaluate the first sample by default.
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,C] or [B,T,C], got {arr.shape} from {path}")
    return arr.astype(np.float64)


def _safe_percentile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, q))


def _motion_metrics(motion: np.ndarray, gt: Optional[np.ndarray] = None) -> Dict[str, float]:
    t, c = motion.shape
    feat = motion[:, 4:] if c == 151 else motion
    vel = np.diff(feat, axis=0)
    speed = np.sqrt(np.sum(vel * vel, axis=-1)) if len(vel) else np.zeros((0,))
    acc = np.diff(feat, n=2, axis=0)
    jerk = np.sqrt(np.sum(acc * acc, axis=-1)) if len(acc) else np.zeros((0,))

    if c == 151:
        root_xz = motion[:, [4, 6]]
        root_range = float(np.linalg.norm(root_xz.max(axis=0) - root_xz.min(axis=0)))
        rot = motion[:, 7:151]
        rot_speed = np.sqrt(np.sum(np.diff(rot, axis=0) ** 2, axis=-1)) if t > 1 else np.zeros((0,))
    else:
        root_range = 0.0
        rot_speed = speed

    total_energy = float(speed.sum())
    denom = total_energy + 1e-8
    n = max(1, len(speed))
    early = speed[: max(1, n // 3)].sum() / denom
    mid = speed[max(1, n // 3): max(2, 2 * n // 3)].sum() / denom
    late = speed[max(2, 2 * n // 3):].sum() / denom

    out = {
        "frames": float(t),
        "dims": float(c),
        "speed_mean": float(speed.mean()) if speed.size else 0.0,
        "jump_p95": _safe_percentile(speed, 95),
        "jump_p99": _safe_percentile(speed, 99),
        "jerk_p95": _safe_percentile(jerk, 95),
        "jerk_p99": _safe_percentile(jerk, 99),
        "root_xz_range": root_range,
        "rot_speed_mean": float(rot_speed.mean()) if rot_speed.size else 0.0,
        "energy_total": total_energy,
        "energy_early_ratio": float(early),
        "energy_mid_ratio": float(mid),
        "energy_late_ratio": float(late),
        "top4_velocity_share": float(np.sort(speed)[-4:].sum() / denom) if speed.size else 0.0,
    }

    if gt is not None:
        if gt.ndim == 3:
            gt = gt[0]
        min_t = min(len(motion), len(gt))
        pred_m = motion[:min_t]
        gt_m = gt[:min_t]
        out["mse_all"] = float(np.mean((pred_m - gt_m) ** 2))
        if pred_m.shape[-1] == 151 and gt_m.shape[-1] == 151:
            out["mse_rootxz"] = float(np.mean((pred_m[:, [4, 6]] - gt_m[:, [4, 6]]) ** 2))
            out["mse_rot"] = float(np.mean((pred_m[:, 7:151] - gt_m[:, 7:151]) ** 2))
            out["mse_contact"] = float(np.mean((pred_m[:, 0:4] - gt_m[:, 0:4]) ** 2))
    return out


def _iter_pred_files(pred: Optional[str], pred_dir: Optional[str]) -> Iterable[Path]:
    if pred:
        yield Path(pred)
    if pred_dir:
        root = Path(pred_dir)
        for path in sorted(root.glob("*.npy")):
            yield path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default="", help="single predicted .npy")
    ap.add_argument("--gt", type=str, default="", help="optional single GT .npy")
    ap.add_argument("--pred_dir", type=str, default="", help="directory of predicted .npy files")
    ap.add_argument("--gt_dir", type=str, default="", help="optional GT directory; matched by file stem")
    ap.add_argument("--out_csv", type=str, default="", help="write metrics CSV")
    args = ap.parse_args()

    gt_single = _load_motion(Path(args.gt)) if args.gt else None
    gt_dir = Path(args.gt_dir) if args.gt_dir else None

    rows = []
    for pred_path in _iter_pred_files(args.pred, args.pred_dir):
        pred_motion = _load_motion(pred_path)
        gt_motion = gt_single
        if gt_motion is None and gt_dir is not None:
            candidate = gt_dir / pred_path.name
            if candidate.exists():
                gt_motion = _load_motion(candidate)
        metrics = _motion_metrics(pred_motion, gt_motion)
        metrics["file"] = str(pred_path)
        rows.append(metrics)

    if not rows:
        raise SystemExit("No prediction files found.")

    fieldnames = ["file"] + sorted(k for k in rows[0] if k != "file")
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"✅ wrote {out}")

    for row in rows:
        print("\n" + row["file"])
        for key in fieldnames:
            if key == "file":
                continue
            print(f"  {key}: {row.get(key, '')}")


if __name__ == "__main__":
    main()
