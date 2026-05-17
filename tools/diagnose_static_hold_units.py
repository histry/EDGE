#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_motion(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    if isinstance(arr, np.lib.npyio.NpzFile):
        for k in ["motion", "pred", "sample", "arr_0"]:
            if k in arr:
                arr = arr[k]
                break

    if getattr(arr, "ndim", None) == 0:
        obj = arr.item()
        if isinstance(obj, dict):
            for k in ["motion", "pred", "sample", "arr_0"]:
                if k in obj:
                    arr = obj[k]
                    break

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2 or arr.shape[-1] != 151:
        raise ValueError(f"{path}: expected [T,151] or [1,T,151], got {arr.shape}")

    return arr


def rot_indices(joints: List[int]) -> List[int]:
    out: List[int] = []
    for j in joints:
        s = 7 + 6 * int(j)
        out.extend(range(s, s + 6))
    return out


UPPER_JOINTS = list(range(12, 24))
TORSO_JOINTS = [3, 6, 9, 12, 13, 14, 15]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]

UPPER_IDX = rot_indices(UPPER_JOINTS)
TORSO_IDX = rot_indices(TORSO_JOINTS)
LOWER_IDX = rot_indices(LOWER_JOINTS)
UPPER_TORSO_IDX = sorted(set(UPPER_IDX + TORSO_IDX))


def safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


def find_gt_for_pred(pred_path: Path) -> Optional[Path]:
    # pred: e100/unit_55370.npy
    # gt:   e100/unit_55370_assets/unit_55370_gt.npy
    stem = pred_path.stem
    cands = [
        pred_path.parent / f"{stem}_assets" / f"{stem}_gt.npy",
        pred_path.parent / f"{stem}_gt.npy",
        pred_path.parent / "assets" / f"{stem}_gt.npy",
    ]
    for p in cands:
        if p.exists():
            return p
    return None


def compute_metrics(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, object]:
    if pred.shape[0] < 2:
        return {
            "active_frame_ratio": 0.0,
            "freeze_tail_ratio": 1.0,
            "static_hold_bad": True,
        }

    d_full = pred[1:] - pred[:-1]
    full_delta = safe_norm(d_full, axis=-1)

    d_ut = pred[1:, UPPER_TORSO_IDX] - pred[:-1, UPPER_TORSO_IDX]
    ut_delta = safe_norm(d_ut, axis=-1)

    d_upper = pred[1:, UPPER_IDX] - pred[:-1, UPPER_IDX]
    d_torso = pred[1:, TORSO_IDX] - pred[:-1, TORSO_IDX]
    d_lower = pred[1:, LOWER_IDX] - pred[:-1, LOWER_IDX]

    upper_delta = safe_norm(d_upper, axis=-1)
    torso_delta = safe_norm(d_torso, axis=-1)
    lower_delta = safe_norm(d_lower, axis=-1)

    if gt is not None and gt.shape == pred.shape:
        gt_ut = gt[1:, UPPER_TORSO_IDX] - gt[:-1, UPPER_TORSO_IDX]
        gt_ut_delta = safe_norm(gt_ut, axis=-1)

        gt_mean = float(np.mean(gt_ut_delta) + 1e-8)
        motion_energy_ratio_to_gt = float(np.mean(ut_delta) / gt_mean)

        # Threshold tied to GT motion scale; catches "one jump then hold".
        active_thr = max(1e-5, 0.20 * gt_mean)

        gt_tail = gt_ut_delta[len(gt_ut_delta) // 2:]
        gt_tail_mean = float(np.mean(gt_tail) + 1e-8) if len(gt_tail) else gt_mean
        pred_tail_mean = float(np.mean(ut_delta[len(ut_delta) // 2:])) if len(ut_delta) else 0.0
        tail_energy_ratio_to_gt = float(pred_tail_mean / gt_tail_mean)
    else:
        gt_mean = float("nan")
        motion_energy_ratio_to_gt = float("nan")
        tail_energy_ratio_to_gt = float("nan")

        # Fallback threshold from prediction distribution.
        active_thr = max(1e-5, 0.15 * float(np.percentile(ut_delta, 75)))

    active = ut_delta > active_thr
    active_frame_ratio = float(np.mean(active)) if len(active) else 0.0

    half = len(active) // 2
    tail_active = active[half:] if len(active) else np.array([], dtype=bool)
    freeze_tail_ratio = float(1.0 - np.mean(tail_active)) if len(tail_active) else 1.0

    first_quarter = max(1, len(ut_delta) // 4)
    early_jump_max = float(np.max(ut_delta[:first_quarter])) if len(ut_delta) else 0.0
    max_jump = float(np.max(ut_delta)) if len(ut_delta) else 0.0
    jump_p95 = float(np.percentile(ut_delta, 95)) if len(ut_delta) else 0.0

    jerk = ut_delta[1:] - ut_delta[:-1] if len(ut_delta) > 1 else np.zeros((0,), dtype=np.float32)
    jerk_abs = np.abs(jerk)
    jerk_p95 = float(np.percentile(jerk_abs, 95)) if len(jerk_abs) else 0.0

    mean_delta = float(np.mean(ut_delta)) if len(ut_delta) else 0.0
    tail_mean_delta = float(np.mean(ut_delta[half:])) if len(ut_delta[half:]) else 0.0

    # Main bad-case rule:
    # one transition then static hold, or generated motion much weaker than GT.
    bad_reasons: List[str] = []

    if active_frame_ratio < 0.35:
        bad_reasons.append("low_active_frame_ratio")
    if freeze_tail_ratio > 0.70:
        bad_reasons.append("high_freeze_tail_ratio")
    if not np.isnan(motion_energy_ratio_to_gt) and motion_energy_ratio_to_gt < 0.35:
        bad_reasons.append("low_motion_energy_ratio_to_gt")
    if not np.isnan(tail_energy_ratio_to_gt) and tail_energy_ratio_to_gt < 0.35:
        bad_reasons.append("low_tail_energy_ratio_to_gt")

    static_hold_bad = len(bad_reasons) > 0

    return {
        "active_frame_ratio": active_frame_ratio,
        "freeze_tail_ratio": freeze_tail_ratio,
        "motion_energy_ratio_to_gt": motion_energy_ratio_to_gt,
        "tail_energy_ratio_to_gt": tail_energy_ratio_to_gt,
        "mean_delta_upper_torso": mean_delta,
        "tail_mean_delta_upper_torso": tail_mean_delta,
        "upper_active_mean": float(np.mean(upper_delta)) if len(upper_delta) else 0.0,
        "torso_active_mean": float(np.mean(torso_delta)) if len(torso_delta) else 0.0,
        "lower_active_mean": float(np.mean(lower_delta)) if len(lower_delta) else 0.0,
        "full_mean_delta": float(np.mean(full_delta)) if len(full_delta) else 0.0,
        "max_jump_upper_torso": max_jump,
        "early_jump_max_upper_torso": early_jump_max,
        "jump_p95_upper_torso": jump_p95,
        "jerk_p95_upper_torso": jerk_p95,
        "active_thr": active_thr,
        "static_hold_bad": static_hold_bad,
        "bad_reasons": "|".join(bad_reasons),
    }


def should_skip(path: Path) -> bool:
    name = path.name
    if "_assets" in str(path):
        return True
    if name.endswith("_raw.npy"):
        return True
    if name.endswith("_target_traj.npy"):
        return True
    if name.endswith("_start.npy") or name.endswith("_end.npy") or name.endswith("_gt.npy"):
        return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_files = [
        p for p in sorted(pred_dir.glob("unit_*.npy"))
        if p.is_file() and not should_skip(p)
    ]

    rows: List[Dict[str, object]] = []

    for pred_path in pred_files:
        unit = pred_path.stem.replace("unit_", "")
        try:
            pred = load_motion(pred_path)
        except Exception as exc:
            rows.append({
                "unit": unit,
                "file": str(pred_path),
                "error": str(exc),
                "static_hold_bad": True,
                "bad_reasons": "load_error",
            })
            continue

        gt_path = find_gt_for_pred(pred_path)
        gt = None
        if gt_path is not None:
            try:
                gt = load_motion(gt_path)
            except Exception:
                gt = None

        row: Dict[str, object] = {
            "unit": unit,
            "file": str(pred_path),
            "gt_file": str(gt_path) if gt_path else "",
        }
        row.update(compute_metrics(pred, gt))
        rows.append(row)

    fieldnames = [
        "unit",
        "file",
        "gt_file",
        "static_hold_bad",
        "bad_reasons",
        "active_frame_ratio",
        "freeze_tail_ratio",
        "motion_energy_ratio_to_gt",
        "tail_energy_ratio_to_gt",
        "mean_delta_upper_torso",
        "tail_mean_delta_upper_torso",
        "upper_active_mean",
        "torso_active_mean",
        "lower_active_mean",
        "full_mean_delta",
        "max_jump_upper_torso",
        "early_jump_max_upper_torso",
        "jump_p95_upper_torso",
        "jerk_p95_upper_torso",
        "active_thr",
        "error",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"saved: {out_csv}")
    print("unit,bad,reasons,active,freeze,energy_ratio,tail_ratio")
    for r in rows:
        print(
            f"{r.get('unit')},"
            f"{r.get('static_hold_bad')},"
            f"{r.get('bad_reasons','')},"
            f"{float(r.get('active_frame_ratio', 0) or 0):.3f},"
            f"{float(r.get('freeze_tail_ratio', 0) or 0):.3f},"
            f"{float(r.get('motion_energy_ratio_to_gt', float('nan'))):.3f},"
            f"{float(r.get('tail_energy_ratio_to_gt', float('nan'))):.3f}"
        )


if __name__ == "__main__":
    main()
