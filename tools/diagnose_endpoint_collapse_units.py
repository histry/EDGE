#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def load_motion(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
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
        raise ValueError(f"{path}: expected [T,151], got {arr.shape}")
    return arr


def rot_indices(joints: List[int]) -> List[int]:
    out = []
    for j in joints:
        start = 7 + 6 * int(j)
        out.extend(range(start, start + 6))
    return out


UPPER_JOINTS = list(range(12, 24))
TORSO_JOINTS = [3, 6, 9, 12, 13, 14, 15]
UPPER_TORSO_IDX = sorted(set(rot_indices(UPPER_JOINTS) + rot_indices(TORSO_JOINTS)))


def safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


def find_gt_for_pred(pred_path: Path) -> Optional[Path]:
    stem = pred_path.stem
    for p in [pred_path.parent / f"{stem}_assets" / f"{stem}_gt.npy", pred_path.parent / f"{stem}_gt.npy"]:
        if p.exists():
            return p
    return None


def should_skip(path: Path) -> bool:
    name = path.name
    return (
        "_assets" in str(path)
        or name.endswith("_raw.npy")
        or name.endswith("_target_traj.npy")
        or name.endswith("_start.npy")
        or name.endswith("_end.npy")
        or name.endswith("_gt.npy")
    )


def compute(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, object]:
    T = pred.shape[0]
    feat = pred[:, UPPER_TORSO_IDX]
    start = feat[0]
    end = feat[-1]
    dist_end = safe_norm(feat - end[None, :], axis=-1)
    start_end_dist = float(safe_norm(end - start, axis=-1))
    delta = safe_norm(feat[1:] - feat[:-1], axis=-1)
    mean_delta = float(delta.mean()) if len(delta) else 0.0
    max_delta = float(delta.max()) if len(delta) else 0.0
    eps_end = max(0.08 * start_end_dist, 1.5 * mean_delta, 1e-5)
    near_end = dist_end <= eps_end
    reach_frames = np.where(near_end)[0]
    if len(reach_frames):
        end_reach_frame = int(reach_frames[0])
        end_reach_frac = float(end_reach_frame / max(T - 1, 1))
        end_hold_ratio = float(np.mean(near_end[end_reach_frame:]))
    else:
        end_reach_frame = -1
        end_reach_frac = 1.0
        end_hold_ratio = 0.0

    total_motion = float(delta.sum() + 1e-8)
    cumsum = np.cumsum(delta) / total_motion if len(delta) else np.zeros((0,), dtype=np.float32)

    def cum_at(frac: float) -> float:
        if len(cumsum) == 0:
            return 0.0
        idx = min(len(cumsum) - 1, max(0, int(round(frac * (len(cumsum) - 1)))))
        return float(cumsum[idx])

    cum_25 = cum_at(0.25)
    cum_50 = cum_at(0.50)
    cum_75 = cum_at(0.75)
    sorted_delta = np.sort(delta)[::-1] if len(delta) else np.zeros((0,), dtype=np.float32)
    top1_share = float(sorted_delta[:1].sum() / total_motion) if len(sorted_delta) else 0.0
    top3_share = float(sorted_delta[:3].sum() / total_motion) if len(sorted_delta) else 0.0
    top5_share = float(sorted_delta[:5].sum() / total_motion) if len(sorted_delta) else 0.0

    gt_cum_25 = gt_cum_50 = gt_cum_75 = np.nan
    progress_l1 = np.nan
    if gt is not None and gt.shape == pred.shape:
        gt_feat = gt[:, UPPER_TORSO_IDX]
        gt_delta = safe_norm(gt_feat[1:] - gt_feat[:-1], axis=-1)
        gt_total = float(gt_delta.sum() + 1e-8)
        gt_cumsum = np.cumsum(gt_delta) / gt_total if len(gt_delta) else np.zeros((0,), dtype=np.float32)
        if len(gt_cumsum) == len(cumsum) and len(cumsum):
            progress_l1 = float(np.mean(np.abs(cumsum - gt_cumsum)))
            gt_cum_25 = float(gt_cumsum[min(len(gt_cumsum)-1, int(round(0.25 * (len(gt_cumsum)-1))))])
            gt_cum_50 = float(gt_cumsum[min(len(gt_cumsum)-1, int(round(0.50 * (len(gt_cumsum)-1))))])
            gt_cum_75 = float(gt_cumsum[min(len(gt_cumsum)-1, int(round(0.75 * (len(gt_cumsum)-1))))])

    reasons: List[str] = []
    if end_reach_frac < 0.35 and end_hold_ratio > 0.60:
        reasons.append("early_reach_end_and_hold")
    if cum_25 > 0.55:
        reasons.append("front_loaded_motion_cum25")
    if cum_50 > 0.75:
        reasons.append("front_loaded_motion_cum50")
    if top1_share > 0.35:
        reasons.append("single_jump_dominates")
    if top3_share > 0.60:
        reasons.append("top3_jumps_dominate")
    if not np.isnan(progress_l1) and progress_l1 > 0.25:
        reasons.append("temporal_progress_mismatch_gt")

    return {
        "endpoint_collapse_bad": len(reasons) > 0,
        "bad_reasons": "|".join(reasons),
        "end_reach_frame": end_reach_frame,
        "end_reach_frac": end_reach_frac,
        "end_hold_ratio": end_hold_ratio,
        "eps_end": eps_end,
        "start_end_dist": start_end_dist,
        "mean_delta": mean_delta,
        "max_delta": max_delta,
        "cum_motion_25": cum_25,
        "cum_motion_50": cum_50,
        "cum_motion_75": cum_75,
        "gt_cum_motion_25": gt_cum_25,
        "gt_cum_motion_50": gt_cum_50,
        "gt_cum_motion_75": gt_cum_75,
        "progress_l1_vs_gt": progress_l1,
        "top1_delta_share": top1_share,
        "top3_delta_share": top3_share,
        "top5_delta_share": top5_share,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    pred_dir = Path(args.pred_dir)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(pred_dir.glob("unit_*.npy")) if p.is_file() and not should_skip(p)]
    rows: List[Dict[str, object]] = []
    for p in files:
        unit = p.stem.replace("unit_", "")
        gt_path = find_gt_for_pred(p)
        try:
            pred = load_motion(p)
            gt = load_motion(gt_path) if gt_path and gt_path.exists() else None
            row = {"unit": unit, "file": str(p), "gt_file": str(gt_path) if gt_path else ""}
            row.update(compute(pred, gt))
        except Exception as exc:
            row = {"unit": unit, "file": str(p), "gt_file": str(gt_path) if gt_path else "", "endpoint_collapse_bad": True, "bad_reasons": f"error:{exc}"}
        rows.append(row)

    fieldnames = ["unit", "file", "gt_file", "endpoint_collapse_bad", "bad_reasons", "end_reach_frame", "end_reach_frac", "end_hold_ratio", "eps_end", "start_end_dist", "mean_delta", "max_delta", "cum_motion_25", "cum_motion_50", "cum_motion_75", "gt_cum_motion_25", "gt_cum_motion_50", "gt_cum_motion_75", "progress_l1_vs_gt", "top1_delta_share", "top3_delta_share", "top5_delta_share"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"saved: {out_csv}")
    print("unit,bad,reasons,end_reach_frac,end_hold,cum25,cum50,top1,top3,progress_l1")
    for r in rows:
        print(f"{r.get('unit')},{r.get('endpoint_collapse_bad')},{r.get('bad_reasons','')},{float(r.get('end_reach_frac', 0) or 0):.3f},{float(r.get('end_hold_ratio', 0) or 0):.3f},{float(r.get('cum_motion_25', 0) or 0):.3f},{float(r.get('cum_motion_50', 0) or 0):.3f},{float(r.get('top1_delta_share', 0) or 0):.3f},{float(r.get('top3_delta_share', 0) or 0):.3f},{float(r.get('progress_l1_vs_gt', 0) or 0):.3f}")


if __name__ == "__main__":
    main()
