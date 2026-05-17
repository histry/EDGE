#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np

UPPER_JOINTS = list(range(12, 24))
TORSO_JOINTS = [3, 6, 9, 12, 13, 14, 15]

def rot_indices(joints):
    out = []
    for joint in joints:
        s = 7 + 6 * int(joint)
        out.extend(range(s, s + 6))
    return out

IDX = sorted(set(rot_indices(UPPER_JOINTS) + rot_indices(TORSO_JOINTS)))

def load_motion(path: Path):
    arr = np.load(path, allow_pickle=True)
    if getattr(arr, "ndim", None) == 0:
        obj = arr.item()
        if isinstance(obj, dict):
            for key in ["motion", "pred", "sample", "arr_0"]:
                if key in obj:
                    arr = obj[key]
                    break
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {arr.shape}")
    return arr

def norm(x):
    return np.sqrt(np.sum(x * x, axis=-1) + 1e-8)

def skip(path: Path):
    name = path.name
    return ("_assets" in str(path) or name.endswith("_raw.npy") or name.endswith("_target_traj.npy")
            or name.endswith("_gt.npy") or name.endswith("_start.npy") or name.endswith("_end.npy"))

def metrics(x):
    feat = x[:, IDX]
    v = norm(feat[1:] - feat[:-1])
    a = np.diff(v) if len(v) > 1 else np.zeros((0,), dtype=np.float32)
    j = np.diff(a) if len(a) > 1 else np.zeros((0,), dtype=np.float32)
    if len(v) == 0:
        return {"burst_jitter_bad": True, "bad_reasons": "too_short"}
    med = float(np.median(v))
    p75 = float(np.percentile(v, 75))
    p95 = float(np.percentile(v, 95))
    vmax = float(np.max(v))
    thr = max(p75 * 1.8, med * 3.0, 1e-5)
    mask = v > thr
    burst_count = int(np.sum(mask))
    groups, prev = 0, False
    for b in mask:
        if bool(b) and not prev:
            groups += 1
        prev = bool(b)
    top4 = float(np.sort(v)[::-1][:4].sum() / (v.sum() + 1e-8))
    top8 = float(np.sort(v)[::-1][:8].sum() / (v.sum() + 1e-8))
    acc_p95 = float(np.percentile(np.abs(a), 95)) if len(a) else 0.0
    jerk_p95 = float(np.percentile(np.abs(j), 95)) if len(j) else 0.0
    reasons = []
    if burst_count >= 4: reasons.append("burst_count>=4")
    if groups >= 2: reasons.append("multi_burst_groups")
    if top4 > 0.55: reasons.append("top4_velocity_dominates")
    if p95 / max(med, 1e-6) > 6.0: reasons.append("high_p95_to_median_velocity")
    if jerk_p95 > 50.0: reasons.append("high_jerk_p95")
    return {
        "burst_jitter_bad": bool(reasons),
        "bad_reasons": "|".join(reasons),
        "vel_median": med, "vel_p75": p75, "vel_p95": p95, "vel_max": vmax,
        "p95_to_median": p95 / max(med, 1e-6),
        "burst_count": burst_count, "burst_groups": groups,
        "top4_velocity_share": top4, "top8_velocity_share": top8,
        "acc_p95": acc_p95, "jerk_p95": jerk_p95,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)
    rows = []
    for path in sorted(pred_dir.glob("unit_*.npy")):
        if not path.is_file() or skip(path):
            continue
        unit = path.stem.replace("unit_", "")
        row = {"unit": unit, "file": str(path)}
        try:
            row.update(metrics(load_motion(path)))
        except Exception as exc:
            row.update({"burst_jitter_bad": True, "bad_reasons": f"error:{exc}"})
        rows.append(row)
    fields = ["unit","file","burst_jitter_bad","bad_reasons","vel_median","vel_p75","vel_p95","vel_max","p95_to_median","burst_count","burst_groups","top4_velocity_share","top8_velocity_share","acc_p95","jerk_p95"]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    print("saved:", out)
    print("unit,bad,reasons,burst_count,groups,top4,top8,jerk_p95")
    for row in rows:
        print(row["unit"], row.get("burst_jitter_bad"), row.get("bad_reasons",""), row.get("burst_count",""), row.get("burst_groups",""), round(float(row.get("top4_velocity_share",0) or 0),3), round(float(row.get("top8_velocity_share",0) or 0),3), round(float(row.get("jerk_p95",0) or 0),3), sep=",")

if __name__ == "__main__":
    main()
