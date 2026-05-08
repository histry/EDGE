import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT_X = 4
ROOT_Z = 6


def load_npy(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        d = arr.item()
        if "motion" in d:
            arr = d["motion"]
        elif "pose" in d:
            arr = d["pose"]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def resample(arr, T):
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == T:
        return arr
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, T)
    out = np.stack(
        [np.interp(x_new, x_old, arr[:, i]) for i in range(arr.shape[1])],
        axis=1,
    )
    return out.astype(np.float32)


def root_xz(motion):
    motion = np.asarray(motion, dtype=np.float32)

    if motion.ndim == 2 and motion.shape[1] >= 151:
        return motion[:, [ROOT_X, ROOT_Z]]

    if motion.ndim == 2 and motion.shape[1] >= 3:
        return motion[:, [0, 2]]

    raise ValueError(f"Unknown motion shape: {motion.shape}")


def ade_fde(motion, traj):
    pred = root_xz(motion)
    traj = np.asarray(traj, dtype=np.float32)

    if traj.ndim == 3:
        traj = traj[0]

    traj = traj[:, :2]
    traj = resample(traj, len(pred))

    err = np.linalg.norm(pred - traj, axis=1)
    return float(err.mean()), float(err[-1])


def jerk_root(motion):
    r = root_xz(motion)
    if len(r) < 4:
        return 0.0

    v = np.diff(r, axis=0)
    a = np.diff(v, axis=0)
    j = np.diff(a, axis=0)
    return float(np.linalg.norm(j, axis=1).mean())


def root_speed(motion):
    r = root_xz(motion)
    if len(r) < 2:
        return 0.0

    v = np.diff(r, axis=0)
    return float(np.linalg.norm(v, axis=1).mean())


def motion_energy(motion):
    motion = np.asarray(motion, dtype=np.float32)
    if len(motion) < 2:
        return 0.0

    d = np.diff(motion, axis=0)
    return float(np.linalg.norm(d, axis=1).mean())


def read_meta(path):
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def infer_case_name(final_path):
    parts = final_path.parts
    if "v10_ckpt_selection" in parts:
        i = parts.index("v10_ckpt_selection")
        if i + 1 < len(parts):
            return parts[i + 1]
    return final_path.parent.name


def evaluate_case(final_path):
    final_path = Path(final_path)
    stem = final_path.stem
    out_dir = final_path.parent

    raw_path = out_dir / f"{stem}_raw.npy"
    traj_path = out_dir / f"{stem}_target_traj.npy"
    meta_path = out_dir / f"{stem}_meta.json"

    final = load_npy(final_path)
    meta = read_meta(meta_path)

    row = {
        "case": infer_case_name(final_path),
        "final_path": str(final_path),
        "checkpoint": meta.get("checkpoint", ""),
        "post_anchor_trajectory": meta.get("post_anchor_trajectory", ""),
        "trajectory_anchor_strength": meta.get("trajectory_anchor_strength", ""),
        "sampler": meta.get("sampler", ""),
        "use_tto": meta.get("use_tto", ""),
        "has_raw": raw_path.exists(),
        "has_target_traj": traj_path.exists(),
    }

    row["final_root_speed"] = root_speed(final)
    row["final_root_jerk"] = jerk_root(final)
    row["final_motion_energy"] = motion_energy(final)

    if traj_path.exists():
        traj = load_npy(traj_path)
        ade, fde = ade_fde(final, traj)
        row["final_ADE"] = ade
        row["final_FDE"] = fde

    if raw_path.exists():
        raw = load_npy(raw_path)
        row["raw_root_speed"] = root_speed(raw)
        row["raw_root_jerk"] = jerk_root(raw)
        row["raw_motion_energy"] = motion_energy(raw)

        if traj_path.exists():
            traj = load_npy(traj_path)
            ade, fde = ade_fde(raw, traj)
            row["raw_ADE"] = ade
            row["raw_FDE"] = fde

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    files = []

    for p in root.rglob("motion.npy"):
        files.append(p)

    rows = []
    for p in sorted(files):
        try:
            rows.append(evaluate_case(p))
        except Exception as exc:
            rows.append({
                "case": infer_case_name(p),
                "final_path": str(p),
                "error": str(exc),
            })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"saved: {out_csv}")
    print(f"rows: {len(rows)}")

    for row in rows:
        print(
            row.get("case", ""),
            "raw_ADE=", row.get("raw_ADE", ""),
            "final_ADE=", row.get("final_ADE", ""),
            "raw_jerk=", row.get("raw_root_jerk", ""),
            "final_jerk=", row.get("final_root_jerk", ""),
        )


if __name__ == "__main__":
    main()
