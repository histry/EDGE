import argparse
import csv
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def path_len_xz(m):
    xz = m[:, [ROOT_X, ROOT_Z]]
    return float(np.linalg.norm(np.diff(xz, axis=0), axis=1).sum())

def activity(m, sl):
    d = np.diff(m[:, sl], axis=0)
    return float(np.linalg.norm(d, axis=-1).mean())

def jerk(m, sl):
    if len(m) < 3:
        return 0.0
    j = m[2:, sl] - 2 * m[1:-1, sl] + m[:-2, sl]
    return float(np.linalg.norm(j, axis=-1).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out_rootlock", required=True)
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    m = np.load(args.motion).astype(np.float32)
    gt = np.load(args.gt).astype(np.float32)

    rootlocked = m.copy()
    rootlocked[:, ROOT_X] = rootlocked[0, ROOT_X]
    rootlocked[:, ROOT_Z] = rootlocked[0, ROOT_Z]
    Path(args.out_rootlock).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_rootlock, rootlocked.astype(np.float32))

    n = min(len(m), len(gt))
    m2 = m[:n]
    gt2 = gt[:n]
    rl2 = rootlocked[:n]

    row = {
        "tag": args.tag,
        "motion": args.motion,
        "root_path_raw": path_len_xz(m2),
        "root_path_rootlock": path_len_xz(rl2),
        "rot_activity_raw": activity(m2, ROT),
        "rot_activity_rootlock": activity(rl2, ROT),
        "rot_jerk_raw": jerk(m2, ROT),
        "rot_jerk_rootlock": jerk(rl2, ROT),
        "mse_vs_gt_all": float(np.mean((m2 - gt2) ** 2)),
        "mse_vs_gt_rot": float(np.mean((m2[:, ROT] - gt2[:, ROT]) ** 2)),
        "nan_count": int(np.isnan(m2).sum()),
    }

    csv_path = Path(args.metrics_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("✅ rootlock saved:", args.out_rootlock)
    print("✅ metrics appended:", args.metrics_csv)
    print(row)

if __name__ == "__main__":
    main()
