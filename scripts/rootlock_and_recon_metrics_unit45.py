import argparse
import csv
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT = slice(7, 151)

def path_len_xz(m):
    return float(np.linalg.norm(np.diff(m[:, [ROOT_X, ROOT_Z]], axis=0), axis=1).sum())

def diff_norm(m, sl):
    if len(m) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(m[:, sl], axis=0), axis=-1).mean())

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

    n = min(len(m), len(gt))
    m = m[:n]
    gt = gt[:n]

    rl = m.copy()
    rl[:, ROOT_X] = rl[0, ROOT_X]
    rl[:, ROOT_Z] = rl[0, ROOT_Z]
    Path(args.out_rootlock).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_rootlock, rl)

    row = {
        "tag": args.tag,
        "motion": args.motion,
        "T": n,
        "root_path_raw": path_len_xz(m),
        "root_path_rootlock": path_len_xz(rl),
        "rot_activity_raw": diff_norm(m, ROT),
        "rot_activity_rootlock": diff_norm(rl, ROT),
        "rot_jerk_raw": jerk(m, ROT),
        "rot_jerk_rootlock": jerk(rl, ROT),
        "mse_all_vs_gt": float(np.mean((m - gt) ** 2)),
        "mse_rot_vs_gt": float(np.mean((m[:, ROT] - gt[:, ROT]) ** 2)),
        "mse_rootxz_vs_gt": float(np.mean((m[:, [ROOT_X, ROOT_Z]] - gt[:, [ROOT_X, ROOT_Z]]) ** 2)),
        "nan_count": int(np.isnan(m).sum()),
    }

    csv_path = Path(args.metrics_csv)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print("✅ metrics:", row)

if __name__ == "__main__":
    main()
