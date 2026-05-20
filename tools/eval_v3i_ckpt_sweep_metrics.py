#!/usr/bin/env python3
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from functional_choreo_metrics import functional_choreo_stats


def load_batch(path):
    arr = np.load(path, allow_pickle=True).astype(np.float32)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"bad shape {arr.shape}: {path}")


def jerk_p95(m):
    if len(m) < 4:
        return 0.0
    j = m[3:] - 3*m[2:-1] + 3*m[1:-2] - m[:-3]
    frame = np.sqrt(np.mean(j*j, axis=1))
    return float(np.percentile(frame, 95))


def jump_p95(m):
    if len(m) < 2:
        return 0.0
    d = m[1:] - m[:-1]
    frame = np.sqrt(np.mean(d*d, axis=1))
    return float(np.percentile(frame, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.input_dir).glob("*samples.npy")):
        epoch = f.stem.replace("v3i_e", "").replace("_samples", "")
        batch = load_batch(f)
        for bi, m in enumerate(batch):
            s = functional_choreo_stats(m)
            row = {
                "file": str(f),
                "epoch": int(epoch) if epoch.isdigit() else epoch,
                "sample": bi,
                "jump_p95": jump_p95(m),
                "jerk_p95": jerk_p95(m),
            }
            row.update(s)
            rows.append(row)

    if not rows:
        raise RuntimeError("No samples found.")

    keys = sorted({k for r in rows for k in r.keys()})
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    Path(args.out_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ wrote {out_csv}, rows={len(rows)}")
    print("epoch,sample,root_path,lower,torso,upper,expr,contact_switch,jump_p95,jerk_p95")
    for r in rows:
        print(
            f'{r["epoch"]},{r["sample"]},'
            f'{r.get("root_path",0):.6f},'
            f'{r.get("lower_activity",0):.6f},'
            f'{r.get("torso_activity",0):.6f},'
            f'{r.get("upper_activity",0):.6f},'
            f'{r.get("expression_activity",0):.6f},'
            f'{r.get("contact_switch",0):.6f},'
            f'{r.get("jump_p95",0):.6f},'
            f'{r.get("jerk_p95",0):.6f}'
        )


if __name__ == "__main__":
    main()
