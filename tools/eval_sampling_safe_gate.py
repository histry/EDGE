#!/usr/bin/env python3
import argparse, csv, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functional_choreo_metrics import functional_choreo_stats

def jump_p95(m):
    if len(m) < 2:
        return 0.0
    d = m[1:] - m[:-1]
    return float(np.percentile(np.sqrt(np.mean(d*d, axis=1)), 95))

def jerk_p95(m):
    if len(m) < 4:
        return 0.0
    j = m[3:] - 3*m[2:-1] + 3*m[1:-2] - m[:-3]
    return float(np.percentile(np.sqrt(np.mean(j*j, axis=1)), 95))

def frame_motion(m):
    if len(m) < 2:
        return 0.0
    d = m[1:] - m[:-1]
    return float(np.sqrt(np.mean(d * d)))

def parse_unit(name):
    m = re.search(r"unit(\d+)|u(\d+)|supportq_u(\d+)", name)
    if not m:
        return ""
    for g in m.groups():
        if g:
            return str(int(g))
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_root", type=float, default=0.0040)
    ap.add_argument("--min_upper", type=float, default=0.0055)
    ap.add_argument("--min_lower", type=float, default=0.0055)
    ap.add_argument("--min_frame_motion", type=float, default=0.0045)
    ap.add_argument("--max_jump", type=float, default=0.020)
    ap.add_argument("--max_jerk", type=float, default=0.045)
    args = ap.parse_args()

    rows = []
    for f in sorted(Path(args.input_dir).glob("*.npy")):
        arr = np.load(f, allow_pickle=True).astype("float32")
        if arr.ndim == 3:
            arr = arr[0]

        s = functional_choreo_stats(arr)
        r = {
            "name": f.stem,
            "file": str(f),
            "unit": parse_unit(f.stem),
            "jump_p95": jump_p95(arr),
            "jerk_p95": jerk_p95(arr),
            "frame_motion": frame_motion(arr),
        }
        r.update(s)

        root = float(r.get("root_path", 0.0))
        lower = float(r.get("lower_activity", 0.0))
        upper = float(r.get("upper_activity", 0.0))
        contact = float(r.get("contact_switch", 0.0))
        jump = float(r.get("jump_p95", 0.0))
        jerk = float(r.get("jerk_p95", 0.0))
        fm = float(r.get("frame_motion", 0.0))

        reasons = []
        if root < args.min_root:
            reasons.append("low_root")
        if max(lower, upper) < max(args.min_lower, args.min_upper):
            reasons.append("low_body_activity")
        if fm < args.min_frame_motion:
            reasons.append("low_frame_motion")
        if jump > args.max_jump:
            reasons.append("large_jump")
        if jerk > args.max_jerk:
            reasons.append("large_jerk")
        if contact > 0 and max(lower, upper) < 0.0045:
            reasons.append("contact_without_visible_body_motion")

        passed = len(reasons) == 0

        # 新 safe score：必须先能动，再谈平滑
        r["sampling_safe_gate"] = int(passed)
        r["reject_reasons"] = ";".join(reasons) if reasons else "ok"
        r["sampling_safe_score"] = (
            2.5 * lower
            + 2.5 * upper
            + 1.0 * root
            + 0.5 * contact
            + 1.0 * fm
            - 0.5 * jump
            - 0.5 * jerk
        )
        rows.append(r)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for r in rows for k in r})
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    Path(args.out_json).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ wrote", out_csv, "rows", len(rows))
    print("===== PASS =====")
    for r in sorted(rows, key=lambda x: x["sampling_safe_score"], reverse=True):
        if r["sampling_safe_gate"]:
            print(
                r["name"],
                f'score={r["sampling_safe_score"]:.6f}',
                f'root={r.get("root_path",0):.5f}',
                f'lower={r.get("lower_activity",0):.5f}',
                f'upper={r.get("upper_activity",0):.5f}',
                f'contact={r.get("contact_switch",0):.5f}',
                f'jump={r.get("jump_p95",0):.5f}',
                f'jerk={r.get("jerk_p95",0):.5f}',
                "ok"
            )

    print("===== REJECT top examples =====")
    for r in sorted(rows, key=lambda x: x["sampling_safe_score"], reverse=True)[:20]:
        if not r["sampling_safe_gate"]:
            print(
                r["name"],
                f'score={r["sampling_safe_score"]:.6f}',
                f'root={r.get("root_path",0):.5f}',
                f'lower={r.get("lower_activity",0):.5f}',
                f'upper={r.get("upper_activity",0):.5f}',
                f'contact={r.get("contact_switch",0):.5f}',
                "reason=" + r["reject_reasons"]
            )

if __name__ == "__main__":
    main()
