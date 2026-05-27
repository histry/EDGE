#!/usr/bin/env python3
import argparse, csv, json, pickle, re
from pathlib import Path
import numpy as np

def load_motion_pkl(path):
    obj = pickle.load(open(path, "rb"))
    m = obj.get("motion_151", obj.get("motion"))
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    return m[:45], obj

def frame_diff(feat):
    if len(feat) < 2:
        return np.zeros((0,), dtype=np.float32)
    d = feat[1:] - feat[:-1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def pose_dist_from_start(feat):
    d = feat - feat[:1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def unit_id_from_name(path, obj):
    if "unit_index" in obj:
        return str(int(obj["unit_index"]))
    m = re.search(r"u(\d+)", Path(path).stem)
    return m.group(1) if m else Path(path).stem

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/dunhuang_bvh/support_quality_prior_pool_v15_u45")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_mid", type=float, default=0.0035)
    ap.add_argument("--min_tail", type=float, default=0.0030)
    ap.add_argument("--max_early_ratio", type=float, default=1.60)
    ap.add_argument("--min_return_ratio", type=float, default=0.45)
    ap.add_argument("--min_support_q", type=float, default=0.30)
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.pool).glob("*.pkl")):
        try:
            m, obj = load_motion_pkl(p)
            feat = m[:, 4:].astype(np.float32)

            d = frame_diff(feat)
            dist = pose_dist_from_start(feat)

            early = float(d[:6].mean()) if len(d) >= 6 else float(d.mean() if len(d) else 0.0)
            mid = float(d[8:30].mean()) if len(d) > 30 else 0.0
            tail = float(d[30:].mean()) if len(d) > 30 else 0.0

            later = max((mid + tail) * 0.5, 1e-8)
            early_ratio = early / later

            peak = float(dist.max()) if len(dist) else 0.0
            peak_frame = int(dist.argmax()) if len(dist) else 0
            final_dist = float(dist[-1]) if len(dist) else 0.0
            return_ratio = final_dist / max(peak, 1e-8)

            root = m[:, 4:7]
            rad = np.sqrt(root[:, 0] ** 2 + root[:, 2] ** 2)
            root_radius = float(rad.max())

            support_q = float(obj.get("support_prior_quality", obj.get("quality", 0.0)))
            unit = unit_id_from_name(p, obj)

            reasons = []
            if support_q < args.min_support_q:
                reasons.append("low_support_q")
            if mid < args.min_mid:
                reasons.append("low_mid")
            if tail < args.min_tail:
                reasons.append("low_tail")
            if early_ratio > args.max_early_ratio:
                reasons.append("initial_impulse")
            if peak_frame < 8 and return_ratio < args.min_return_ratio:
                reasons.append("early_peak_return")
            if root_radius > 0.12:
                reasons.append("root_not_inplace")

            gate = len(reasons) == 0
            score = (
                3.0 * mid
                + 3.0 * tail
                + 0.5 * final_dist
                + 0.8 * support_q
                - 0.8 * max(0.0, early_ratio - 1.0) * later
                - 0.2 * root_radius
            )

            rows.append({
                "file": str(p),
                "name": p.stem,
                "unit": unit,
                "support_prior_quality": support_q,
                "early_activity": early,
                "mid_activity": mid,
                "tail_activity": tail,
                "early_ratio": early_ratio,
                "peak_frame": peak_frame,
                "peak_dist": peak,
                "final_dist": final_dist,
                "return_ratio": return_ratio,
                "root_radius": root_radius,
                "prior_temporal_gate": int(gate),
                "prior_temporal_score": score,
                "reject_reasons": ";".join(reasons) if reasons else "ok",
            })
        except Exception as e:
            rows.append({
                "file": str(p),
                "name": p.stem,
                "prior_temporal_gate": 0,
                "prior_temporal_score": -999.0,
                "reject_reasons": f"error:{e}",
            })

    rows = sorted(rows, key=lambda r: float(r.get("prior_temporal_score", -999)), reverse=True)

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

    passed = [r for r in rows if str(r.get("prior_temporal_gate")) == "1"]
    print("✅ wrote", out_csv)
    print("total", len(rows), "pass", len(passed))
    print("===== top pass =====")
    for r in passed[:30]:
        print(
            r["name"],
            "unit", r["unit"],
            f'score={float(r["prior_temporal_score"]):.5f}',
            f'q={float(r["support_prior_quality"]):.3f}',
            f'early_ratio={float(r["early_ratio"]):.2f}',
            f'mid={float(r["mid_activity"]):.5f}',
            f'tail={float(r["tail_activity"]):.5f}',
            f'peak={r["peak_frame"]}',
            f'return={float(r["return_ratio"]):.2f}',
        )

if __name__ == "__main__":
    main()
