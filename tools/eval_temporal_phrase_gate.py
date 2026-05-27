#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

def load_motion(path):
    arr = np.load(path, allow_pickle=True).astype("float32")
    if arr.ndim == 3:
        arr = arr[0]
    return arr

def frame_diff(feat):
    if len(feat) < 2:
        return np.zeros((0,), dtype=np.float32)
    d = feat[1:] - feat[:-1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def pose_dist_from_start(feat):
    d = feat - feat[:1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safe_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--early_frames", type=int, default=6)
    ap.add_argument("--mid_start", type=int, default=8)
    ap.add_argument("--tail_start", type=int, default=30)
    ap.add_argument("--min_mid_activity", type=float, default=0.0040)
    ap.add_argument("--min_tail_activity", type=float, default=0.0035)
    ap.add_argument("--max_early_ratio", type=float, default=2.4)
    ap.add_argument("--min_return_ratio", type=float, default=0.35)
    ap.add_argument("--max_peak_frame", type=int, default=14)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.safe_csv, encoding="utf-8")))
    out_rows = []

    for r in rows:
        f = Path(r["file"])
        m = load_motion(f)

        # 不用 contacts，只看 root+rotations，避免 contact 数值误导。
        feat = m[:, 4:].astype(np.float32)

        d = frame_diff(feat)
        dist = pose_dist_from_start(feat)

        early = float(d[:args.early_frames].mean()) if len(d) >= args.early_frames else float(d.mean() if len(d) else 0.0)
        mid = float(d[args.mid_start:args.tail_start].mean()) if len(d) > args.mid_start else 0.0
        tail = float(d[args.tail_start:].mean()) if len(d) > args.tail_start else 0.0

        later = max((mid + tail) * 0.5, 1e-8)
        early_ratio = early / later

        peak = float(dist.max()) if len(dist) else 0.0
        peak_frame = int(dist.argmax()) if len(dist) else 0
        final_dist = float(dist[-1]) if len(dist) else 0.0
        return_ratio = final_dist / max(peak, 1e-8)

        sampling_ok = str(r.get("sampling_safe_gate", "0")) == "1"

        reasons = []
        if not sampling_ok:
            reasons.append("sampling_safe_reject")
        if mid < args.min_mid_activity:
            reasons.append("low_mid_activity")
        if tail < args.min_tail_activity:
            reasons.append("low_tail_activity")
        if early_ratio > args.max_early_ratio:
            reasons.append("initial_impulse_dominates")
        if peak_frame <= args.max_peak_frame and return_ratio < args.min_return_ratio and peak > 0.006:
            reasons.append("early_peak_return_to_start")

        phrase_ok = len(reasons) == 0

        # 新分数：鼓励中后段持续动作，惩罚开头脉冲和回原点。
        phrase_score = (
            3.0 * mid
            + 3.0 * tail
            + 0.5 * final_dist
            - 1.2 * max(0.0, early_ratio - 1.0) * later
            + 0.5 * return_ratio * peak
        )

        rr = dict(r)
        rr.update({
            "early_activity": early,
            "mid_activity": mid,
            "tail_activity": tail,
            "early_ratio": early_ratio,
            "peak_dist": peak,
            "peak_frame": peak_frame,
            "final_dist": final_dist,
            "return_ratio": return_ratio,
            "temporal_phrase_gate": int(phrase_ok),
            "temporal_phrase_score": phrase_score,
            "temporal_reject_reasons": ";".join(reasons) if reasons else "ok",
        })
        out_rows.append(rr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    keys = sorted({k for r in out_rows for k in r.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)

    Path(args.out_json).write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ wrote", out_csv, "rows", len(out_rows))

    print("===== TEMPORAL PASS =====")
    for r in sorted(out_rows, key=lambda x: safe_float(x.get("temporal_phrase_score")), reverse=True):
        if str(r.get("temporal_phrase_gate")) == "1":
            print(
                r["name"],
                f'phrase={safe_float(r.get("temporal_phrase_score")):.6f}',
                f'early={safe_float(r.get("early_activity")):.5f}',
                f'mid={safe_float(r.get("mid_activity")):.5f}',
                f'tail={safe_float(r.get("tail_activity")):.5f}',
                f'early_ratio={safe_float(r.get("early_ratio")):.2f}',
                f'return={safe_float(r.get("return_ratio")):.2f}',
                f'peak_frame={r.get("peak_frame")}',
                "ok"
            )

    print("===== TEMPORAL REJECT top examples =====")
    for r in sorted(out_rows, key=lambda x: safe_float(x.get("sampling_safe_score")), reverse=True)[:24]:
        if str(r.get("temporal_phrase_gate")) != "1":
            print(
                r["name"],
                f'safe={safe_float(r.get("sampling_safe_score")):.6f}',
                f'early_ratio={safe_float(r.get("early_ratio")):.2f}',
                f'return={safe_float(r.get("return_ratio")):.2f}',
                f'peak_frame={r.get("peak_frame")}',
                "reason=" + r.get("temporal_reject_reasons", "")
            )

if __name__ == "__main__":
    main()
