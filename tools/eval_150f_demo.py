#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from functional_choreo_metrics import functional_choreo_stats

def load_motion(path):
    arr = np.load(path, allow_pickle=True).astype("float32")
    if arr.ndim == 3:
        arr = arr[0]
    return arr

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
    return float(np.sqrt(np.mean(d*d)))

def segment_stats(m, seg_len=45):
    out = []
    start = 0
    idx = 0
    while start < len(m):
        end = min(len(m), start + seg_len)
        seg = m[start:end]
        s = functional_choreo_stats(seg)
        s["idx"] = idx
        s["start"] = start
        s["end"] = end
        s["frame_motion"] = frame_motion(seg)
        s["jump_p95"] = jump_p95(seg)
        s["jerk_p95"] = jerk_p95(seg)
        out.append(s)
        start += seg_len
        idx += 1
    return out

def boundary_metrics(m, boundaries, window=5):
    d = np.sqrt(np.mean((m[1:] - m[:-1]) ** 2, axis=1)) if len(m) > 1 else np.zeros((0,))
    rows = []
    for b in boundaries:
        lo = max(0, b - window)
        hi = min(len(d), b + window)
        rows.append({
            "boundary": int(b),
            "local_jump_mean": float(d[lo:hi].mean()) if hi > lo else 0.0,
            "local_jump_max": float(d[lo:hi].max()) if hi > lo else 0.0,
            "contact_l1": float(np.mean(np.abs(m[max(0,b-1),0:4] - m[min(len(m)-1,b),0:4]))),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--stitch_report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    m = load_motion(args.motion)
    stitch = json.loads(Path(args.stitch_report).read_text(encoding="utf-8"))
    boundaries = stitch.get("boundaries", [])

    root = m[:, 4:7]
    rad = np.sqrt(root[:,0] ** 2 + root[:,2] ** 2)

    stats = functional_choreo_stats(m)
    report = {
        "motion": args.motion,
        "frames": int(len(m)),
        "global_stats": stats,
        "jump_p95": jump_p95(m),
        "jerk_p95": jerk_p95(m),
        "frame_motion": frame_motion(m),
        "root_max_radius": float(rad.max()),
        "root_final_xz": root[-1,[0,2]].tolist(),
        "segment_stats_45f": segment_stats(m, seg_len=45),
        "boundaries": boundaries,
        "boundary_metrics": boundary_metrics(m, boundaries),
        "selected_units": stitch.get("selected", []),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ wrote", args.out)
    print("frames:", report["frames"])
    print("root_max_radius:", f'{report["root_max_radius"]:.5f}')
    print("root_path:", f'{stats.get("root_path",0):.5f}')
    print("lower:", f'{stats.get("lower_activity",0):.5f}')
    print("upper:", f'{stats.get("upper_activity",0):.5f}')
    print("contact:", f'{stats.get("contact_switch",0):.5f}')
    print("jump:", f'{report["jump_p95"]:.5f}')
    print("jerk:", f'{report["jerk_p95"]:.5f}')

    print("boundary metrics:")
    for b in report["boundary_metrics"]:
        print(b)

if __name__ == "__main__":
    main()
