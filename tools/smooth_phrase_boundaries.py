import argparse
import json
from pathlib import Path
import numpy as np

ROOT_X = 4
ROOT_Z = 6
ROT_SLICE = slice(7, 151)

def load_motion(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        obj = arr.item()
        arr = obj.get("motion", obj.get("motion_151", obj.get("pose", arr)))
    arr = np.asarray(arr, dtype=np.float32)
    squeeze = False
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
        squeeze = True
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"expected [T,151] or [1,T,151], got {arr.shape}")
    return arr, squeeze

def save_motion(path, motion, squeeze):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = motion[None] if squeeze else motion
    np.save(path, out.astype(np.float32))

def gaussian_smooth(x, sigma=2.0):
    radius = max(1, int(round(sigma * 3)))
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(grid ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    pad = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = (pad[i:i + len(kernel)] * kernel[:, None]).sum(axis=0)
    return y

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT_SLICE]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1)
    out = {
        "frames": int(len(m)),
        "root_max_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "global_root_jump_p95": float(np.percentile(droot, 95)) if len(droot) else 0.0,
        "global_rot_jump_p95": float(np.percentile(drot, 95)) if len(drot) else 0.0,
        "segment_activity_mean": float(drot.mean()) if len(drot) else 0.0,
    }
    for b in [35, 70, 74, 105, 108, 140, 142]:
        if 2 <= b < len(m) - 2:
            lo = max(1, b - 2)
            hi = min(len(m), b + 3)
            local = np.linalg.norm(rot[lo:hi] - rot[lo-1:hi-1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local.max())
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--boundaries", default="35,74,105,108")
    ap.add_argument("--radius", type=int, default=7)
    ap.add_argument("--sigma", type=float, default=2.2)
    ap.add_argument("--strength", type=float, default=0.70)
    args = ap.parse_args()

    motion, squeeze = load_motion(args.input)
    out = motion.copy()

    boundaries = [int(x) for x in args.boundaries.replace(";", ",").split(",") if x.strip()]
    radius = int(args.radius)
    strength = float(np.clip(args.strength, 0.0, 1.0))

    rot = out[:, ROT_SLICE].copy()

    for b in boundaries:
        if b <= 2 or b >= len(out) - 2:
            continue
        lo = max(0, b - radius)
        hi = min(len(out), b + radius + 1)

        seg = rot[lo:hi].copy()
        sm = gaussian_smooth(seg, sigma=args.sigma)

        n = hi - lo
        pos = np.linspace(0, 1, n, dtype=np.float32)
        center = (b - lo) / max(n - 1, 1)
        dist = np.abs(pos - center)
        w = 1.0 - np.clip(dist / max(center, 1 - center, 1e-6), 0, 1)
        w = smoothstep(w)[:, None] * strength

        rot[lo:hi] = (1.0 - w) * seg + w * sm

    out[:, ROT_SLICE] = rot
    save_motion(args.output, out, squeeze)

    report = {
        "input": args.input,
        "output": args.output,
        "boundaries": boundaries,
        "radius": radius,
        "sigma": args.sigma,
        "strength": strength,
        "before": metrics(motion),
        "after": metrics(out),
    }

    report_path = Path(args.output).with_suffix(".smooth_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
