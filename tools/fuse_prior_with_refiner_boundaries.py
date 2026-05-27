import argparse
import json
from pathlib import Path
import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)

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
        raise ValueError(f"{path}: expected [T,151] or [1,T,151], got {arr.shape}")
    return arr, squeeze

def save_motion(path, motion, squeeze):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, motion[None].astype(np.float32) if squeeze else motion.astype(np.float32))

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
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
    ap.add_argument("--prior", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--boundaries", default="35,74,105,108")
    ap.add_argument("--radius", type=int, default=5)
    ap.add_argument("--strength", type=float, default=0.75)
    ap.add_argument("--include_root_y", action="store_true")
    args = ap.parse_args()

    prior, squeeze = load_motion(args.prior)
    refiner, _ = load_motion(args.refiner)

    T = min(len(prior), len(refiner))
    prior = prior[:T].copy()
    refiner = refiner[:T].copy()

    out = prior.copy()
    boundaries = [int(x) for x in args.boundaries.replace(";", ",").split(",") if x.strip()]
    radius = int(args.radius)
    strength = float(np.clip(args.strength, 0.0, 1.0))

    for b in boundaries:
        if b <= 1 or b >= T - 2:
            continue

        lo = max(0, b - radius)
        hi = min(T, b + radius + 1)
        n = hi - lo

        # center-heavy bell weight: only repair local phrase boundary
        grid = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        w = (1.0 - np.abs(grid))
        w = smoothstep(w)[:, None] * strength

        out[lo:hi, ROT] = (1.0 - w) * prior[lo:hi, ROT] + w * refiner[lo:hi, ROT]

        if args.include_root_y:
            out[lo:hi, ROOT_Y:ROOT_Y+1] = (
                (1.0 - w) * prior[lo:hi, ROOT_Y:ROOT_Y+1]
                + w * refiner[lo:hi, ROOT_Y:ROOT_Y+1]
            )

    # Always preserve original contacts and root X/Z from prior.
    out[:, CONTACT] = prior[:, CONTACT]
    out[:, ROOT_X] = prior[:, ROOT_X]
    out[:, ROOT_Z] = prior[:, ROOT_Z]

    save_motion(args.output, out, squeeze)

    report = {
        "prior": args.prior,
        "refiner": args.refiner,
        "output": args.output,
        "boundaries": boundaries,
        "radius": radius,
        "strength": strength,
        "include_root_y": bool(args.include_root_y),
        "metrics_prior": metrics(prior),
        "metrics_refiner": metrics(refiner),
        "metrics_fused": metrics(out),
    }

    report_path = Path(args.output).with_suffix(".fuse_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
