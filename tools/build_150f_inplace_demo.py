#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import numpy as np

def load_motion(path):
    arr = np.load(path, allow_pickle=True).astype("float32")
    if arr.ndim == 3:
        arr = arr[0]
    return arr

def normalize_6d_block(m):
    """Re-normalize 6D rotations after linear blend."""
    x = m.copy()
    if x.shape[-1] < 151:
        return x
    r = x[:, 7:151].reshape(x.shape[0], 24, 6)
    a = r[..., 0:3]
    b = r[..., 3:6]

    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - (a * b).sum(axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)

    r[..., 0:3] = a
    r[..., 3:6] = b
    x[:, 7:151] = r.reshape(x.shape[0], -1)
    return x

def localize_unit(m):
    """Make each unit local/in-place before stitching."""
    x = m.copy()
    # subtract initial root X/Z; keep root_y
    x[:, 4] -= x[0, 4]
    x[:, 6] -= x[0, 6]
    return x

def boundary_cost(a, b, k=6):
    """Lower is better: exit of a vs entry of b."""
    aa = a[-k:, :].astype("float32")
    bb = b[:k, :].astype("float32")

    # ignore contacts for pose distance; use root_y + rotations
    pose_a = aa[:, 5:151]
    pose_b = bb[:, 5:151]
    pose = float(np.sqrt(np.mean((pose_a - pose_b) ** 2)))

    # contact consistency
    ca = aa[:, 0:4]
    cb = bb[:, 0:4]
    contact = float(np.mean(np.abs(ca - cb)))

    # root height consistency
    root_y = float(np.mean(np.abs(aa[:, 5] - bb[:, 5])))

    return 1.0 * pose + 0.6 * contact + 0.4 * root_y

def greedy_order(items):
    """items: list of dict with motion, score, file."""
    if len(items) <= 1:
        return items

    # start from highest phrase score
    remaining = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)
    ordered = [remaining.pop(0)]

    while remaining:
        prev = ordered[-1]["motion"]
        best_i, best_val = None, None
        for i, cand in enumerate(remaining):
            c = boundary_cost(prev, cand["motion"])
            # prefer high phrase score as tie-breaker
            val = c - 0.1 * cand.get("score", 0.0)
            if best_val is None or val < best_val:
                best_i, best_val = i, val
        ordered.append(remaining.pop(best_i))

    return ordered

def stitch_units(units, overlap=10, target_len=150):
    out = units[0].copy()
    boundaries = []

    for u in units[1:]:
        if overlap <= 0:
            boundaries.append(len(out))
            out = np.concatenate([out, u], axis=0)
            continue

        start_boundary = len(out) - overlap
        tail = out[-overlap:].copy()
        head = u[:overlap].copy()

        alpha = np.linspace(0.0, 1.0, overlap, dtype=np.float32)[:, None]
        blend = (1.0 - alpha) * tail + alpha * head

        # contacts: avoid fractional confusion; choose by alpha
        choose_next = (alpha[:, 0] >= 0.5)
        blend[:, 0:4] = tail[:, 0:4]
        blend[choose_next, 0:4] = head[choose_next, 0:4]

        blend = normalize_6d_block(blend)

        out[-overlap:] = blend
        boundaries.append(start_boundary)
        out = np.concatenate([out, u[overlap:]], axis=0)

    if len(out) > target_len:
        out = out[:target_len]
    elif len(out) < target_len:
        pad = np.repeat(out[-1:], target_len - len(out), axis=0)
        out = np.concatenate([out, pad], axis=0)

    return out.astype("float32"), boundaries

def root_drift_correct(m, max_radius=0.08):
    x = m.copy()
    root = x[:, 4:7].copy()

    # remove linear drift in X/Z
    xz0 = root[0, [0, 2]].copy()
    xz1 = root[-1, [0, 2]].copy()
    drift = np.linspace(0.0, 1.0, len(root), dtype=np.float32)[:, None] * (xz1 - xz0)[None, :]
    root[:, [0, 2]] = root[:, [0, 2]] - drift
    root[:, [0, 2]] = root[:, [0, 2]] - root[0, [0, 2]][None, :]

    # soft clamp radius to keep in-place
    rad = np.sqrt(root[:, 0] ** 2 + root[:, 2] ** 2)
    mx = float(rad.max()) if len(rad) else 0.0
    if mx > max_radius:
        scale = max_radius / max(mx, 1e-8)
        root[:, 0] *= scale
        root[:, 2] *= scale

    x[:, 4:7] = root
    return x

def frame_diff(m):
    if len(m) < 2:
        return np.zeros((0,), dtype=np.float32)
    d = m[1:] - m[:-1]
    return np.sqrt(np.mean(d * d, axis=1)).astype(np.float32)

def boundary_report(m, boundaries, window=4):
    d = frame_diff(m)
    rows = []
    for b in boundaries:
        lo = max(0, b - window)
        hi = min(len(d), b + window)
        rows.append({
            "boundary": int(b),
            "local_jump_mean": float(d[lo:hi].mean()) if hi > lo else 0.0,
            "local_jump_max": float(d[lo:hi].max()) if hi > lo else 0.0,
            "contact_before": m[max(0,b-1), 0:4].round(3).tolist(),
            "contact_after": m[min(len(m)-1,b), 0:4].round(3).tolist(),
            "contact_l1": float(np.mean(np.abs(m[max(0,b-1),0:4] - m[min(len(m)-1,b),0:4]))),
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal_csv", default="output/v15_final_safe_verified/eval/v15_verified_temporal_phrase_gate.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--target_len", type=int, default=150)
    ap.add_argument("--overlap", type=int, default=10)
    ap.add_argument("--num_units", type=int, default=4)
    ap.add_argument("--name_contains", default="p4")
    ap.add_argument("--max_radius", type=float, default=0.08)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.temporal_csv, encoding="utf-8")))
    rows = [r for r in rows if str(r.get("temporal_phrase_gate")) == "1"]

    if args.name_contains:
        rows2 = [r for r in rows if args.name_contains in r.get("name", "")]
        if len(rows2) >= args.num_units:
            rows = rows2

    rows = sorted(rows, key=lambda r: float(r.get("temporal_phrase_score", 0.0)), reverse=True)

    items = []
    for r in rows:
        p = Path(r["file"])
        if not p.exists():
            continue
        m = load_motion(p)[:45]
        m = localize_unit(m)
        m = normalize_6d_block(m)
        items.append({
            "file": str(p),
            "name": r.get("name", p.stem),
            "score": float(r.get("temporal_phrase_score", 0.0)),
            "motion": m,
        })

    if len(items) < args.num_units:
        raise SystemExit(f"Need at least {args.num_units} valid units, got {len(items)}")

    items = greedy_order(items[: max(args.num_units * 3, args.num_units)])
    selected = items[:args.num_units]

    motion, boundaries = stitch_units([x["motion"] for x in selected], overlap=args.overlap, target_len=args.target_len)
    motion = root_drift_correct(motion, max_radius=args.max_radius)
    motion = normalize_6d_block(motion)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, motion[None].astype("float32"))

    root = motion[:, 4:7]
    rad = np.sqrt(root[:, 0] ** 2 + root[:, 2] ** 2)

    rep = {
        "out": str(out),
        "target_len": args.target_len,
        "overlap": args.overlap,
        "num_units": args.num_units,
        "selected": [
            {"name": x["name"], "file": x["file"], "score": x["score"]}
            for x in selected
        ],
        "boundaries": boundaries,
        "root_max_radius": float(rad.max()),
        "root_final_xz": root[-1, [0,2]].tolist(),
        "boundary_report": boundary_report(motion, boundaries),
    }

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ wrote", out, "shape", motion[None].shape)
    print("✅ report", report)
    print("selected:")
    for x in selected:
        print(" ", x["name"], x["file"], "score", x["score"])
    print("boundaries:", boundaries)
    print("root_max_radius:", rep["root_max_radius"])

if __name__ == "__main__":
    main()
