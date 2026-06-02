#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)


def parse_starts(text):
    return [int(float(x.strip())) for x in text.replace(";", ",").split(",") if x.strip()]


def load_pkl_motion(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    for k in ["motion", "motion_151", "poses"]:
        if k in obj:
            x = np.asarray(obj[k], dtype=np.float32)
            if x.ndim == 3 and x.shape[0] == 1:
                x = x[0]
            if x.ndim == 2 and x.shape[-1] == 151:
                return x.astype(np.float32), obj
    raise ValueError(f"No [T,151] motion in {path}")


def metrics(m):
    root = m[:, [ROOT_X, ROOT_Z]]
    rot = m[:, ROT]
    droot = np.linalg.norm(root[1:] - root[:-1], axis=1) if len(m) > 1 else np.zeros(1)
    drot = np.linalg.norm(rot[1:] - rot[:-1], axis=1) if len(m) > 1 else np.zeros(1)
    out = {
        "frames": int(len(m)),
        "root_max_radius": float(np.linalg.norm(root - root[:1], axis=1).max()),
        "global_root_jump_p95": float(np.percentile(droot, 95)),
        "global_rot_jump_p95": float(np.percentile(drot, 95)),
        "segment_activity_mean": float(drot.mean()),
    }
    for b in [35, 45, 70, 74, 90, 105, 108, 135, 140, 142]:
        if 2 <= b < len(m) - 2:
            lo = max(1, b - 2)
            hi = min(len(m), b + 3)
            local = np.linalg.norm(rot[lo:hi] - rot[lo - 1:hi - 1], axis=1)
            out[f"boundary_{b}_local_rot_jump_max"] = float(local.max())
    return out


def transition_cost(prev, cand):
    prev_exit = prev[-1, ROT]
    cand_entry = cand[0, ROT]
    rot_jump = float(np.linalg.norm(cand_entry - prev_exit))

    prev_root = prev[-1, [ROOT_X, ROOT_Z]]
    cand_root = cand[0, [ROOT_X, ROOT_Z]]
    root_jump = float(np.linalg.norm(cand_root - prev_root))

    return rot_jump + 10.0 * root_jump


def localize_root(m):
    out = m.copy()
    out[:, ROOT_X] -= out[0, ROOT_X]
    out[:, ROOT_Z] -= out[0, ROOT_Z]
    return out


def paste_with_blend(canvas, unit, start, blend_radius):
    T = canvas.shape[0]
    unit = localize_root(unit)
    end = min(T, start + len(unit))
    length = end - start
    if length <= 0:
        return canvas

    seg = unit[:length]

    if np.allclose(canvas[start:end], 0.0):
        canvas[start:end] = seg
        return canvas

    # Soft blend only where canvas already has content.
    for i in range(length):
        t = start + i
        existing_nonzero = np.any(np.abs(canvas[t]) > 1e-8)
        if not existing_nonzero:
            canvas[t] = seg[i]
            continue

        if i < blend_radius:
            a = i / max(blend_radius, 1)
        else:
            a = 1.0
        a = np.clip(a, 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)

        canvas[t, CONTACT] = seg[i, CONTACT]
        canvas[t, ROOT_X] = 0.0
        canvas[t, ROOT_Z] = 0.0
        canvas[t, ROOT_Y] = (1.0 - a) * canvas[t, ROOT_Y] + a * seg[i, ROOT_Y]
        canvas[t, ROT] = (1.0 - a) * canvas[t, ROT] + a * seg[i, ROT]

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--starts", default="0,35,74,108")
    ap.add_argument("--candidate_top_k", type=int, default=120)
    ap.add_argument("--transition_weight", type=float, default=0.20)
    ap.add_argument("--activity_weight", type=float, default=0.15)
    ap.add_argument("--visual_weight", type=float, default=1.0)
    ap.add_argument("--blend_radius", type=int, default=6)
    ap.add_argument("--max_transition_cost", type=float, default=999.0)
    args = ap.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    items = report["items"][: args.candidate_top_k]
    starts = parse_starts(args.starts)

    candidates = []
    for item in items:
        pkl = item["pkl"]
        motion, obj = load_pkl_motion(pkl)
        candidates.append({
            "motion": motion,
            "item": item,
            "pkl": pkl,
            "source_index": item.get("source_index", -1),
            "final_score": float(item.get("final_score", item.get("visual_score", 0.0))),
            "visual_score": float(item.get("visual_score", 0.0)),
            "activity": float(item.get("activity", 0.0)),
        })

    if not candidates:
        raise RuntimeError("No candidates loaded.")

    canvas = np.zeros((args.num_frames, 151), dtype=np.float32)
    chosen = []

    for slot, st in enumerate(starts):
        best = None
        best_score = -1e9

        for cand in candidates:
            if cand["source_index"] in [c["source_index"] for c in chosen]:
                continue

            sc = args.visual_weight * cand["final_score"] + args.activity_weight * cand["activity"]

            if chosen:
                prev = chosen[-1]["motion"]
                cost = transition_cost(prev, cand["motion"])
                if cost > args.max_transition_cost:
                    continue
                sc -= args.transition_weight * cost
            else:
                cost = 0.0

            if sc > best_score:
                best_score = sc
                best = {**cand, "slot": slot, "start": st, "transition_cost": float(cost), "slot_score": float(sc)}

        if best is None:
            # fallback: allow reuse if all transition-filtered out
            best = max(candidates, key=lambda c: c["final_score"])
            best = {**best, "slot": slot, "start": st, "transition_cost": None, "slot_score": float(best["final_score"])}

        chosen.append(best)
        canvas = paste_with_blend(canvas, best["motion"], st, args.blend_radius)

    # Fill any empty tail with last valid frame.
    nz = np.where(np.any(np.abs(canvas) > 1e-8, axis=1))[0]
    if len(nz):
        last = int(nz[-1])
        for t in range(last + 1, args.num_frames):
            canvas[t] = canvas[last]

    # Keep root X/Z exactly in-place.
    canvas[:, ROOT_X] = 0.0
    canvas[:, ROOT_Z] = 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, canvas[None].astype(np.float32))

    schedule = []
    for c in chosen:
        schedule.append({
            "slot": c["slot"],
            "start": c["start"],
            "source_index": int(c["source_index"]),
            "pkl": c["pkl"],
            "final_score": float(c["final_score"]),
            "visual_score": float(c["visual_score"]),
            "activity": float(c["activity"]),
            "transition_cost": c["transition_cost"],
            "slot_score": c["slot_score"],
        })

    out_report = {
        "out": str(out),
        "source_report": args.report,
        "num_frames": args.num_frames,
        "starts": starts,
        "schedule": schedule,
        "metrics": metrics(canvas),
    }

    report_path = out.with_suffix(".schedule_report.json")
    report_path.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(out_report, ensure_ascii=False, indent=2))
    print(f"saved: {out}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
