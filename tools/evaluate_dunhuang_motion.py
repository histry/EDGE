#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule-based Dunhuang Motion Evaluator / Motion Critic v0.

This is designed as a reliable first evaluator before training any learned critic.
It produces global scores, boundary diagnostics and bad-region hints that can be
fed back to prior selection and scheduling.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from tools.v20_motion_utils import (
    CONTACT,
    ROOT_X,
    ROOT_Z,
    ROT,
    compute_motion_curves,
    describe_motion_event,
    iter_motion_files,
    jerk_score,
    load_motion_any,
    root_path_radius,
    validate_motion,
    write_json,
)


def boundary_jumps(motion: np.ndarray, boundaries: List[int], radius: int = 3) -> List[Dict]:
    m = validate_motion(motion)
    out = []
    for b in boundaries:
        if b <= 0 or b >= len(m):
            continue
        lo = max(0, b - radius)
        hi = min(len(m) - 1, b + radius)
        local = []
        for t in range(lo, hi):
            local.append(float(np.linalg.norm(m[t + 1, ROT] - m[t, ROT]) / np.sqrt(144.0)))
        val = float(max(local)) if local else 0.0
        out.append({"frame": int(b), "local_jump_max": val})
    return out


def infer_boundaries_from_schedule(report_path: Path | None) -> List[int]:
    if not report_path or not report_path.exists():
        return []
    import json
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        frames = []
        for step in data.get("schedule", []):
            f = step.get("start_frame_est", None)
            if f is not None and int(f) > 0:
                frames.append(int(f))
        return sorted(set(frames))
    except Exception:
        return []


def evaluate_motion(motion: np.ndarray, boundaries: List[int] | None = None) -> Dict:
    m = validate_motion(motion)
    desc = describe_motion_event(m)
    curves = compute_motion_curves(m, smooth=5)
    upper = float(np.mean(curves["upper"]))
    torso = float(np.mean(curves["torso"]))
    lower = float(np.mean(curves["lower"]))
    full = float(np.mean(curves["full"]))
    root_radius = root_path_radius(m)
    jerk = jerk_score(m)
    contact_switch = float(np.mean(curves["contact_switch"]))
    bjs = boundary_jumps(m, boundaries or [])
    max_bj = max([x["local_jump_max"] for x in bjs], default=0.0)
    # Rule scores in [0,1]
    root_score = float(np.exp(-12.0 * root_radius))
    smooth_score = float(np.exp(-0.05 * jerk))
    boundary_score = float(np.exp(-3.5 * max_bj))
    activity_score = float(np.tanh((upper + torso + 0.5 * lower) * 6.0))
    style_tension = float(np.tanh(float(desc.get("style_tension", 0.0)) * 8.0))
    support_score = float(np.exp(-2.0 * max(0.0, contact_switch - 0.30)))
    dunhuang_presence = float(0.35 * style_tension + 0.30 * activity_score + 0.20 * boundary_score + 0.15 * root_score)
    global_score = float(0.30 * dunhuang_presence + 0.25 * boundary_score + 0.20 * root_score + 0.15 * smooth_score + 0.10 * support_score)
    bad = []
    for bj in bjs:
        if bj["local_jump_max"] > 0.85:
            bad.append({"start": max(0, bj["frame"] - 6), "end": min(len(m), bj["frame"] + 7), "type": "boundary_jump", "value": bj["local_jump_max"], "repair_hint": "reroute boundary or use longer DPN transition"})
    if root_radius > 0.05:
        bad.append({"start": 0, "end": len(m), "type": "root_drift", "value": root_radius, "repair_hint": "apply root lock or support-aware trajectory adapter"})
    if activity_score < 0.25:
        bad.append({"start": 0, "end": len(m), "type": "over_smoothing", "value": activity_score, "repair_hint": "increase event/activity weight or reduce preserve loss"})
    return {
        "global_score": global_score,
        "dunhuang_presence": dunhuang_presence,
        "motion_quality": smooth_score,
        "boundary_harmony": boundary_score,
        "root_stability": root_score,
        "support_stability": support_score,
        "style_tension": style_tension,
        "activity_score": activity_score,
        "upper_activity": upper,
        "torso_activity": torso,
        "lower_activity": lower,
        "full_activity": full,
        "root_max_radius": root_radius,
        "jerk": jerk,
        "contact_switch": contact_switch,
        "boundary_jumps": bjs,
        "bad_regions": bad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", default="")
    ap.add_argument("--motion_dir", default="")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--boundaries", default="", help="comma-separated boundary frames, optional")
    args = ap.parse_args()
    files: List[Path] = []
    if args.motion:
        files = [Path(args.motion)]
    elif args.motion_dir:
        files = iter_motion_files(args.motion_dir, exts=(".npy", ".npz", ".pkl"))
    else:
        raise ValueError("Provide --motion or --motion_dir")
    explicit_boundaries = [int(x) for x in args.boundaries.replace(";", ",").split(",") if x.strip()]
    rows = []
    for f in files:
        try:
            m, _ = load_motion_any(f)
            report_path = f.with_suffix(".schedule_report.json")
            b = explicit_boundaries or infer_boundaries_from_schedule(report_path)
            ev = evaluate_motion(m, b)
            rows.append({"motion": str(f), **ev})
            print(f"{f}: score={ev['global_score']:.4f} presence={ev['dunhuang_presence']:.4f} bad={len(ev['bad_regions'])}")
        except Exception as exc:
            rows.append({"motion": str(f), "error": str(exc)})
            print(f"FAIL {f}: {exc}")
    summary = {"num_items": len(rows), "items": rows}
    if rows:
        vals = [r.get("global_score") for r in rows if isinstance(r.get("global_score"), (int, float))]
        if vals:
            summary["mean_global_score"] = float(np.mean(vals))
            summary["best_motion"] = max(rows, key=lambda r: float(r.get("global_score", -1))).get("motion")
    write_json(summary, args.out_json)
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
