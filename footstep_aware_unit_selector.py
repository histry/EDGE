#!/usr/bin/env python3
"""Stage-1 selector: Footstep-aware dual-score / dynamic RAG unit selection.

This is intentionally standalone so you can validate retrieval quality without
changing the existing generate_v10_choreo.py pipeline.

It writes:
  <out_prefix>_mid01_f25.npy          center pose
  <out_prefix>_mid01_f25_unit.npy     selected motion unit [T,151]
  <out_prefix>_footstep_plan.json     audit plan

Use the generated *_unit.npy files directly with segment_lower_body_compositor.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from footstep_phase_utils import add_dual_scores, as_t151, robust_norm, unit_basic_stats


def parse_frames(text: str, num_frames: int) -> List[int]:
    if text.strip():
        out = []
        for item in text.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            if 0.0 < value < 1.0:
                value *= (num_frames - 1)
            out.append(max(1, min(num_frames - 2, int(round(value)))))
        return out
    return [25, 50, 75, 100, 125] if num_frames == 150 else [int(round((i + 1) * (num_frames - 1) / 6)) for i in range(5)]


def candidate_array(npz) -> Tuple[str, np.ndarray]:
    for key in ["unit_motions_physical", "unit_motions", "motion_units", "motions", "units", "x"]:
        if key not in npz.files:
            continue
        arr = np.asarray(npz[key])
        if arr.ndim == 3 and (arr.shape[-1] == 151 or arr.shape[1] == 151):
            if arr.shape[1] == 151:
                arr = np.transpose(arr, (0, 2, 1))
            return key, arr.astype(np.float32)
    raise ValueError(f"No unit array found. keys={npz.files}")


def load_db(path: str):
    npz = np.load(path, allow_pickle=True)
    key, units = candidate_array(npz)
    n = len(units)
    stats: Dict[str, np.ndarray] = {}
    raw_keys = [
        "motion_energy", "unit_energy", "root_speed", "upper_activity", "lower_activity",
        "spatial_range", "turning", "contact_stability", "contact_switch",
        "alternating_foot_phase", "root_lower_sync", "expressiveness_score",
        "locomotion_score", "footstep_score", "mobile_score",
    ]
    for k in raw_keys:
        if k in npz.files and len(npz[k]) >= n:
            stats[k] = np.asarray(npz[k][:n], dtype=np.float32)
    # Fill missing raw stats by computing from units.
    required = ["root_speed", "upper_activity", "lower_activity", "spatial_range", "contact_switch", "root_lower_sync"]
    if any(k not in stats for k in required):
        rows = [unit_basic_stats(u) for u in units]
        for k in ["motion_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning", "contact_stability", "contact_switch", "alternating_foot_phase", "root_lower_sync"]:
            if k not in stats:
                stats[k] = np.asarray([r.get(k, 0.0) for r in rows], dtype=np.float32)
    if "unit_energy" not in stats and "motion_energy" in stats:
        stats["unit_energy"] = stats["motion_energy"].copy()
    for k in ["motion_energy", "unit_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning", "contact_switch", "root_lower_sync"]:
        nk = k + "_norm"
        if nk not in stats:
            stats[nk] = robust_norm(stats.get(k, np.zeros(n, dtype=np.float32)))[0]
    add_dual_scores(stats)
    meta = []
    source = np.asarray(npz["source"]) if "source" in npz.files and len(npz["source"]) >= n else np.asarray([""] * n)
    unit_start = np.asarray(npz["unit_start"]) if "unit_start" in npz.files and len(npz["unit_start"]) >= n else np.full(n, -1)
    unit_center = np.asarray(npz["unit_center"]) if "unit_center" in npz.files and len(npz["unit_center"]) >= n else np.full(n, -1)
    for i in range(n):
        meta.append({"index": int(i), "source": str(source[i]), "unit_start": int(unit_start[i]), "unit_center": int(unit_center[i])})
    return units, stats, meta, key


def local_speed(traj: Optional[np.ndarray], frame: int, radius: int = 8) -> float:
    if traj is None or len(traj) < 2:
        return 0.0
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    traj = traj[:, :2]
    a = max(0, int(frame) - int(radius))
    b = min(len(traj) - 1, int(frame) + int(radius))
    if b <= a:
        return 0.0
    return float(np.linalg.norm(traj[a + 1 : b + 1] - traj[a:b], axis=-1).mean())


def select_indices(units, stats, frames, traj, threshold, top_k, disallow_same_source, meta):
    selected = []
    used_sources = set()
    plan = []
    expr = stats["expressiveness_score"]
    mobile = stats["mobile_score"]
    foot = stats["footstep_score"]
    loco = stats["locomotion_score"]
    lower = stats.get("lower_activity_norm", np.zeros_like(expr))
    root = stats.get("root_speed_norm", np.zeros_like(expr))

    for order, frame in enumerate(frames, start=1):
        spd = local_speed(traj, frame)
        mobile_route = spd >= threshold
        if mobile_route:
            score = 0.55 * mobile + 0.25 * foot + 0.20 * loco
            score = score - 0.20 * np.maximum(0.0, 0.25 - lower) - 0.20 * np.maximum(0.0, 0.20 - root)
            route = "mobile"
        else:
            score = 0.75 * expr + 0.15 * stats.get("upper_activity_norm", np.zeros_like(expr)) + 0.10 * foot
            route = "expressive"
        score = np.asarray(score, dtype=np.float32).copy()
        # Basic diversity: avoid exact same source if possible.
        if disallow_same_source:
            for i, m in enumerate(meta):
                if m.get("source") in used_sources and len(used_sources) < len(meta):
                    score[i] -= 0.25
        for idx in selected:
            score[int(idx)] -= 1.0
        ranking = np.argsort(-score)
        chosen = int(ranking[0])
        selected.append(chosen)
        used_sources.add(meta[chosen].get("source", ""))
        plan.append({
            "slot": order,
            "frame": int(frame),
            "route": route,
            "target_speed": float(spd),
            "index": chosen,
            "score": float(score[chosen]),
            "source": meta[chosen].get("source", ""),
            "unit_start": meta[chosen].get("unit_start", -1),
            "unit_center": meta[chosen].get("unit_center", -1),
            "expressiveness_score": float(expr[chosen]),
            "locomotion_score": float(loco[chosen]),
            "footstep_score": float(foot[chosen]),
            "mobile_score": float(mobile[chosen]),
            "lower_activity_norm": float(lower[chosen]),
            "root_speed_norm": float(root[chosen]),
            "top_candidates": [int(x) for x in ranking[: int(top_k)]],
        })
    return selected, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_db", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--target_traj", default="", help="Optional [T,2] or [1,T,2] target trajectory .npy")
    parser.add_argument("--mid_frames", default="25,50,75,100,125")
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--mobile_speed_threshold", type=float, default=0.010)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--allow_same_source", action="store_true")
    args = parser.parse_args()

    units, stats, meta, unit_key = load_db(args.rag_db)
    traj = np.load(args.target_traj, allow_pickle=True) if args.target_traj else None
    frames = parse_frames(args.mid_frames, args.num_frames)
    selected, plan = select_indices(
        units, stats, frames, traj,
        threshold=float(args.mobile_speed_threshold),
        top_k=int(args.top_k),
        disallow_same_source=not args.allow_same_source,
        meta=meta,
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for slot, (idx, frame) in enumerate(zip(selected, frames), start=1):
        unit = as_t151(units[idx]).astype(np.float32)
        center = unit[len(unit) // 2].astype(np.float32)
        np.save(f"{out_prefix}_mid{slot:02d}_f{frame}_unit.npy", unit)
        np.save(f"{out_prefix}_mid{slot:02d}_f{frame}.npy", center)
    payload = {
        "rag_db": str(args.rag_db),
        "unit_key": unit_key,
        "target_traj": args.target_traj,
        "mobile_speed_threshold": float(args.mobile_speed_threshold),
        "frames": frames,
        "selected": plan,
    }
    plan_path = f"{out_prefix}_footstep_plan.json"
    Path(plan_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Selected {len(selected)} footstep-aware units")
    print(f"   plan={plan_path}")
    for item in plan:
        print(f"   mid{item['slot']:02d} frame={item['frame']} route={item['route']} idx={item['index']} mobile={item['mobile_score']:.3f} expr={item['expressiveness_score']:.3f} src={item['source']}")


if __name__ == "__main__":
    main()
