#!/usr/bin/env python3
"""Select functional dual contexts from an augmented ChoreoRAG DB.

Output:
  prefix_support_midXX_fYY.npy
  prefix_support_midXX_fYY_unit.npy
  prefix_expressive_midXX_fYY.npy
  prefix_expressive_midXX_fYY_unit.npy
  prefix_functional_context.env

Design:
  Support context decides WHEN support/weight-shift/step events happen.
  Expressive-mobile context decides HOW torso/arms respond around those events.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from footstep_phase_utils import ROOT_X_IDX, ROOT_Z_IDX


def parse_points(text: str) -> np.ndarray:
    pts = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        a, b = item.split(",")[:2]
        pts.append([float(a), float(b)])
    if len(pts) < 2:
        raise ValueError("--trajectory must contain at least two points, e.g. '0,0;1,1'")
    return np.asarray(pts, dtype=np.float32)


def parse_frames(text: str, count: int, seq_len: int) -> List[int]:
    if text.strip():
        return [int(round(float(x))) for x in text.replace(";", ",").split(",") if x.strip()]
    return [int(round((i + 1) * seq_len / (count + 1))) for i in range(count)]


def interp_traj(points: np.ndarray, seq_len: int) -> np.ndarray:
    src = np.linspace(0, 1, len(points))
    dst = np.linspace(0, 1, seq_len)
    x = np.interp(dst, src, points[:, 0])
    z = np.interp(dst, src, points[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def frame_speed_turn(traj: np.ndarray):
    T = len(traj)
    speed = np.zeros((T,), dtype=np.float32)
    turn = np.zeros((T,), dtype=np.float32)
    if T > 1:
        v = traj[1:] - traj[:-1]
        speed[1:] = np.linalg.norm(v, axis=1)
        speed[0] = speed[1]
        if len(v) > 1:
            v1, v2 = v[:-1], v[1:]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
            turn[2:] = 1.0 - np.clip(cos, -1.0, 1.0)
            turn[1] = turn[2] if T > 2 else 0.0
    return speed, turn


def field(db, name: str, default: float = 0.0) -> np.ndarray:
    if name in db.files:
        return np.asarray(db[name], dtype=np.float32)
    n = len(db["unit_motions"]) if "unit_motions" in db.files else len(db["poses"])
    return np.full((n,), float(default), dtype=np.float32)


def norm01(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, [10, 90])
    return np.clip((x - lo) / max(float(hi - lo), 1e-8), 0.0, 1.0).astype(np.float32)


def greedy_select(score, db, k: int, used: set, source_gap: int = 0):
    order = np.argsort(-score)
    out = []
    sources = db["source"].astype(str) if "source" in db.files else np.asarray([""] * len(score))
    centers = db["unit_center"] if "unit_center" in db.files else db.get("source_frame", np.arange(len(score)))
    for idx in order:
        idx = int(idx)
        if idx in used:
            continue
        src = sources[idx]
        cf = int(centers[idx])
        ok = True
        for j in out:
            if source_gap > 0 and sources[j] == src and abs(int(centers[j]) - cf) < source_gap:
                ok = False
                break
        if not ok:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= k:
            break
    return out


def save_selected(prefix: str, role: str, db, indices: List[int], frames: List[int], out_dir: Path, unit_space: str):
    unit_key = "unit_motions" if unit_space == "normalized" else ("unit_motions_physical" if "unit_motions_physical" in db.files else "unit_motions")
    units = db[unit_key]
    poses = db["poses"] if unit_space == "normalized" else (db["center_pose_physical"] if "center_pose_physical" in db.files else db["poses"])

    unit_paths, pose_paths = [], []
    for n, (idx, frame) in enumerate(zip(indices, frames), start=1):
        pose_path = out_dir / f"{prefix}_{role}_mid{n:02d}_f{int(frame)}.npy"
        unit_path = out_dir / f"{prefix}_{role}_mid{n:02d}_f{int(frame)}_unit.npy"
        np.save(pose_path, np.asarray(poses[idx], dtype=np.float32))
        np.save(unit_path, np.asarray(units[idx], dtype=np.float32))
        pose_paths.append(str(pose_path))
        unit_paths.append(str(unit_path))
    return pose_paths, unit_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--seq_len", type=int, default=150)
    ap.add_argument("--frames", default="")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--support_k", type=int, default=5)
    ap.add_argument("--expressive_k", type=int, default=5)
    ap.add_argument("--source_gap", type=int, default=45)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="dhw4")
    ap.add_argument("--unit_space", choices=["normalized", "physical"], default="normalized")
    ap.add_argument("--support_min_mobile", type=float, default=0.0)
    ap.add_argument("--expressive_min_mobile", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = np.load(args.rag_db, allow_pickle=True)
    traj = interp_traj(parse_points(args.trajectory), args.seq_len)
    speed, turn = frame_speed_turn(traj)
    speed_n = norm01(speed)
    turn_n = norm01(turn)

    frames = parse_frames(args.frames, args.count, args.seq_len)
    frames = frames[: args.count]

    support_base = (
        0.45 * field(db, "support_context_score")
        + 0.25 * field(db, "mobile_score")
        + 0.20 * field(db, "footstep_score")
        + 0.10 * field(db, "speed_lower_sync")
    )

    expressive_base = (
        0.45 * field(db, "expressive_mobile_score")
        + 0.25 * field(db, "mobile_expressive_score")
        + 0.20 * field(db, "functional_coupling_score")
        + 0.10 * field(db, "turn_expression_response")
    )

    # Softly favor mobile units for high-speed / turning frames.
    support_scores = []
    expressive_scores = []
    for f in frames:
        f = int(np.clip(f, 0, args.seq_len - 1))
        event_strength = 0.65 * speed_n[f] + 0.35 * turn_n[f]
        support_scores.append(support_base + event_strength * field(db, "support_context_score"))
        expressive_scores.append(expressive_base + event_strength * field(db, "expressive_mobile_score"))

    support_indices = []
    expressive_indices = []
    used_support, used_expr = set(), set()
    for i in range(len(frames)):
        sidx = greedy_select(support_scores[i], db, 1, used_support, source_gap=args.source_gap)
        eidx = greedy_select(expressive_scores[i], db, 1, used_expr, source_gap=args.source_gap)
        support_indices.extend(sidx)
        expressive_indices.extend(eidx)

    support_indices = support_indices[: args.support_k]
    expressive_indices = expressive_indices[: args.expressive_k]

    support_pose_paths, support_unit_paths = save_selected(args.prefix, "support", db, support_indices, frames, out_dir, args.unit_space)
    expressive_pose_paths, expressive_unit_paths = save_selected(args.prefix, "expressive", db, expressive_indices, frames, out_dir, args.unit_space)

    report = {
        "rag_db": args.rag_db,
        "prefix": args.prefix,
        "trajectory": args.trajectory,
        "frames": frames,
        "unit_space": args.unit_space,
        "support_indices": support_indices,
        "expressive_indices": expressive_indices,
        "support_units": support_unit_paths,
        "expressive_units": expressive_unit_paths,
        "support_mid_poses": support_pose_paths,
        "expressive_mid_poses": expressive_pose_paths,
        "support_scores": [float(support_scores[i][support_indices[i]]) for i in range(min(len(support_indices), len(frames)))],
        "expressive_scores": [float(expressive_scores[i][expressive_indices[i]]) for i in range(min(len(expressive_indices), len(frames)))],
    }

    report_path = out_dir / f"{args.prefix}_functional_context_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    env_path = out_dir / f"{args.prefix}_functional_context.env"
    env_text = "\n".join([
        f'export EDGE_SUPPORT_CONTEXT_UNIT_PATHS="{",".join(support_unit_paths)}"',
        f'export EDGE_SUPPORT_CONTEXT_MID_POSES="{",".join(support_pose_paths)}"',
        f'export EDGE_EXPRESSIVE_CONTEXT_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
        f'export EDGE_EXPRESSIVE_CONTEXT_MID_POSES="{",".join(expressive_pose_paths)}"',
        f'export EDGE_FUNCTIONAL_CONTEXT_FRAMES="{",".join(map(str, frames))}"',
        # Text/Pose RAG should consume expressive-mobile units.
        f'export EDGE_RAG_CONTEXT_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
        f'export EDGE_RAG_SUMMARY_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
    ]) + "\n"
    env_path.write_text(env_text, encoding="utf-8")

    print(f"✅ Functional dual contexts exported: {out_dir}")
    print(f"   report={report_path}")
    print(f"   env={env_path}")
    print(f"   support_units={len(support_unit_paths)} expressive_units={len(expressive_unit_paths)}")
    print("\nRun:")
    print(f"   source {env_path}")


if __name__ == "__main__":
    main()
