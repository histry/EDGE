#!/usr/bin/env python3
"""Select functional dual contexts from an augmented ChoreoRAG DB.

Replacement version with turn-aware event conditioning.

New behavior:
  --auto_event_frames computes turn/speed/curvature events from trajectory.
  support_frames and expressive_frames can be different:
    support happens before a turn, expressive response happens after/around turn.

Outputs:
  prefix_functional_context.env
  prefix_functional_context_report.json
  prefix_turn_events.npy / .json when auto_event_frames is enabled

Backwards compatible:
  --frames still works and produces same support/expressive frames.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from turn_aware_event_utils import (
    TurnEventConfig,
    detect_turn_events,
    interp_traj,
    norm01,
    parse_int_list,
    parse_points,
    save_event_report,
    trajectory_features,
)


def field(db, name: str, default: float = 0.0) -> np.ndarray:
    if name in db.files:
        return np.nan_to_num(np.asarray(db[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if "unit_motions" in db.files:
        n = len(db["unit_motions"])
    elif "poses" in db.files:
        n = len(db["poses"])
    else:
        raise KeyError("Cannot infer DB length; expected unit_motions or poses")
    return np.full((n,), float(default), dtype=np.float32)


def parse_frames_or_even(text: str, count: int, seq_len: int) -> List[int]:
    frames = parse_int_list(text)
    if frames:
        return [int(np.clip(f, 0, seq_len - 1)) for f in frames[:count]]
    return [int(round((i + 1) * seq_len / (count + 1))) for i in range(count)]


def greedy_select(score: np.ndarray, db, k: int, used: set, source_gap: int = 0) -> List[int]:
    score = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-score)
    sources = db["source"].astype(str) if "source" in db.files else np.asarray([""] * len(score))
    if "unit_center" in db.files:
        centers = np.asarray(db["unit_center"])
    elif "source_frame" in db.files:
        centers = np.asarray(db["source_frame"])
    else:
        centers = np.arange(len(score))
    out: List[int] = []
    for idx in order:
        idx = int(idx)
        if idx in used:
            continue
        src = str(sources[idx])
        cf = int(centers[idx])
        ok = True
        for j in out:
            if source_gap > 0 and str(sources[j]) == src and abs(int(centers[j]) - cf) < source_gap:
                ok = False
                break
        if not ok:
            continue
        out.append(idx)
        used.add(idx)
        if len(out) >= k:
            break
    return out


def save_selected(
    prefix: str,
    role: str,
    db,
    indices: Sequence[int],
    frames: Sequence[int],
    out_dir: Path,
    unit_space: str,
) -> Tuple[List[str], List[str]]:
    unit_key = "unit_motions" if unit_space == "normalized" else (
        "unit_motions_physical" if "unit_motions_physical" in db.files else "unit_motions"
    )
    pose_key = "poses" if unit_space == "normalized" else (
        "center_pose_physical" if "center_pose_physical" in db.files else "poses"
    )
    units = db[unit_key]
    poses = db[pose_key]
    unit_paths: List[str] = []
    pose_paths: List[str] = []
    for n, idx in enumerate(indices, start=1):
        frame = int(frames[min(n - 1, len(frames) - 1)])
        pose_path = out_dir / f"{prefix}_{role}_mid{n:02d}_f{frame}.npy"
        unit_path = out_dir / f"{prefix}_{role}_mid{n:02d}_f{frame}_unit.npy"
        np.save(pose_path, np.asarray(poses[int(idx)], dtype=np.float32))
        np.save(unit_path, np.asarray(units[int(idx)], dtype=np.float32))
        pose_paths.append(str(pose_path))
        unit_paths.append(str(unit_path))
    return pose_paths, unit_paths


def score_contexts(db, event_strength: float, role: str) -> np.ndarray:
    support_base = (
        0.42 * field(db, "support_context_score")
        + 0.20 * field(db, "mobile_score")
        + 0.22 * field(db, "footstep_score")
        + 0.16 * field(db, "speed_lower_sync")
    )
    expressive_base = (
        0.34 * field(db, "expressive_mobile_score")
        + 0.22 * field(db, "mobile_expressive_score")
        + 0.22 * field(db, "functional_coupling_score")
        + 0.22 * field(db, "turn_expression_response")
    )
    if role == "support":
        return support_base + float(event_strength) * (
            0.50 * field(db, "support_context_score")
            + 0.25 * field(db, "contact_switch_norm")
            + 0.25 * field(db, "speed_lower_sync_norm")
        )
    if role == "expressive":
        return expressive_base + float(event_strength) * (
            0.45 * field(db, "turn_expression_response_norm")
            + 0.30 * field(db, "expressive_mobile_score")
            + 0.25 * field(db, "speed_expression_sync_norm")
        )
    raise ValueError(role)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_db", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--seq_len", type=int, default=150)
    ap.add_argument("--frames", default="")
    ap.add_argument("--support_frames", default="")
    ap.add_argument("--expressive_frames", default="")
    ap.add_argument("--auto_event_frames", action="store_true")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--support_k", type=int, default=5)
    ap.add_argument("--expressive_k", type=int, default=5)
    ap.add_argument("--source_gap", type=int, default=45)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="dhw4")
    ap.add_argument("--unit_space", choices=["normalized", "physical"], default="normalized")
    ap.add_argument("--support_lag", type=int, default=None)
    ap.add_argument("--expressive_lag", type=int, default=None)
    ap.add_argument("--min_gap", type=int, default=None)
    ap.add_argument("--edge_margin", type=int, default=None)
    ap.add_argument("--gate_sigma", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = np.load(args.rag_db, allow_pickle=True)

    event_report = None
    event_features_path = ""
    event_report_path = ""
    if args.auto_event_frames:
        cfg = TurnEventConfig.from_env(seq_len=args.seq_len, count=args.count)
        if args.support_lag is not None:
            cfg.support_lag = args.support_lag
        if args.expressive_lag is not None:
            cfg.expressive_lag = args.expressive_lag
        if args.min_gap is not None:
            cfg.min_gap = args.min_gap
        if args.edge_margin is not None:
            cfg.edge_margin = args.edge_margin
        if args.gate_sigma is not None:
            cfg.gate_sigma = args.gate_sigma
        event_report = detect_turn_events(args.trajectory, seq_len=args.seq_len, count=args.count, config=cfg)
        support_frames = list(map(int, event_report["support_frames"]))
        expressive_frames = list(map(int, event_report["expressive_frames"]))
        event_report_path = str(out_dir / f"{args.prefix}_turn_events.json")
        event_features_path = str(out_dir / f"{args.prefix}_turn_event_features.npy")
        save_event_report(event_report, event_report_path, event_features_path)
    else:
        common = parse_frames_or_even(args.frames, args.count, args.seq_len)
        support_frames = parse_int_list(args.support_frames) or common
        expressive_frames = parse_int_list(args.expressive_frames) or common
        support_frames = [int(np.clip(f, 0, args.seq_len - 1)) for f in support_frames[: args.count]]
        expressive_frames = [int(np.clip(f, 0, args.seq_len - 1)) for f in expressive_frames[: args.count]]

    traj = interp_traj(parse_points(args.trajectory), args.seq_len)
    feat = trajectory_features(traj)
    event_strength = np.clip(
        0.55 * feat["turn_norm"] + 0.30 * feat["speed_norm"] + 0.15 * feat["curvature_norm"],
        0.0,
        1.0,
    )

    support_indices: List[int] = []
    expressive_indices: List[int] = []
    used_support, used_expr = set(), set()

    for f in support_frames:
        score = score_contexts(db, float(event_strength[int(np.clip(f, 0, args.seq_len - 1))]), "support")
        support_indices.extend(greedy_select(score, db, 1, used_support, args.source_gap))
    for f in expressive_frames:
        score = score_contexts(db, float(event_strength[int(np.clip(f, 0, args.seq_len - 1))]), "expressive")
        expressive_indices.extend(greedy_select(score, db, 1, used_expr, args.source_gap))

    support_indices = support_indices[: args.support_k]
    expressive_indices = expressive_indices[: args.expressive_k]
    support_pose_paths, support_unit_paths = save_selected(
        args.prefix, "support", db, support_indices, support_frames, out_dir, args.unit_space
    )
    expressive_pose_paths, expressive_unit_paths = save_selected(
        args.prefix, "expressive", db, expressive_indices, expressive_frames, out_dir, args.unit_space
    )

    report = {
        "rag_db": args.rag_db,
        "prefix": args.prefix,
        "trajectory": args.trajectory,
        "auto_event_frames": bool(args.auto_event_frames),
        "support_frames": support_frames,
        "expressive_frames": expressive_frames,
        "unit_space": args.unit_space,
        "support_indices": support_indices,
        "expressive_indices": expressive_indices,
        "support_units": support_unit_paths,
        "expressive_units": expressive_unit_paths,
        "support_mid_poses": support_pose_paths,
        "expressive_mid_poses": expressive_pose_paths,
        "turn_event_report": event_report_path,
        "turn_event_features": event_features_path,
    }
    if event_report is not None:
        report["event_centers"] = event_report["event_centers"]

    report_path = out_dir / f"{args.prefix}_functional_context_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Backward-compatible common frames: expressive frames are used for RAG injection; compositor reads split frames.
    common_frames = expressive_frames
    env_lines = [
        f'export EDGE_SUPPORT_CONTEXT_UNIT_PATHS="{",".join(support_unit_paths)}"',
        f'export EDGE_SUPPORT_CONTEXT_MID_POSES="{",".join(support_pose_paths)}"',
        f'export EDGE_SUPPORT_CONTEXT_FRAMES="{",".join(map(str, support_frames))}"',
        f'export EDGE_EXPRESSIVE_CONTEXT_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
        f'export EDGE_EXPRESSIVE_CONTEXT_MID_POSES="{",".join(expressive_pose_paths)}"',
        f'export EDGE_EXPRESSIVE_CONTEXT_FRAMES="{",".join(map(str, expressive_frames))}"',
        f'export EDGE_FUNCTIONAL_CONTEXT_FRAMES="{",".join(map(str, common_frames))}"',
        f'export EDGE_RAG_CONTEXT_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
        f'export EDGE_RAG_SUMMARY_UNIT_PATHS="{",".join(expressive_unit_paths)}"',
    ]
    if event_features_path:
        env_lines.append(f'export EDGE_TURN_EVENT_FEATURES="{event_features_path}"')
        env_lines.append(f'export EDGE_TURN_EVENT_REPORT="{event_report_path}"')
    env_path = out_dir / f"{args.prefix}_functional_context.env"
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print(f"✅ Functional dual contexts exported: {out_dir}")
    print(f"   report={report_path}")
    print(f"   env={env_path}")
    print(f"   support_frames={support_frames}")
    print(f"   expressive_frames={expressive_frames}")
    if event_report is not None:
        print(f"   event_centers={event_report['event_centers']}")
    print(f"   support_units={len(support_unit_paths)} expressive_units={len(expressive_unit_paths)}")
    print("\nRun:")
    print(f"   source {env_path}")


if __name__ == "__main__":
    main()
