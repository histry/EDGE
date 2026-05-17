#!/usr/bin/env python3
"""Select functional dual contexts from an augmented ChoreoRAG DB.

Drop-in replacement with in-the-wild video music-motion sync support.

New fields used when present:
  video_music_sync_score
  video_music_sync_score_norm
  video_expressive_sync_score
  video_support_sync_score
  motion_highfreq_score_norm
  is_inwild_video

Environment flags:
  EDGE_ENABLE_INWILD_VIDEO_RAG=1
  EDGE_VIDEO_SYNC_CONTEXT_WEIGHT=0.35
  EDGE_VIDEO_SYNC_EVENT_GAIN=0.50
  EDGE_VIDEO_SYNC_REQUIRE_RIGHTS=0
  EDGE_VIDEO_SYNC_ALLOWED_RIGHTS=owned_or_permitted,cc,research_permitted

The selector still outputs the same env files as before:
  EDGE_SUPPORT_CONTEXT_UNIT_PATHS
  EDGE_EXPRESSIVE_CONTEXT_UNIT_PATHS
  EDGE_RAG_CONTEXT_UNIT_PATHS
  EDGE_RAG_SUMMARY_UNIT_PATHS
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


_TRUE = {"1", "true", "yes", "y", "on"}


def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in _TRUE


def env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def env_list(name, default):
    v = os.environ.get(name)
    if v is None:
        return list(default)
    return [x.strip() for x in str(v).split(",") if x.strip()]


def norm01(x):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)


def field(db, name, default=0.0):
    if name in db.files:
        arr = np.asarray(db[name])
        if arr.dtype.kind in {"U", "S", "O"}:
            return arr
        return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if "unit_motions" in db.files:
        n = len(db["unit_motions"])
    elif "poses" in db.files:
        n = len(db["poses"])
    else:
        raise KeyError("Cannot infer DB length; expected unit_motions or poses")
    return np.full((n,), float(default), dtype=np.float32)


def parse_points(text):
    pts = []
    for item in str(text).replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) < 2:
            continue
        pts.append([float(parts[0]), float(parts[1])])
    if len(pts) < 2:
        pts = [[0.0, 0.0], [0.0, 0.0]]
    return np.asarray(pts, dtype=np.float32)


def interp_traj(points, seq_len):
    pts = parse_points(points) if isinstance(points, str) else np.asarray(points, dtype=np.float32)
    if pts.shape[0] == 1:
        return np.repeat(pts[:1, :2], int(seq_len), axis=0).astype(np.float32)
    xp = np.linspace(0, int(seq_len) - 1, pts.shape[0], dtype=np.float32)
    xq = np.arange(int(seq_len), dtype=np.float32)
    return np.stack([
        np.interp(xq, xp, pts[:, 0]),
        np.interp(xq, xp, pts[:, 1]),
    ], axis=-1).astype(np.float32)


def smooth1d(x, radius=2):
    x = np.asarray(x, dtype=np.float32)
    if radius <= 0 or len(x) < 3:
        return x
    k = np.ones((2 * radius + 1,), dtype=np.float32)
    k /= k.sum()
    pad = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(pad, k, mode="valid").astype(np.float32)


def trajectory_features(traj):
    traj = np.asarray(traj, dtype=np.float32)[..., :2]
    T = len(traj)
    vel = np.zeros_like(traj)
    if T > 1:
        vel[1:] = traj[1:] - traj[:-1]
        vel[0] = vel[1]
    speed = np.sqrt(np.sum(vel * vel, axis=-1) + 1e-8).astype(np.float32)
    speed_s = smooth1d(speed, radius=2)
    heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)
    dhead = np.zeros((T,), dtype=np.float32)
    if T > 1:
        raw = heading[1:] - heading[:-1]
        dhead[1:] = np.arctan2(np.sin(raw), np.cos(raw))
        dhead[0] = dhead[1]
    turning = np.abs(dhead).astype(np.float32)
    curvature = smooth1d(turning / (speed_s + 1e-4), radius=2)
    acc = np.zeros_like(speed_s)
    if T > 1:
        acc[1:] = speed_s[1:] - speed_s[:-1]
        acc[0] = acc[1]
    return {
        "speed_norm": norm01(speed_s),
        "turn_norm": norm01(turning),
        "curvature_norm": norm01(curvature),
        "acc_norm": norm01(np.abs(acc)),
    }


def gaussian_gate(T, centers, sigma=5.0):
    t = np.arange(int(T), dtype=np.float32)
    out = np.zeros((int(T),), dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    for c in centers:
        c = float(np.clip(int(c), 0, int(T) - 1))
        out = np.maximum(out, np.exp(-0.5 * ((t - c) / sigma) ** 2).astype(np.float32))
    return out


def detect_events_from_trajectory(traj, count=5, support_lag=8, expressive_lag=4, min_gap=18, gate_sigma=5.0):
    feat = trajectory_features(traj)
    T = len(traj)
    score = 0.60 * feat["curvature_norm"] + 0.25 * feat["speed_norm"] + 0.15 * feat["acc_norm"]
    score = score.astype(np.float32)
    if T > 8:
        score[:4] = -1.0
        score[-4:] = -1.0
    order = list(np.argsort(score)[::-1])
    centers = []
    for idx in order:
        idx = int(idx)
        if score[idx] < 0:
            continue
        if all(abs(idx - c) >= int(min_gap) for c in centers):
            centers.append(idx)
        if len(centers) >= int(count):
            break
    if len(centers) < int(count):
        uniform = np.linspace(0.22 * (T - 1), 0.78 * (T - 1), int(count)).round().astype(int).tolist()
        for u in uniform:
            if all(abs(int(u) - c) >= max(4, int(min_gap) // 2) for c in centers):
                centers.append(int(u))
            if len(centers) >= int(count):
                break
    centers = sorted(centers[: int(count)])
    support = [int(np.clip(c - int(support_lag), 0, T - 1)) for c in centers]
    expressive = [int(np.clip(c + int(expressive_lag), 0, T - 1)) for c in centers]
    event_mat = np.stack([
        feat["speed_norm"],
        feat["turn_norm"],
        feat["curvature_norm"],
        gaussian_gate(T, centers, gate_sigma),
        gaussian_gate(T, support, gate_sigma),
        gaussian_gate(T, expressive, gate_sigma),
    ], axis=-1).astype(np.float32)
    report = {
        "event_centers": centers,
        "support_frames": support,
        "expressive_frames": expressive,
        "seq_len": T,
        "support_lag": int(support_lag),
        "expressive_lag": int(expressive_lag),
        "min_gap": int(min_gap),
        "gate_sigma": float(gate_sigma),
    }
    return report, event_mat, feat


def parse_int_list(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [int(x) for x in text]
    out = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if item:
            out.append(int(float(item)))
    return out


def parse_frames_or_even(text, count, seq_len):
    frames = parse_int_list(text)
    if frames:
        return [int(np.clip(f, 0, seq_len - 1)) for f in frames[:count]]
    return [int(round((i + 1) * seq_len / (count + 1))) for i in range(count)]


def rights_mask(db):
    n = len(field(db, "is_inwild_video", 0.0))
    if not env_bool("EDGE_VIDEO_SYNC_REQUIRE_RIGHTS", False):
        return np.ones((n,), dtype=np.float32)
    allowed = set(env_list("EDGE_VIDEO_SYNC_ALLOWED_RIGHTS", ["owned_or_permitted", "cc", "research_permitted"]))
    if "rights_tag" not in db.files:
        return np.zeros((n,), dtype=np.float32)
    tags = db["rights_tag"].astype(str)
    return np.asarray([1.0 if t in allowed else 0.0 for t in tags], dtype=np.float32)


def video_bonus(db, role, event_strength):
    if not env_bool("EDGE_ENABLE_INWILD_VIDEO_RAG", False):
        return np.zeros_like(field(db, "mobile_score", 0.0), dtype=np.float32)

    w = env_float("EDGE_VIDEO_SYNC_CONTEXT_WEIGHT", 0.35)
    event_gain = env_float("EDGE_VIDEO_SYNC_EVENT_GAIN", 0.50)
    inwild = field(db, "is_inwild_video", 0.0)
    rmask = rights_mask(db)
    event_scale = float(w) * (1.0 + float(event_gain) * float(event_strength))

    if role == "support":
        base = (
            0.45 * field(db, "video_support_sync_score", 0.0)
            + 0.30 * field(db, "video_music_sync_score_norm", 0.0)
            + 0.25 * field(db, "video_onset_peak_score_norm", 0.0)
        )
    else:
        base = (
            0.45 * field(db, "video_expressive_sync_score", 0.0)
            + 0.30 * field(db, "video_music_sync_score_norm", 0.0)
            + 0.25 * field(db, "motion_highfreq_score_norm", 0.0)
        )
    return event_scale * base * inwild * rmask


def score_contexts(db, event_strength, role):
    support_base = (
        0.42 * field(db, "support_context_score", 0.0)
        + 0.20 * field(db, "mobile_score", 0.0)
        + 0.22 * field(db, "footstep_score", 0.0)
        + 0.16 * field(db, "speed_lower_sync", 0.0)
    )
    expressive_base = (
        0.34 * field(db, "expressive_mobile_score", 0.0)
        + 0.22 * field(db, "mobile_expressive_score", 0.0)
        + 0.22 * field(db, "functional_coupling_score", 0.0)
        + 0.22 * field(db, "turn_expression_response", 0.0)
    )

    if role == "support":
        score = support_base + float(event_strength) * (
            0.50 * field(db, "support_context_score", 0.0)
            + 0.25 * field(db, "contact_switch_norm", 0.0)
            + 0.25 * field(db, "speed_lower_sync_norm", 0.0)
        )
        score = score + video_bonus(db, "support", event_strength)
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    if role == "expressive":
        score = expressive_base + float(event_strength) * (
            0.45 * field(db, "turn_expression_response_norm", 0.0)
            + 0.30 * field(db, "expressive_mobile_score", 0.0)
            + 0.25 * field(db, "speed_expression_sync_norm", 0.0)
        )
        score = score + video_bonus(db, "expressive", event_strength)
        return np.nan_to_num(score.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    raise ValueError(role)


def greedy_select(score, db, k, used, source_gap=0):
    score = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(-score)
    sources = db["source"].astype(str) if "source" in db.files else np.asarray([""] * len(score))
    if "unit_center" in db.files:
        centers = np.asarray(db["unit_center"])
    elif "source_frame" in db.files:
        centers = np.asarray(db["source_frame"])
    else:
        centers = np.arange(len(score))
    out = []
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
        if len(out) >= int(k):
            break
    return out


def save_selected(prefix, role, db, indices, frames, out_dir, unit_space):
    unit_key = "unit_motions" if unit_space == "normalized" else (
        "unit_motions_physical" if "unit_motions_physical" in db.files else "unit_motions"
    )
    pose_key = "poses" if unit_space == "normalized" else (
        "center_pose_physical" if "center_pose_physical" in db.files else "poses"
    )
    if pose_key not in db.files:
        pose_key = "poses"
    units = db[unit_key]
    poses = db[pose_key]
    unit_paths = []
    pose_paths = []
    for n, idx in enumerate(indices, start=1):
        frame = int(frames[min(n - 1, len(frames) - 1)])
        pose_path = out_dir / ("%s_%s_mid%02d_f%d.npy" % (prefix, role, n, frame))
        unit_path = out_dir / ("%s_%s_mid%02d_f%d_unit.npy" % (prefix, role, n, frame))
        np.save(pose_path, np.asarray(poses[int(idx)], dtype=np.float32))
        np.save(unit_path, np.asarray(units[int(idx)], dtype=np.float32))
        pose_paths.append(str(pose_path))
        unit_paths.append(str(unit_path))
    return pose_paths, unit_paths


def main():
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
    ap.add_argument("--prefix", default="dhw")
    ap.add_argument("--unit_space", choices=["normalized", "physical"], default="normalized")
    ap.add_argument("--support_lag", type=int, default=8)
    ap.add_argument("--expressive_lag", type=int, default=4)
    ap.add_argument("--min_gap", type=int, default=18)
    ap.add_argument("--gate_sigma", type=float, default=5.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db = np.load(args.rag_db, allow_pickle=True)

    traj = interp_traj(args.trajectory, args.seq_len)
    if args.auto_event_frames:
        event_report, event_mat, feat = detect_events_from_trajectory(
            traj,
            count=args.count,
            support_lag=args.support_lag,
            expressive_lag=args.expressive_lag,
            min_gap=args.min_gap,
            gate_sigma=args.gate_sigma,
        )
        support_frames = list(map(int, event_report["support_frames"]))
        expressive_frames = list(map(int, event_report["expressive_frames"]))
        event_report_path = str(out_dir / ("%s_turn_events.json" % args.prefix))
        event_features_path = str(out_dir / ("%s_turn_event_features.npy" % args.prefix))
        np.save(event_features_path, event_mat.astype(np.float32))
        Path(event_report_path).write_text(json.dumps(event_report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        common = parse_frames_or_even(args.frames, args.count, args.seq_len)
        support_frames = parse_int_list(args.support_frames) or common
        expressive_frames = parse_int_list(args.expressive_frames) or common
        support_frames = [int(np.clip(f, 0, args.seq_len - 1)) for f in support_frames[:args.count]]
        expressive_frames = [int(np.clip(f, 0, args.seq_len - 1)) for f in expressive_frames[:args.count]]
        event_report = None
        event_report_path = ""
        event_features_path = ""
        feat = trajectory_features(traj)

    event_strength = np.clip(
        0.55 * feat["turn_norm"] + 0.30 * feat["speed_norm"] + 0.15 * feat["curvature_norm"],
        0.0,
        1.0,
    )

    support_indices = []
    expressive_indices = []
    used_support, used_expr = set(), set()

    for f in support_frames:
        s = score_contexts(db, float(event_strength[int(np.clip(f, 0, args.seq_len - 1))]), "support")
        support_indices.extend(greedy_select(s, db, 1, used_support, args.source_gap))
    for f in expressive_frames:
        s = score_contexts(db, float(event_strength[int(np.clip(f, 0, args.seq_len - 1))]), "expressive")
        expressive_indices.extend(greedy_select(s, db, 1, used_expr, args.source_gap))

    support_indices = support_indices[:args.support_k]
    expressive_indices = expressive_indices[:args.expressive_k]

    support_pose_paths, support_unit_paths = save_selected(
        args.prefix, "support", db, support_indices, support_frames, out_dir, args.unit_space
    )
    expressive_pose_paths, expressive_unit_paths = save_selected(
        args.prefix, "expressive", db, expressive_indices, expressive_frames, out_dir, args.unit_space
    )

    def selected_rows(indices):
        rows = []
        for idx in indices:
            idx = int(idx)
            rows.append({
                "idx": idx,
                "source": str(db["source"][idx]) if "source" in db.files else "",
                "title": str(db["title"][idx]) if "title" in db.files else "",
                "rights_tag": str(db["rights_tag"][idx]) if "rights_tag" in db.files else "",
                "is_inwild_video": float(field(db, "is_inwild_video", 0.0)[idx]),
                "video_music_sync_score": float(field(db, "video_music_sync_score", 0.0)[idx]),
                "motion_highfreq_score": float(field(db, "motion_highfreq_score", 0.0)[idx]),
            })
        return rows

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
        "support_selected": selected_rows(support_indices),
        "expressive_selected": selected_rows(expressive_indices),
        "support_units": support_unit_paths,
        "expressive_units": expressive_unit_paths,
        "support_mid_poses": support_pose_paths,
        "expressive_mid_poses": expressive_pose_paths,
        "turn_event_report": event_report_path,
        "turn_event_features": event_features_path,
        "EDGE_ENABLE_INWILD_VIDEO_RAG": os.environ.get("EDGE_ENABLE_INWILD_VIDEO_RAG", "0"),
        "EDGE_VIDEO_SYNC_CONTEXT_WEIGHT": os.environ.get("EDGE_VIDEO_SYNC_CONTEXT_WEIGHT", ""),
    }
    if event_report is not None:
        report["event_centers"] = event_report["event_centers"]

    report_path = out_dir / ("%s_functional_context_report.json" % args.prefix)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    common_frames = expressive_frames
    env_lines = [
        'export EDGE_SUPPORT_CONTEXT_UNIT_PATHS="%s"' % ",".join(support_unit_paths),
        'export EDGE_SUPPORT_CONTEXT_MID_POSES="%s"' % ",".join(support_pose_paths),
        'export EDGE_SUPPORT_CONTEXT_FRAMES="%s"' % ",".join(map(str, support_frames)),
        'export EDGE_EXPRESSIVE_CONTEXT_UNIT_PATHS="%s"' % ",".join(expressive_unit_paths),
        'export EDGE_EXPRESSIVE_CONTEXT_MID_POSES="%s"' % ",".join(expressive_pose_paths),
        'export EDGE_EXPRESSIVE_CONTEXT_FRAMES="%s"' % ",".join(map(str, expressive_frames)),
        'export EDGE_FUNCTIONAL_CONTEXT_FRAMES="%s"' % ",".join(map(str, common_frames)),
        'export EDGE_RAG_CONTEXT_UNIT_PATHS="%s"' % ",".join(expressive_unit_paths),
        'export EDGE_RAG_SUMMARY_UNIT_PATHS="%s"' % ",".join(expressive_unit_paths),
    ]
    if event_features_path:
        env_lines.append('export EDGE_TURN_EVENT_FEATURES="%s"' % event_features_path)
        env_lines.append('export EDGE_TURN_EVENT_REPORT="%s"' % event_report_path)

    env_path = out_dir / ("%s_functional_context.env" % args.prefix)
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    print("✅ Functional dual contexts exported:", out_dir)
    print("   report=", report_path)
    print("   env=", env_path)
    print("   support_frames=", support_frames)
    print("   expressive_frames=", expressive_frames)
    if event_report is not None:
        print("   event_centers=", event_report["event_centers"])
    print("   support_units=%d expressive_units=%d" % (len(support_unit_paths), len(expressive_unit_paths)))
    if env_bool("EDGE_ENABLE_INWILD_VIDEO_RAG", False):
        print("   inwild video RAG enabled, weight=", env_float("EDGE_VIDEO_SYNC_CONTEXT_WEIGHT", 0.35))
    print("\nRun:")
    print("   source %s" % env_path)


if __name__ == "__main__":
    main()
