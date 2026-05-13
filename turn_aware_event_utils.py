#!/usr/bin/env python3
"""Turn-aware event utilities for EDGE functional ChoreoRAG.

This file turns the manual frame sweep into a deterministic event detector.
It is intentionally independent from the EDGE model internals, so it can be
used by selector/compositor/evaluator/training scripts.

Motion contract used by EDGE:
  [0:4]   foot contacts
  [4:7]   root xyz
  [7:151] 24 joints * 6D rotation
Ground-plane trajectory is X/Z.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ROOT_X_IDX = 4
ROOT_Z_IDX = 6


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


@dataclass
class TurnEventConfig:
    seq_len: int = 150
    count: int = 5
    support_lag: int = 8
    expressive_lag: int = 4
    settle_lag: int = 12
    min_gap: int = 18
    edge_margin: int = 12
    speed_weight: float = 0.30
    turn_weight: float = 0.55
    curvature_weight: float = 0.15
    gate_sigma: float = 5.0
    include_speed_peaks: bool = True

    @classmethod
    def from_env(cls, seq_len: int = 150, count: int = 5) -> "TurnEventConfig":
        return cls(
            seq_len=seq_len,
            count=count,
            support_lag=_env_int("EDGE_TURN_SUPPORT_LAG", 8),
            expressive_lag=_env_int("EDGE_TURN_EXPR_LAG", 4),
            settle_lag=_env_int("EDGE_TURN_SETTLE_LAG", 12),
            min_gap=_env_int("EDGE_TURN_MIN_GAP", 18),
            edge_margin=_env_int("EDGE_TURN_EDGE_MARGIN", 12),
            speed_weight=_env_float("EDGE_TURN_SPEED_WEIGHT", 0.30),
            turn_weight=_env_float("EDGE_TURN_TURN_WEIGHT", 0.55),
            curvature_weight=_env_float("EDGE_TURN_CURV_WEIGHT", 0.15),
            gate_sigma=_env_float("EDGE_TURN_GATE_SIGMA", 5.0),
            include_speed_peaks=os.environ.get("EDGE_TURN_INCLUDE_SPEED", "1") != "0",
        )


def parse_points(text: str) -> np.ndarray:
    pts: List[List[float]] = []
    for item in str(text).replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        xs = [x.strip() for x in item.split(",") if x.strip()]
        if len(xs) < 2:
            raise ValueError(f"Bad trajectory point: {item!r}")
        pts.append([float(xs[0]), float(xs[1])])
    if len(pts) < 2:
        raise ValueError("Trajectory must contain at least two points, e.g. '0,0;1,1'.")
    return np.asarray(pts, dtype=np.float32)


def parse_int_list(text: str) -> List[int]:
    if text is None:
        return []
    return [int(round(float(x))) for x in str(text).replace(";", ",").split(",") if x.strip()]


def interp_traj(points: np.ndarray, seq_len: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    src = np.linspace(0.0, 1.0, len(points))
    dst = np.linspace(0.0, 1.0, int(seq_len))
    x = np.interp(dst, src, points[:, 0])
    z = np.interp(dst, src, points[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)


def norm01(x: np.ndarray, lo_q: float = 10.0, hi_q: float = 90.0) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return x.astype(np.float32)
    lo, hi = np.percentile(x, [lo_q, hi_q])
    if float(hi - lo) <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / float(hi - lo), 0.0, 1.0).astype(np.float32)


def trajectory_features(traj: np.ndarray) -> Dict[str, np.ndarray]:
    """Return per-frame speed/heading/turn/curvature features for [T,2] X/Z trajectory."""
    traj = np.asarray(traj, dtype=np.float32)
    T = len(traj)
    vel = np.zeros((T, 2), dtype=np.float32)
    if T > 1:
        vel[1:] = traj[1:] - traj[:-1]
        vel[0] = vel[1]
    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    heading = np.arctan2(vel[:, 1], vel[:, 0]).astype(np.float32)
    heading_unwrapped = np.unwrap(heading).astype(np.float32)
    heading_delta = np.zeros((T,), dtype=np.float32)
    if T > 1:
        heading_delta[1:] = np.abs(np.diff(heading_unwrapped))
        heading_delta[0] = heading_delta[1]
    curvature = heading_delta / np.clip(speed, 1e-6, None)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    acceleration = np.zeros((T,), dtype=np.float32)
    if T > 1:
        acceleration[1:] = np.abs(np.diff(speed))
        acceleration[0] = acceleration[1]
    return {
        "traj_x": traj[:, 0].astype(np.float32),
        "traj_z": traj[:, 1].astype(np.float32),
        "vel_x": vel[:, 0].astype(np.float32),
        "vel_z": vel[:, 1].astype(np.float32),
        "speed": speed,
        "speed_norm": norm01(speed),
        "heading": heading,
        "heading_sin": np.sin(heading).astype(np.float32),
        "heading_cos": np.cos(heading).astype(np.float32),
        "heading_delta": heading_delta,
        "turn_norm": norm01(heading_delta),
        "curvature": curvature,
        "curvature_norm": norm01(curvature),
        "acceleration": acceleration,
        "acceleration_norm": norm01(acceleration),
    }


def gaussian_gate(T: int, centers: Sequence[int], sigma: float = 5.0) -> np.ndarray:
    xs = np.arange(T, dtype=np.float32)
    gate = np.zeros((T,), dtype=np.float32)
    sigma = max(float(sigma), 1e-3)
    for c in centers:
        gate = np.maximum(gate, np.exp(-0.5 * ((xs - int(c)) / sigma) ** 2).astype(np.float32))
    return np.clip(gate, 0.0, 1.0).astype(np.float32)


def greedy_peak_select(score: np.ndarray, count: int, min_gap: int, edge_margin: int) -> List[int]:
    score = np.nan_to_num(np.asarray(score, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    T = len(score)
    if T == 0:
        return []
    valid = np.ones((T,), dtype=bool)
    valid[: max(0, edge_margin)] = False
    valid[min(T, T - max(0, edge_margin)) :] = False
    order = np.argsort(-score)
    out: List[int] = []
    for idx in order:
        idx = int(idx)
        if not valid[idx]:
            continue
        if any(abs(idx - j) < min_gap for j in out):
            continue
        out.append(idx)
        if len(out) >= count:
            break
    if len(out) < count:
        # Fill with evenly spaced frames when trajectory has too few peaks.
        fallback = [int(round((i + 1) * T / (count + 1))) for i in range(count)]
        for f in fallback:
            f = int(np.clip(f, edge_margin, max(edge_margin, T - edge_margin - 1)))
            if all(abs(f - j) >= max(4, min_gap // 2) for j in out):
                out.append(f)
            if len(out) >= count:
                break
    return sorted(out[:count])


def detect_turn_events(
    trajectory: str | np.ndarray,
    seq_len: int = 150,
    count: int = 5,
    config: Optional[TurnEventConfig] = None,
) -> Dict[str, object]:
    cfg = config or TurnEventConfig.from_env(seq_len=seq_len, count=count)
    if isinstance(trajectory, str):
        traj = interp_traj(parse_points(trajectory), seq_len)
        traj_text = trajectory
    else:
        traj = np.asarray(trajectory, dtype=np.float32)
        if len(traj) != seq_len:
            src = np.linspace(0, 1, len(traj))
            dst = np.linspace(0, 1, seq_len)
            traj = np.stack([
                np.interp(dst, src, traj[:, 0]),
                np.interp(dst, src, traj[:, 1]),
            ], axis=-1).astype(np.float32)
        traj_text = "<array>"
    feat = trajectory_features(traj)
    event_score = (
        cfg.speed_weight * feat["speed_norm"]
        + cfg.turn_weight * feat["turn_norm"]
        + cfg.curvature_weight * feat["curvature_norm"]
    ).astype(np.float32)
    centers = greedy_peak_select(event_score, cfg.count, cfg.min_gap, cfg.edge_margin)
    T = seq_len
    support_frames = [int(np.clip(c - cfg.support_lag, 0, T - 1)) for c in centers]
    expressive_frames = [int(np.clip(c + cfg.expressive_lag, 0, T - 1)) for c in centers]
    settle_frames = [int(np.clip(c + cfg.settle_lag, 0, T - 1)) for c in centers]
    gates = {
        "turn_gate": gaussian_gate(T, centers, sigma=cfg.gate_sigma),
        "support_prepare_gate": gaussian_gate(T, support_frames, sigma=cfg.gate_sigma),
        "expressive_response_gate": gaussian_gate(T, expressive_frames, sigma=cfg.gate_sigma),
        "settle_gate": gaussian_gate(T, settle_frames, sigma=cfg.gate_sigma),
    }
    event_features = np.stack(
        [
            feat["speed_norm"],
            feat["heading_sin"],
            feat["heading_cos"],
            feat["turn_norm"],
            feat["curvature_norm"],
            feat["acceleration_norm"],
            gates["turn_gate"],
            gates["support_prepare_gate"],
            gates["expressive_response_gate"],
            gates["settle_gate"],
        ],
        axis=-1,
    ).astype(np.float32)
    return {
        "trajectory": traj_text,
        "seq_len": int(seq_len),
        "config": asdict(cfg),
        "event_centers": centers,
        "support_frames": support_frames,
        "expressive_frames": expressive_frames,
        "settle_frames": settle_frames,
        "event_score": event_score,
        "features": feat,
        "gates": gates,
        "event_features": event_features,
    }


def save_event_report(report: Dict[str, object], out_json: str | Path, out_npy: Optional[str | Path] = None) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    clean = {
        k: v
        for k, v in report.items()
        if k not in {"features", "gates", "event_score", "event_features"}
    }
    clean["event_feature_dim"] = int(np.asarray(report["event_features"]).shape[-1])
    out_json.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_npy is not None:
        np.save(out_npy, np.asarray(report["event_features"], dtype=np.float32))


def load_event_features(path: str | Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"event features must be [T,D], got {arr.shape}")
    return arr


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--seq_len", type=int, default=150)
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_npy", default="")
    args = ap.parse_args()
    rep = detect_turn_events(args.trajectory, seq_len=args.seq_len, count=args.count)
    save_event_report(rep, args.out_json, args.out_npy or None)
    print(f"✅ turn-aware event report saved: {args.out_json}")
    if args.out_npy:
        print(f"✅ event features saved: {args.out_npy}")
    print("event_centers:", rep["event_centers"])
    print("support_frames:", rep["support_frames"])
    print("expressive_frames:", rep["expressive_frames"])
