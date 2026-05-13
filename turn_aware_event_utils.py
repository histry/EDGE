#!/usr/bin/env python3
"""Turn-aware event utilities for EDGE functional ChoreoRAG.

This module replaces manual frame sweep with trajectory-derived event timing.
It is intentionally independent of the EDGE model internals, so it can be used
by selector/compositor/evaluator/training scripts.

EDGE motion contract:
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
CONTACT_SLICE_FALLBACK = slice(0, 4)

try:  # Prefer project-defined body groups when available.
    from footstep_phase_utils import (  # type: ignore
        CONTACT_SLICE,
        LOWER_ROT_INDEX,
        TORSO_ROT_INDEX,
        UPPER_ROT_INDEX,
    )
except Exception:  # Robust fallback for standalone usage.
    CONTACT_SLICE = CONTACT_SLICE_FALLBACK
    _rot = np.arange(7, 151, dtype=np.int64).reshape(24, 6)
    _lower_joints = [1, 2, 4, 5, 7, 8, 10, 11]
    _torso_joints = [0, 3, 6, 9, 12]
    _upper_joints = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    LOWER_ROT_INDEX = _rot[_lower_joints].reshape(-1)
    TORSO_ROT_INDEX = _rot[_torso_joints].reshape(-1)
    UPPER_ROT_INDEX = _rot[_upper_joints].reshape(-1)


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


def parse_int_list(text: Optional[str]) -> List[int]:
    if text is None:
        return []
    return [int(round(float(x))) for x in str(text).replace(";", ",").split(",") if x.strip()]


def list_to_csv(xs: Sequence[int]) -> str:
    return ",".join(str(int(x)) for x in xs)


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
        fill = [int(round((i + 1) * T / (count + 1))) for i in range(count)]
        for f in fill:
            f = int(np.clip(f, edge_margin, max(edge_margin, T - edge_margin - 1)))
            if all(abs(f - j) >= max(4, min_gap // 2) for j in out):
                out.append(f)
            if len(out) >= count:
                break
    return sorted(out[:count])


def detect_turn_events(trajectory: str | np.ndarray, cfg: TurnEventConfig) -> Dict[str, object]:
    if isinstance(trajectory, str):
        traj = interp_traj(parse_points(trajectory), cfg.seq_len)
    else:
        traj = np.asarray(trajectory, dtype=np.float32)
        if len(traj) != cfg.seq_len:
            # Reinterpolate to cfg.seq_len by treating input rows as control samples.
            traj = interp_traj(traj[:, :2], cfg.seq_len)
    feat = trajectory_features(traj)
    score = (
        cfg.speed_weight * feat["speed_norm"]
        + cfg.turn_weight * feat["turn_norm"]
        + cfg.curvature_weight * feat["curvature_norm"]
    )
    if not cfg.include_speed_peaks:
        score = cfg.turn_weight * feat["turn_norm"] + cfg.curvature_weight * feat["curvature_norm"]
    centers = greedy_peak_select(score, cfg.count, cfg.min_gap, cfg.edge_margin)
    support_frames = [int(np.clip(c - cfg.support_lag, 0, cfg.seq_len - 1)) for c in centers]
    expressive_frames = [int(np.clip(c + cfg.expressive_lag, 0, cfg.seq_len - 1)) for c in centers]
    settle_frames = [int(np.clip(c + cfg.settle_lag, 0, cfg.seq_len - 1)) for c in centers]
    gates = {
        "turn_gate": gaussian_gate(cfg.seq_len, centers, cfg.gate_sigma),
        "support_gate": gaussian_gate(cfg.seq_len, support_frames, cfg.gate_sigma),
        "expressive_gate": gaussian_gate(cfg.seq_len, expressive_frames, cfg.gate_sigma),
        "settle_gate": gaussian_gate(cfg.seq_len, settle_frames, cfg.gate_sigma),
        "speed_gate": norm01(feat["speed"]),
    }
    return {
        "config": asdict(cfg),
        "trajectory": traj,
        "features": feat,
        "score": score.astype(np.float32),
        "event_centers": centers,
        "support_frames": support_frames,
        "expressive_frames": expressive_frames,
        "settle_frames": settle_frames,
        "gates": gates,
    }


def event_feature_matrix(trajectory: str | np.ndarray, cfg: Optional[TurnEventConfig] = None) -> Tuple[np.ndarray, List[str], Dict[str, object]]:
    if cfg is None:
        if isinstance(trajectory, np.ndarray):
            seq_len = int(len(trajectory))
        else:
            seq_len = 150
        cfg = TurnEventConfig.from_env(seq_len=seq_len)
    ev = detect_turn_events(trajectory, cfg)
    feat: Dict[str, np.ndarray] = ev["features"]  # type: ignore
    gates: Dict[str, np.ndarray] = ev["gates"]  # type: ignore
    names = [
        "speed_norm",
        "turn_norm",
        "curvature_norm",
        "acceleration_norm",
        "heading_sin",
        "heading_cos",
        "turn_gate",
        "support_gate",
        "expressive_gate",
        "settle_gate",
        "speed_gate",
    ]
    cols = [
        feat["speed_norm"],
        feat["turn_norm"],
        feat["curvature_norm"],
        feat["acceleration_norm"],
        feat["heading_sin"],
        feat["heading_cos"],
        gates["turn_gate"],
        gates["support_gate"],
        gates["expressive_gate"],
        gates["settle_gate"],
        gates["speed_gate"],
    ]
    mat = np.stack(cols, axis=-1).astype(np.float32)
    return mat, names, ev


def save_event_report(path: str | Path, event: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        "config": event.get("config"),
        "event_centers": event.get("event_centers"),
        "support_frames": event.get("support_frames"),
        "expressive_frames": event.get("expressive_frames"),
        "settle_frames": event.get("settle_frames"),
    }
    path.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
