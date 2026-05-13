"""Turn-aware trajectory event utilities for EDGE / Dunhuang ChoreoRAG.

This module is intentionally dependency-light and safe to import from runtime
patches.  It provides both numpy and torch implementations of the same event
features so the features used in no-train selection, adapter training and model
inference stay consistent.

Feature layout returned by event_feature_matrix / event_feature_matrix_torch:
  0 x_norm
  1 z_norm
  2 speed_norm
  3 heading_sin
  4 heading_cos
  5 curvature_norm
  6 turn_gate
  7 support_gate       (typically before turn event)
  8 expressive_gate    (typically after turn event)
  9 acceleration_norm
 10 speed_gate
 11 curvature_signed
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    F = None

EVENT_DIM = 12


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def parse_trajectory_string(text: str, seq_len: int = 150) -> np.ndarray:
    """Parse "x,z;x,z;..." and linearly interpolate to [T,2]."""
    pts: List[Tuple[float, float]] = []
    for item in str(text).replace("|", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) < 2:
            continue
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) == 0:
        return np.zeros((seq_len, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(np.asarray(pts, dtype=np.float32), seq_len, axis=0)
    pts_np = np.asarray(pts, dtype=np.float32)
    xp = np.linspace(0, seq_len - 1, len(pts_np), dtype=np.float32)
    xq = np.arange(seq_len, dtype=np.float32)
    out = np.stack([
        np.interp(xq, xp, pts_np[:, 0]),
        np.interp(xq, xp, pts_np[:, 1]),
    ], axis=-1).astype(np.float32)
    return out


def _smooth_np(x: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or x.shape[0] < 3:
        return x
    k = np.ones((radius * 2 + 1,), dtype=np.float32)
    k /= k.sum()
    pad = np.pad(x, (radius, radius), mode="edge")
    return np.convolve(pad, k, mode="valid").astype(np.float32)


def _angle_diff_np(a: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    if len(a) > 1:
        raw = a[1:] - a[:-1]
        d[1:] = np.arctan2(np.sin(raw), np.cos(raw))
        d[0] = d[1]
    return d.astype(np.float32)


def _gaussian_gate_np(T: int, centers: Sequence[int], sigma: float = 5.0) -> np.ndarray:
    if T <= 0:
        return np.zeros((0,), dtype=np.float32)
    t = np.arange(T, dtype=np.float32)
    gate = np.zeros((T,), dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    for c in centers:
        c = float(np.clip(int(c), 0, T - 1))
        gate = np.maximum(gate, np.exp(-0.5 * ((t - c) / sigma) ** 2).astype(np.float32))
    return gate.clip(0.0, 1.0)


def _topk_peaks(score: np.ndarray, k: int, min_gap: int = 18, pad: int = 4) -> List[int]:
    score = np.asarray(score, dtype=np.float32).copy()
    T = int(score.shape[0])
    if T == 0 or k <= 0:
        return []
    if pad > 0:
        score[:pad] = -np.inf
        score[-pad:] = -np.inf
    order = list(np.argsort(score)[::-1])
    chosen: List[int] = []
    for idx in order:
        if not np.isfinite(score[idx]):
            continue
        idx = int(idx)
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
            if len(chosen) >= k:
                break
    if not chosen:
        chosen = [T // 2]
    chosen = sorted(chosen)
    return chosen


@dataclass
class TurnEventReport:
    event_centers: List[int]
    support_frames: List[int]
    expressive_frames: List[int]
    speed_peaks: List[int]
    turn_score_mean: float
    turn_score_max: float
    speed_mean: float
    speed_max: float
    seq_len: int
    support_lag: int
    expressive_lag: int
    min_gap: int

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def trajectory_event_scores(traj: np.ndarray) -> Dict[str, np.ndarray]:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[-1] < 2:
        raise ValueError(f"trajectory must be [T,2+], got {traj.shape}")
    traj = traj[..., :2]
    T = traj.shape[0]
    vel = np.zeros_like(traj)
    if T > 1:
        vel[1:] = traj[1:] - traj[:-1]
        vel[0] = vel[1]
    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    speed_s = _smooth_np(speed, radius=2)
    heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)
    heading_delta = _angle_diff_np(heading)
    turning = np.abs(heading_delta).astype(np.float32)
    curvature = turning / (speed_s + 1e-4)
    curvature = _smooth_np(curvature.astype(np.float32), radius=2)
    acc = np.zeros_like(speed_s)
    if T > 1:
        acc[1:] = speed_s[1:] - speed_s[:-1]
        acc[0] = acc[1]
    speed_n = speed_s / (float(speed_s.max()) + 1e-6)
    curvature_n = curvature / (float(curvature.max()) + 1e-6)
    acc_n = np.abs(acc) / (float(np.abs(acc).max()) + 1e-6)
    turn_score = (0.70 * curvature_n + 0.20 * speed_n + 0.10 * acc_n).astype(np.float32)
    return {
        "traj": traj,
        "vel": vel,
        "speed": speed_s.astype(np.float32),
        "speed_norm": speed_n.astype(np.float32),
        "heading": heading.astype(np.float32),
        "heading_delta": heading_delta.astype(np.float32),
        "turning": turning.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "curvature_norm": curvature_n.astype(np.float32),
        "acceleration_norm": acc_n.astype(np.float32),
        "turn_score": turn_score,
    }


def detect_turn_events(
    traj: np.ndarray,
    count: int = 5,
    support_lag: Optional[int] = None,
    expressive_lag: Optional[int] = None,
    min_gap: Optional[int] = None,
    gate_sigma: Optional[float] = None,
) -> Tuple[TurnEventReport, Dict[str, np.ndarray]]:
    traj = np.asarray(traj, dtype=np.float32)
    T = int(traj.shape[0])
    support_lag = env_int("EDGE_TURN_SUPPORT_LAG", 8) if support_lag is None else int(support_lag)
    expressive_lag = env_int("EDGE_TURN_EXPR_LAG", 4) if expressive_lag is None else int(expressive_lag)
    min_gap = env_int("EDGE_TURN_MIN_GAP", 18) if min_gap is None else int(min_gap)
    gate_sigma = env_float("EDGE_TURN_GATE_SIGMA", 5.0) if gate_sigma is None else float(gate_sigma)
    scores = trajectory_event_scores(traj)
    centers = _topk_peaks(scores["turn_score"], k=count, min_gap=min_gap, pad=4)
    # Keep an ordered event phrase.  If too few peaks, fill with uniform anchors.
    if len(centers) < count:
        uniform = np.linspace(0.22 * (T - 1), 0.78 * (T - 1), count).round().astype(int).tolist()
        for u in uniform:
            if all(abs(u - c) >= max(4, min_gap // 2) for c in centers):
                centers.append(int(u))
            if len(centers) >= count:
                break
        centers = sorted(centers)[:count]
    support_frames = [int(np.clip(c - support_lag, 0, T - 1)) for c in centers]
    expressive_frames = [int(np.clip(c + expressive_lag, 0, T - 1)) for c in centers]
    speed_peaks = _topk_peaks(scores["speed_norm"], k=count, min_gap=min_gap, pad=4)
    report = TurnEventReport(
        event_centers=[int(c) for c in centers],
        support_frames=support_frames,
        expressive_frames=expressive_frames,
        speed_peaks=[int(c) for c in speed_peaks],
        turn_score_mean=float(np.mean(scores["turn_score"])),
        turn_score_max=float(np.max(scores["turn_score"])),
        speed_mean=float(np.mean(scores["speed"])),
        speed_max=float(np.max(scores["speed"])),
        seq_len=T,
        support_lag=support_lag,
        expressive_lag=expressive_lag,
        min_gap=min_gap,
    )
    scores["turn_gate"] = _gaussian_gate_np(T, centers, sigma=gate_sigma)
    scores["support_gate"] = _gaussian_gate_np(T, support_frames, sigma=gate_sigma)
    scores["expressive_gate"] = _gaussian_gate_np(T, expressive_frames, sigma=gate_sigma)
    scores["speed_gate"] = _gaussian_gate_np(T, speed_peaks, sigma=gate_sigma)
    return report, scores


def event_feature_matrix(
    traj: np.ndarray,
    count: int = 5,
    support_lag: Optional[int] = None,
    expressive_lag: Optional[int] = None,
    min_gap: Optional[int] = None,
    gate_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, TurnEventReport]:
    report, scores = detect_turn_events(
        traj,
        count=count,
        support_lag=support_lag,
        expressive_lag=expressive_lag,
        min_gap=min_gap,
        gate_sigma=gate_sigma,
    )
    traj = scores["traj"]
    x = traj[:, 0]
    z = traj[:, 1]
    # Normalize only by local range to keep this representation scale-stable.
    x_n = (x - x.mean()) / (x.std() + 1e-6)
    z_n = (z - z.mean()) / (z.std() + 1e-6)
    heading = scores["heading"]
    signed_curv = np.sign(scores["heading_delta"]) * scores["curvature_norm"]
    feat = np.stack(
        [
            x_n,
            z_n,
            scores["speed_norm"],
            np.sin(heading),
            np.cos(heading),
            scores["curvature_norm"],
            scores["turn_gate"],
            scores["support_gate"],
            scores["expressive_gate"],
            scores["acceleration_norm"],
            scores["speed_gate"],
            signed_curv,
        ],
        axis=-1,
    ).astype(np.float32)
    return feat, report


def event_feature_matrix_torch(
    traj: "torch.Tensor",
    count: int = 5,
    support_lag: Optional[int] = None,
    expressive_lag: Optional[int] = None,
    min_gap: Optional[int] = None,
    gate_sigma: Optional[float] = None,
) -> "torch.Tensor":
    """Torch-compatible event features.  Peak selection is done per item via numpy.

    The returned tensor has shape [B,T,EVENT_DIM].  This is intentionally not
    differentiable w.r.t. trajectory peak locations; the event extractor is a
    conditioning preprocessor, not a learned operation.
    """
    if torch is None:
        raise RuntimeError("torch is required for event_feature_matrix_torch")
    if traj.ndim == 2:
        traj_b = traj.unsqueeze(0)
    elif traj.ndim == 3:
        traj_b = traj
    else:
        raise ValueError(f"trajectory must be [T,2] or [B,T,2], got {tuple(traj.shape)}")
    device, dtype = traj_b.device, traj_b.dtype
    arr = traj_b.detach().float().cpu().numpy()
    feats = []
    for b in range(arr.shape[0]):
        f, _ = event_feature_matrix(
            arr[b, :, :2],
            count=count,
            support_lag=support_lag,
            expressive_lag=expressive_lag,
            min_gap=min_gap,
            gate_sigma=gate_sigma,
        )
        feats.append(f)
    out = torch.from_numpy(np.stack(feats, axis=0)).to(device=device, dtype=dtype)
    return out


def save_event_report(path: str | Path, report: TurnEventReport, features: Optional[np.ndarray] = None) -> None:
    payload: Dict[str, object] = report.to_dict()
    if features is not None:
        payload["event_dim"] = int(features.shape[-1])
        payload["turn_gate_sum"] = float(features[:, 6].sum())
        payload["support_gate_sum"] = float(features[:, 7].sum())
        payload["expressive_gate_sum"] = float(features[:, 8].sum())
        payload["speed_gate_sum"] = float(features[:, 10].sum())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------------
# Backward-compatible helpers for functional_choreo_metrics.py
# Added for turn-aware internal adapter compatibility.
# ---------------------------------------------------------------------

def parse_points(points):
    """Parse trajectory points.

    Accepts:
      - "x,z;x,z;..."
      - list/tuple/np.ndarray with shape [N,2]
    Returns np.ndarray [N,2] float32.
    """
    import numpy as np

    if points is None:
        raise ValueError("parse_points got None")

    if isinstance(points, str):
        pts = []
        for item in points.strip().split(";"):
            item = item.strip()
            if not item:
                continue
            parts = item.split(",")
            if len(parts) != 2:
                raise ValueError(f"Bad trajectory point: {item!r}")
            pts.append([float(parts[0]), float(parts[1])])
        if len(pts) < 2:
            raise ValueError(f"Need at least 2 trajectory points, got {len(pts)}")
        return np.asarray(pts, dtype=np.float32)

    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Trajectory points must be [N,2], got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"Need at least 2 trajectory points, got {arr.shape[0]}")
    return arr


def interp_traj(points, seq_len=150, *args, **kwargs):
    """Interpolate sparse X/Z trajectory points to [seq_len,2].

    Compatible with older callers:
      interp_traj(points, seq_len=150)
      interp_traj(points, T=150)
      interp_traj(points, n=150)
    """
    import numpy as np

    if "T" in kwargs:
        seq_len = kwargs["T"]
    if "n" in kwargs:
        seq_len = kwargs["n"]
    if "num_frames" in kwargs:
        seq_len = kwargs["num_frames"]

    pts = parse_points(points)
    seq_len = int(seq_len)

    old_t = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)

    x = np.interp(new_t, old_t, pts[:, 0])
    z = np.interp(new_t, old_t, pts[:, 1])
    return np.stack([x, z], axis=-1).astype(np.float32)

# ---------------------------------------------------------------------
# Backward-compatible event API for evaluate_functional_choreo_coupling.py
# ---------------------------------------------------------------------

try:
    TurnEventConfig
except NameError:
    from dataclasses import dataclass

    @dataclass
    class TurnEventConfig:
        seq_len: int = 150
        support_lag: int = 8
        expressive_lag: int = 4
        min_gap: int = 18
        gate_sigma: float = 5.0
        top_k: int = 5


try:
    parse_int_list
except NameError:
    def parse_int_list(x, sep=","):
        """Parse '1,2,3' or list-like input into List[int]."""
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return [int(v) for v in x]
        s = str(x).strip()
        if not s:
            return []
        return [int(v.strip()) for v in s.split(sep) if v.strip()]


try:
    _turn_event_gaussian_gate
except NameError:
    def _turn_event_gaussian_gate(T, centers, sigma=5.0):
        import numpy as np
        T = int(T)
        t = np.arange(T, dtype=np.float32)
        gate = np.zeros((T,), dtype=np.float32)
        sigma = max(float(sigma), 1e-6)
        for c in centers:
            c = float(c)
            gate = np.maximum(gate, np.exp(-0.5 * ((t - c) / sigma) ** 2))
        return gate.astype(np.float32)


try:
    event_feature_matrix
except NameError:
    def event_feature_matrix(
        trajectory=None,
        seq_len=150,
        event_centers=None,
        support_frames=None,
        expressive_frames=None,
        support_lag=8,
        expressive_lag=4,
        min_gap=18,
        gate_sigma=5.0,
        **kwargs,
    ):
        """Return [T,11] turn-aware event feature matrix.

        Columns:
          0 speed
          1 heading_sin
          2 heading_cos
          3 curvature/turn score
          4 curvature_sign
          5 speed_gate
          6 turn_gate
          7 support_gate
          8 expressive_gate
          9 normalized_time
          10 speed_gate duplicate for legacy refiner/evaluator
        """
        import numpy as np

        T = int(seq_len)

        if trajectory is None:
            if "traj" in kwargs:
                trajectory = kwargs["traj"]
            elif "points" in kwargs:
                trajectory = kwargs["points"]
            else:
                trajectory = "0,0;1,0"

        traj = interp_traj(trajectory, seq_len=T)

        vel = np.zeros_like(traj, dtype=np.float32)
        vel[1:] = traj[1:] - traj[:-1]

        speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
        heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)
        heading_sin = np.sin(heading).astype(np.float32)
        heading_cos = np.cos(heading).astype(np.float32)

        d_heading = np.zeros((T,), dtype=np.float32)
        if T > 1:
            raw = heading[1:] - heading[:-1]
            raw = (raw + np.pi) % (2 * np.pi) - np.pi
            d_heading[1:] = raw

        turn_score = np.abs(d_heading).astype(np.float32)
        if turn_score.max() > 1e-8:
            turn_score_norm = turn_score / (turn_score.max() + 1e-8)
        else:
            turn_score_norm = turn_score

        curvature_sign = np.sign(d_heading).astype(np.float32)

        if event_centers is None:
            # Prefer existing detect_turn_events implementation when present.
            try:
                ev = detect_turn_events(
                    traj,
                    seq_len=T,
                    support_lag=support_lag,
                    expressive_lag=expressive_lag,
                    min_gap=min_gap,
                    gate_sigma=gate_sigma,
                )
                if isinstance(ev, dict):
                    event_centers = ev.get("event_centers") or ev.get("turn_peaks")
                    support_frames = support_frames or ev.get("support_frames")
                    expressive_frames = expressive_frames or ev.get("expressive_frames")
                else:
                    event_centers = getattr(ev, "event_centers", None)
                    support_frames = support_frames or getattr(ev, "support_frames", None)
                    expressive_frames = expressive_frames or getattr(ev, "expressive_frames", None)
            except Exception:
                event_centers = None

        if event_centers is None:
            # Fallback: local high-turn frames with min_gap.
            order = np.argsort(-turn_score_norm)
            centers = []
            for idx in order:
                idx = int(idx)
                if idx <= 1 or idx >= T - 1:
                    continue
                if all(abs(idx - c) >= int(min_gap) for c in centers):
                    centers.append(idx)
                if len(centers) >= 5:
                    break
            event_centers = sorted(centers) if centers else [T // 4, T // 2, 3 * T // 4]

        event_centers = [int(x) for x in event_centers]

        if support_frames is None:
            support_frames = [max(0, min(T - 1, int(c) - int(support_lag))) for c in event_centers]
        if expressive_frames is None:
            expressive_frames = [max(0, min(T - 1, int(c) + int(expressive_lag))) for c in event_centers]

        turn_gate = _turn_event_gaussian_gate(T, event_centers, sigma=gate_sigma)
        support_gate = _turn_event_gaussian_gate(T, support_frames, sigma=gate_sigma)
        expressive_gate = _turn_event_gaussian_gate(T, expressive_frames, sigma=gate_sigma)

        # Speed gate from top speed peaks.
        speed_norm = speed / (speed.max() + 1e-8) if speed.max() > 1e-8 else speed
        order = np.argsort(-speed_norm)
        speed_centers = []
        for idx in order:
            idx = int(idx)
            if all(abs(idx - c) >= int(min_gap) for c in speed_centers):
                speed_centers.append(idx)
            if len(speed_centers) >= 5:
                break
        speed_gate = _turn_event_gaussian_gate(T, speed_centers, sigma=gate_sigma)

        norm_time = np.linspace(0.0, 1.0, T, dtype=np.float32)

        feat = np.stack(
            [
                speed_norm.astype(np.float32),
                heading_sin,
                heading_cos,
                turn_score_norm.astype(np.float32),
                curvature_sign,
                speed_gate,
                turn_gate,
                support_gate,
                expressive_gate,
                norm_time,
                speed_gate,
            ],
            axis=-1,
        ).astype(np.float32)

        return feat

# ---------------------------------------------------------------------
# FORCE OVERRIDE: compatible event_feature_matrix API
# This definition intentionally overrides any earlier event_feature_matrix.
# ---------------------------------------------------------------------

def _compat_gaussian_gate(T, centers, sigma=5.0):
    import numpy as np
    T = int(T)
    t = np.arange(T, dtype=np.float32)
    gate = np.zeros((T,), dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    for c in centers or []:
        c = float(c)
        gate = np.maximum(gate, np.exp(-0.5 * ((t - c) / sigma) ** 2))
    return gate.astype(np.float32)


def event_feature_matrix(
    trajectory=None,
    seq_len=150,
    event_centers=None,
    support_frames=None,
    expressive_frames=None,
    support_lag=8,
    expressive_lag=4,
    min_gap=18,
    gate_sigma=5.0,
    *args,
    **kwargs,
):
    """Compatible [T,11] turn-aware event feature matrix.

    Accepts both new and legacy call styles:
      event_feature_matrix(trajectory="x,z;...", seq_len=150)
      event_feature_matrix(traj, seq_len=150)
      event_feature_matrix(points=..., T=150)

    Columns:
      0 speed_norm
      1 heading_sin
      2 heading_cos
      3 turn_score_norm
      4 curvature_sign
      5 speed_gate
      6 turn_gate
      7 support_gate
      8 expressive_gate
      9 normalized_time
      10 speed_gate duplicate for legacy refiner/evaluator
    """
    import numpy as np

    if trajectory is None:
        if len(args) > 0:
            trajectory = args[0]
        elif "traj" in kwargs:
            trajectory = kwargs["traj"]
        elif "points" in kwargs:
            trajectory = kwargs["points"]
        elif "trajectory_points" in kwargs:
            trajectory = kwargs["trajectory_points"]
        else:
            trajectory = "0,0;1,0"

    if "T" in kwargs:
        seq_len = kwargs["T"]
    if "n" in kwargs:
        seq_len = kwargs["n"]
    if "num_frames" in kwargs:
        seq_len = kwargs["num_frames"]

    T = int(seq_len)

    # Use existing interp_traj / parse_points if available.
    traj = interp_traj(trajectory, seq_len=T)

    vel = np.zeros_like(traj, dtype=np.float32)
    vel[1:] = traj[1:] - traj[:-1]

    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    speed_norm = speed / (speed.max() + 1e-8) if speed.max() > 1e-8 else speed

    heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)
    heading_sin = np.sin(heading).astype(np.float32)
    heading_cos = np.cos(heading).astype(np.float32)

    d_heading = np.zeros((T,), dtype=np.float32)
    if T > 1:
        raw = heading[1:] - heading[:-1]
        raw = (raw + np.pi) % (2 * np.pi) - np.pi
        d_heading[1:] = raw

    turn_score = np.abs(d_heading).astype(np.float32)
    turn_score_norm = (
        turn_score / (turn_score.max() + 1e-8)
        if turn_score.max() > 1e-8
        else turn_score
    )
    curvature_sign = np.sign(d_heading).astype(np.float32)

    # Detect event centers if not provided.
    if event_centers is None:
        detected = None
        try:
            detected = detect_turn_events(
                traj,
                seq_len=T,
                support_lag=support_lag,
                expressive_lag=expressive_lag,
                min_gap=min_gap,
                gate_sigma=gate_sigma,
            )
        except TypeError:
            try:
                detected = detect_turn_events(
                    traj,
                    support_lag=support_lag,
                    expressive_lag=expressive_lag,
                    min_gap=min_gap,
                    gate_sigma=gate_sigma,
                )
            except Exception:
                detected = None
        except Exception:
            detected = None

        if isinstance(detected, dict):
            event_centers = detected.get("event_centers") or detected.get("turn_peaks")
            support_frames = support_frames or detected.get("support_frames")
            expressive_frames = expressive_frames or detected.get("expressive_frames")
        elif detected is not None:
            event_centers = getattr(detected, "event_centers", None) or getattr(detected, "turn_peaks", None)
            support_frames = support_frames or getattr(detected, "support_frames", None)
            expressive_frames = expressive_frames or getattr(detected, "expressive_frames", None)

    if event_centers is None:
        # Fallback: choose high turn-score frames with min_gap.
        order = np.argsort(-turn_score_norm)
        centers = []
        for idx in order:
            idx = int(idx)
            if idx <= 1 or idx >= T - 1:
                continue
            if all(abs(idx - c) >= int(min_gap) for c in centers):
                centers.append(idx)
            if len(centers) >= 5:
                break
        event_centers = sorted(centers) if centers else [T // 4, T // 2, 3 * T // 4]

    event_centers = [max(0, min(T - 1, int(c))) for c in event_centers]

    if support_frames is None:
        support_frames = [
            max(0, min(T - 1, int(c) - int(support_lag)))
            for c in event_centers
        ]
    if expressive_frames is None:
        expressive_frames = [
            max(0, min(T - 1, int(c) + int(expressive_lag)))
            for c in event_centers
        ]

    support_frames = [max(0, min(T - 1, int(c))) for c in support_frames]
    expressive_frames = [max(0, min(T - 1, int(c))) for c in expressive_frames]

    turn_gate = _compat_gaussian_gate(T, event_centers, sigma=gate_sigma)
    support_gate = _compat_gaussian_gate(T, support_frames, sigma=gate_sigma)
    expressive_gate = _compat_gaussian_gate(T, expressive_frames, sigma=gate_sigma)

    # Speed peaks.
    order = np.argsort(-speed_norm)
    speed_centers = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - c) >= int(min_gap) for c in speed_centers):
            speed_centers.append(idx)
        if len(speed_centers) >= 5:
            break
    speed_gate = _compat_gaussian_gate(T, speed_centers, sigma=gate_sigma)

    norm_time = np.linspace(0.0, 1.0, T, dtype=np.float32)

    feat = np.stack(
        [
            speed_norm.astype(np.float32),
            heading_sin,
            heading_cos,
            turn_score_norm.astype(np.float32),
            curvature_sign.astype(np.float32),
            speed_gate.astype(np.float32),
            turn_gate.astype(np.float32),
            support_gate.astype(np.float32),
            expressive_gate.astype(np.float32),
            norm_time.astype(np.float32),
            speed_gate.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    return feat

# ---------------------------------------------------------------------
# FORCE OVERRIDE V2:
# Add TurnEventConfig.from_env and support legacy return:
#   ev, names, report = event_feature_matrix(trajectory, TurnEventConfig.from_env(...))
# ---------------------------------------------------------------------

def _turn_event_config_from_env(cls, seq_len=150, count=5, **kwargs):
    import os

    def _int_env(name, default):
        try:
            return int(os.environ.get(name, default))
        except Exception:
            return int(default)

    def _float_env(name, default):
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return float(default)

    cfg = cls(
        seq_len=int(seq_len),
        support_lag=_int_env("EDGE_TURN_SUPPORT_LAG", kwargs.get("support_lag", 8)),
        expressive_lag=_int_env("EDGE_TURN_EXPR_LAG", kwargs.get("expressive_lag", 4)),
        min_gap=_int_env("EDGE_TURN_MIN_GAP", kwargs.get("min_gap", 18)),
        gate_sigma=_float_env("EDGE_TURN_GATE_SIGMA", kwargs.get("gate_sigma", 5.0)),
        top_k=int(count),
    )

    # Some older callers may look for count instead of top_k.
    try:
        setattr(cfg, "count", int(count))
    except Exception:
        pass

    return cfg


try:
    TurnEventConfig.from_env = classmethod(_turn_event_config_from_env)
except NameError:
    from dataclasses import dataclass

    @dataclass
    class TurnEventConfig:
        seq_len: int = 150
        support_lag: int = 8
        expressive_lag: int = 4
        min_gap: int = 18
        gate_sigma: float = 5.0
        top_k: int = 5

    TurnEventConfig.from_env = classmethod(_turn_event_config_from_env)


def _compat_extract_event_config(config_or_seq_len=None, seq_len=150, **kwargs):
    """Return config object or None, and seq_len int."""
    if config_or_seq_len is None:
        return None, int(seq_len)

    # Config object path.
    if hasattr(config_or_seq_len, "seq_len"):
        cfg = config_or_seq_len
        return cfg, int(getattr(cfg, "seq_len", seq_len))

    # Integer seq_len path.
    try:
        return None, int(config_or_seq_len)
    except Exception:
        return None, int(seq_len)


def _compat_get_cfg_attr(cfg, name, default):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _compat_event_feature_matrix_impl(
    trajectory=None,
    seq_len=150,
    event_centers=None,
    support_frames=None,
    expressive_frames=None,
    support_lag=8,
    expressive_lag=4,
    min_gap=18,
    gate_sigma=5.0,
    top_k=5,
):
    import numpy as np

    T = int(seq_len)

    if trajectory is None:
        trajectory = "0,0;1,0"

    traj = interp_traj(trajectory, seq_len=T)

    vel = np.zeros_like(traj, dtype=np.float32)
    vel[1:] = traj[1:] - traj[:-1]

    speed = np.linalg.norm(vel, axis=-1).astype(np.float32)
    speed_norm = speed / (speed.max() + 1e-8) if speed.max() > 1e-8 else speed

    heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-8).astype(np.float32)
    heading_sin = np.sin(heading).astype(np.float32)
    heading_cos = np.cos(heading).astype(np.float32)

    d_heading = np.zeros((T,), dtype=np.float32)
    if T > 1:
        raw = heading[1:] - heading[:-1]
        raw = (raw + np.pi) % (2 * np.pi) - np.pi
        d_heading[1:] = raw

    turn_score = np.abs(d_heading).astype(np.float32)
    turn_score_norm = (
        turn_score / (turn_score.max() + 1e-8)
        if turn_score.max() > 1e-8
        else turn_score
    )
    curvature_sign = np.sign(d_heading).astype(np.float32)

    detected_report = {}

    if event_centers is None:
        try:
            detected = detect_turn_events(
                traj,
                seq_len=T,
                support_lag=support_lag,
                expressive_lag=expressive_lag,
                min_gap=min_gap,
                gate_sigma=gate_sigma,
                count=top_k,
            )
        except TypeError:
            try:
                detected = detect_turn_events(
                    traj,
                    support_lag=support_lag,
                    expressive_lag=expressive_lag,
                    min_gap=min_gap,
                    gate_sigma=gate_sigma,
                )
            except Exception:
                detected = None
        except Exception:
            detected = None

        if isinstance(detected, dict):
            detected_report = dict(detected)
            event_centers = detected.get("event_centers") or detected.get("turn_peaks")
            support_frames = support_frames or detected.get("support_frames")
            expressive_frames = expressive_frames or detected.get("expressive_frames")
        elif detected is not None:
            event_centers = getattr(detected, "event_centers", None) or getattr(detected, "turn_peaks", None)
            support_frames = support_frames or getattr(detected, "support_frames", None)
            expressive_frames = expressive_frames or getattr(detected, "expressive_frames", None)
            try:
                detected_report = dict(detected.__dict__)
            except Exception:
                detected_report = {}

    if event_centers is None:
        order = np.argsort(-turn_score_norm)
        centers = []
        for idx in order:
            idx = int(idx)
            if idx <= 1 or idx >= T - 1:
                continue
            if all(abs(idx - c) >= int(min_gap) for c in centers):
                centers.append(idx)
            if len(centers) >= int(top_k):
                break
        event_centers = sorted(centers) if centers else [T // 4, T // 2, 3 * T // 4]

    event_centers = [max(0, min(T - 1, int(c))) for c in event_centers]

    if support_frames is None:
        support_frames = [
            max(0, min(T - 1, int(c) - int(support_lag)))
            for c in event_centers
        ]
    if expressive_frames is None:
        expressive_frames = [
            max(0, min(T - 1, int(c) + int(expressive_lag)))
            for c in event_centers
        ]

    support_frames = [max(0, min(T - 1, int(c))) for c in support_frames]
    expressive_frames = [max(0, min(T - 1, int(c))) for c in expressive_frames]

    turn_gate = _compat_gaussian_gate(T, event_centers, sigma=gate_sigma)
    support_gate = _compat_gaussian_gate(T, support_frames, sigma=gate_sigma)
    expressive_gate = _compat_gaussian_gate(T, expressive_frames, sigma=gate_sigma)

    order = np.argsort(-speed_norm)
    speed_centers = []
    for idx in order:
        idx = int(idx)
        if all(abs(idx - c) >= int(min_gap) for c in speed_centers):
            speed_centers.append(idx)
        if len(speed_centers) >= int(top_k):
            break
    speed_gate = _compat_gaussian_gate(T, speed_centers, sigma=gate_sigma)

    norm_time = np.linspace(0.0, 1.0, T, dtype=np.float32)

    names = [
        "speed",
        "heading_sin",
        "heading_cos",
        "turn_score",
        "curvature_sign",
        "speed_gate",
        "turn_gate",
        "support_gate",
        "expressive_gate",
        "normalized_time",
        "speed_gate_legacy",
    ]

    feat = np.stack(
        [
            speed_norm.astype(np.float32),
            heading_sin.astype(np.float32),
            heading_cos.astype(np.float32),
            turn_score_norm.astype(np.float32),
            curvature_sign.astype(np.float32),
            speed_gate.astype(np.float32),
            turn_gate.astype(np.float32),
            support_gate.astype(np.float32),
            expressive_gate.astype(np.float32),
            norm_time.astype(np.float32),
            speed_gate.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)

    report = dict(detected_report)
    report.update(
        {
            "event_centers": event_centers,
            "support_frames": support_frames,
            "expressive_frames": expressive_frames,
            "speed_peaks": speed_centers,
            "seq_len": T,
            "support_lag": int(support_lag),
            "expressive_lag": int(expressive_lag),
            "min_gap": int(min_gap),
            "gate_sigma": float(gate_sigma),
            "turn_score_mean": float(turn_score_norm.mean()),
            "turn_score_max": float(turn_score_norm.max()),
            "speed_mean": float(speed_norm.mean()),
            "speed_max": float(speed_norm.max()),
        }
    )

    return feat, names, report


def event_feature_matrix(
    trajectory=None,
    config_or_seq_len=None,
    *args,
    seq_len=150,
    event_centers=None,
    support_frames=None,
    expressive_frames=None,
    support_lag=8,
    expressive_lag=4,
    min_gap=18,
    gate_sigma=5.0,
    top_k=5,
    **kwargs,
):
    """Compatible event feature API.

    New/simple call:
      feat = event_feature_matrix(trajectory="...", seq_len=150)

    Legacy evaluator call:
      ev, names, report = event_feature_matrix(trajectory, TurnEventConfig.from_env(...))
    """
    if trajectory is None:
        if len(args) > 0:
            trajectory = args[0]
            args = args[1:]
        elif "traj" in kwargs:
            trajectory = kwargs["traj"]
        elif "points" in kwargs:
            trajectory = kwargs["points"]
        elif "trajectory_points" in kwargs:
            trajectory = kwargs["trajectory_points"]
        else:
            trajectory = "0,0;1,0"

    if "T" in kwargs:
        seq_len = kwargs["T"]
    if "n" in kwargs:
        seq_len = kwargs["n"]
    if "num_frames" in kwargs:
        seq_len = kwargs["num_frames"]

    cfg, seq_len2 = _compat_extract_event_config(config_or_seq_len, seq_len=seq_len)
    seq_len = seq_len2

    support_lag = _compat_get_cfg_attr(cfg, "support_lag", support_lag)
    expressive_lag = _compat_get_cfg_attr(cfg, "expressive_lag", expressive_lag)
    min_gap = _compat_get_cfg_attr(cfg, "min_gap", min_gap)
    gate_sigma = _compat_get_cfg_attr(cfg, "gate_sigma", gate_sigma)
    top_k = _compat_get_cfg_attr(cfg, "top_k", _compat_get_cfg_attr(cfg, "count", top_k))

    feat, names, report = _compat_event_feature_matrix_impl(
        trajectory=trajectory,
        seq_len=seq_len,
        event_centers=event_centers,
        support_frames=support_frames,
        expressive_frames=expressive_frames,
        support_lag=support_lag,
        expressive_lag=expressive_lag,
        min_gap=min_gap,
        gate_sigma=gate_sigma,
        top_k=top_k,
    )

    # If caller passed a config object, old API expects triple.
    if cfg is not None or bool(kwargs.get("return_report", False)):
        return feat, names, report

    # New/simple API expects just feature matrix.
    return feat

# ---------------------------------------------------------------------
# FORCE OVERRIDE: compatible torch event feature API
# Fixes:
#   ValueError: too many values to unpack (expected 2)
# because event_feature_matrix may return feat or (feat,names,report).
# ---------------------------------------------------------------------

def event_feature_matrix_torch(
    trajectory,
    config_or_seq_len=None,
    *args,
    seq_len=None,
    **kwargs,
):
    """Torch wrapper for event_feature_matrix.

    Accepts:
      trajectory: [T,2] or [B,T,2] torch.Tensor / np.ndarray / list

    Returns:
      [T,11] if input is [T,2]
      [B,T,11] if input is [B,T,2]

    Compatible with event_feature_matrix returning:
      feat
      (feat, report)
      (feat, names, report)
    """
    import numpy as np
    import torch

    # Accept aliases used by older patches.
    if config_or_seq_len is None:
        if "config" in kwargs:
            config_or_seq_len = kwargs.pop("config")
        elif "cfg" in kwargs:
            config_or_seq_len = kwargs.pop("cfg")

    is_torch = torch.is_tensor(trajectory)

    if is_torch:
        device = trajectory.device
        dtype = trajectory.dtype
        x = trajectory
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        x = torch.as_tensor(trajectory, dtype=torch.float32)

    original_ndim = x.ndim

    if x.ndim == 2:
        x_b = x.unsqueeze(0)
    elif x.ndim == 3:
        x_b = x
    else:
        raise ValueError(f"trajectory must be [T,D] or [B,T,D], got {tuple(x.shape)}")

    # Keep only X/Z if caller passed more than 2 dims.
    if x_b.shape[-1] > 2:
        x_b = x_b[..., :2]

    B, T = int(x_b.shape[0]), int(x_b.shape[1])

    if seq_len is None:
        if hasattr(config_or_seq_len, "seq_len"):
            seq_len = int(getattr(config_or_seq_len, "seq_len"))
        elif isinstance(config_or_seq_len, int):
            seq_len = int(config_or_seq_len)
        else:
            seq_len = T

    feats = []

    for b in range(B):
        traj_np = x_b[b].detach().float().cpu().numpy().astype(np.float32)

        if hasattr(config_or_seq_len, "seq_len"):
            res = event_feature_matrix(
                traj_np,
                config_or_seq_len,
                *args,
                **kwargs,
            )
        elif isinstance(config_or_seq_len, int):
            res = event_feature_matrix(
                trajectory=traj_np,
                seq_len=int(config_or_seq_len),
                *args,
                **kwargs,
            )
        else:
            res = event_feature_matrix(
                trajectory=traj_np,
                seq_len=int(seq_len),
                *args,
                **kwargs,
            )

        if isinstance(res, tuple):
            feat = res[0]
        else:
            feat = res

        feat = np.asarray(feat, dtype=np.float32)

        if feat.ndim != 2:
            raise ValueError(f"event feature must be [T,D], got {feat.shape}")

        feats.append(feat)

    out_np = np.stack(feats, axis=0).astype(np.float32)
    out = torch.as_tensor(out_np, device=device, dtype=dtype)

    if original_ndim == 2:
        return out[0]

    return out
