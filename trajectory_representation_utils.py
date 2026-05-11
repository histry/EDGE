"""Advanced trajectory representation utilities for EDGE.

The EDGE motion representation is [T,151] with root X/Z in dims [4,6].
For conditioning we keep cond['trajectory'] as [B,T,2] normalized X/Z for
checkpoint compatibility, then optionally add high-frequency / physical
features through runtime patches.

Env flags used by patches:
  EDGE_TRAJ_FOURIER_FEATURES=1
  EDGE_TRAJ_FOURIER_BANDS=6
  EDGE_TRAJ_PHYSICS_FEATURES=1
  EDGE_TRAJ_SPARSE_WAYPOINT=1
  EDGE_TRAJ_WAYPOINT_FRAMES=0,50,100,149
  EDGE_TRAJ_WAYPOINT_KEEP_PROB=0.35
  EDGE_TRAJ_BEV_COND=1
"""
from __future__ import annotations

import math
import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore

_TRUE = {"1", "true", "yes", "y", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def parse_frames(text: str, num_frames: int) -> Sequence[int]:
    if not text:
        return [0, max(0, num_frames // 3), max(0, 2 * num_frames // 3), max(0, num_frames - 1)]
    out = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        v = float(item)
        if 0.0 < v < 1.0:
            v = v * max(0, num_frames - 1)
        out.append(int(round(max(0, min(num_frames - 1, v)))))
    return sorted(set(out))


def _is_tensor(x) -> bool:
    return torch is not None and torch.is_tensor(x)


def _safe_norm_torch(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)


def _safe_norm_np(x, axis=-1, eps=1e-8):
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


def _normalize_01_torch(x, dim=1):
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(1e-8)


def _normalize_01_np(x, axis=0):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = np.min(x, axis=axis, keepdims=True)
    hi = np.max(x, axis=axis, keepdims=True)
    return ((x - lo) / np.maximum(hi - lo, 1e-8)).astype(np.float32)


def trajectory_physics_features(trajectory):
    """Return [B,T,4] = speed_norm, heading_sin, heading_cos, curvature_norm.

    Input can be [T,2] or [B,T,2]. Works with numpy or torch.
    """
    if _is_tensor(trajectory):
        traj = trajectory[..., :2].float()
        squeeze = traj.ndim == 2
        if squeeze:
            traj = traj.unsqueeze(0)
        B, T, _ = traj.shape
        vel = torch.zeros_like(traj)
        if T > 1:
            vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
            vel[:, 0] = vel[:, 1]
        speed = _safe_norm_torch(vel, dim=-1)
        speed_norm = _normalize_01_torch(speed, dim=1)
        heading = vel / speed.unsqueeze(-1).clamp_min(1e-8)
        heading_sin = heading[..., 1]
        heading_cos = heading[..., 0]
        curvature = torch.zeros((B, T), device=traj.device, dtype=traj.dtype)
        if T > 2:
            v1 = traj[:, 1:-1] - traj[:, :-2]
            v2 = traj[:, 2:] - traj[:, 1:-1]
            n1 = _safe_norm_torch(v1, dim=-1)
            n2 = _safe_norm_torch(v2, dim=-1)
            cos = torch.sum(v1 * v2, dim=-1) / (n1 * n2).clamp_min(1e-8)
            curvature[:, 1:-1] = 1.0 - cos.clamp(-1.0, 1.0)
        curvature_norm = _normalize_01_torch(curvature, dim=1)
        out = torch.stack([speed_norm, heading_sin, heading_cos, curvature_norm], dim=-1)
        return out[0] if squeeze else out

    traj = np.asarray(trajectory, dtype=np.float32)[..., :2]
    squeeze = traj.ndim == 2
    if squeeze:
        traj = traj[None]
    B, T, _ = traj.shape
    vel = np.zeros_like(traj, dtype=np.float32)
    if T > 1:
        vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
        vel[:, 0] = vel[:, 1]
    speed = _safe_norm_np(vel, axis=-1)
    speed_norm = _normalize_01_np(speed, axis=1)
    heading = vel / np.maximum(speed[..., None], 1e-8)
    heading_sin = heading[..., 1]
    heading_cos = heading[..., 0]
    curvature = np.zeros((B, T), dtype=np.float32)
    if T > 2:
        v1 = traj[:, 1:-1] - traj[:, :-2]
        v2 = traj[:, 2:] - traj[:, 1:-1]
        n1 = _safe_norm_np(v1, axis=-1)
        n2 = _safe_norm_np(v2, axis=-1)
        cos = np.sum(v1 * v2, axis=-1) / np.maximum(n1 * n2, 1e-8)
        curvature[:, 1:-1] = 1.0 - np.clip(cos, -1.0, 1.0)
    curvature_norm = _normalize_01_np(curvature, axis=1)
    out = np.stack([speed_norm, heading_sin, heading_cos, curvature_norm], axis=-1).astype(np.float32)
    return out[0] if squeeze else out


def trajectory_fourier_features(trajectory, bands: int = 6, include_input: bool = False):
    """Fourier encode X/Z coordinates and ΔX/ΔZ velocities."""
    bands = max(0, int(bands))
    if bands <= 0:
        if _is_tensor(trajectory):
            return trajectory.new_zeros((*trajectory.shape[:-1], 0))
        return np.zeros((*np.asarray(trajectory).shape[:-1], 0), dtype=np.float32)

    if _is_tensor(trajectory):
        traj = trajectory[..., :2].float()
        squeeze = traj.ndim == 2
        if squeeze:
            traj = traj.unsqueeze(0)
        vel = torch.zeros_like(traj)
        if traj.shape[1] > 1:
            vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
            vel[:, 0] = vel[:, 1]
        base = torch.cat([traj, vel], dim=-1)
        freqs = (2.0 ** torch.arange(bands, device=traj.device, dtype=traj.dtype)) * math.pi
        feats = []
        if include_input:
            feats.append(base)
        for f in freqs:
            feats.append(torch.sin(base * f))
            feats.append(torch.cos(base * f))
        out = torch.cat(feats, dim=-1) if feats else base.new_zeros((*base.shape[:-1], 0))
        return out[0] if squeeze else out

    traj = np.asarray(trajectory, dtype=np.float32)[..., :2]
    squeeze = traj.ndim == 2
    if squeeze:
        traj = traj[None]
    vel = np.zeros_like(traj, dtype=np.float32)
    if traj.shape[1] > 1:
        vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
        vel[:, 0] = vel[:, 1]
    base = np.concatenate([traj, vel], axis=-1)
    feats = []
    if include_input:
        feats.append(base)
    for i in range(bands):
        f = (2.0 ** i) * math.pi
        feats.append(np.sin(base * f))
        feats.append(np.cos(base * f))
    out = np.concatenate(feats, axis=-1).astype(np.float32) if feats else np.zeros((*base.shape[:-1], 0), dtype=np.float32)
    return out[0] if squeeze else out


def build_advanced_trajectory_features(
    trajectory,
    use_fourier: Optional[bool] = None,
    use_physics: Optional[bool] = None,
    bands: Optional[int] = None,
):
    """Build optional advanced trajectory features from [B,T,2].

    The base EDGE branch still receives X/Z + ΔX/ΔZ. These are extra residual
    features projected by trajectory_enhancement_patch.
    """
    if use_fourier is None:
        use_fourier = env_bool("EDGE_TRAJ_FOURIER_FEATURES", False)
    if use_physics is None:
        use_physics = env_bool("EDGE_TRAJ_PHYSICS_FEATURES", False)
    if bands is None:
        bands = env_int("EDGE_TRAJ_FOURIER_BANDS", 6)

    feats = []
    if use_physics:
        feats.append(trajectory_physics_features(trajectory))
    if use_fourier:
        feats.append(trajectory_fourier_features(trajectory, bands=bands, include_input=False))
    if not feats:
        if _is_tensor(trajectory):
            return trajectory.new_zeros((*trajectory.shape[:-1], 0))
        return np.zeros((*np.asarray(trajectory).shape[:-1], 0), dtype=np.float32)
    if _is_tensor(feats[0]):
        return torch.cat(feats, dim=-1)
    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_sparse_waypoint_mask(num_frames: int, frames: Optional[Iterable[int]] = None, keep_prob: Optional[float] = None):
    if frames is None:
        frames = parse_frames(os.environ.get("EDGE_TRAJ_WAYPOINT_FRAMES", ""), num_frames)
    if keep_prob is None:
        keep_prob = env_float("EDGE_TRAJ_WAYPOINT_KEEP_PROB", 0.35)
    mask = np.zeros((num_frames, 1), dtype=np.float32)
    for f in frames:
        if 0 <= int(f) < num_frames:
            mask[int(f), 0] = 1.0
    # Keep a small amount of dense signal for stability unless strict sparse is requested.
    if not env_bool("EDGE_TRAJ_STRICT_SPARSE", False):
        mask = np.maximum(mask, float(keep_prob)).astype(np.float32)
    return mask


def apply_sparse_waypoint_mask(trajectory, mask=None, fill: str = "linear"):
    """Return trajectory with non-waypoint frames masked or interpolated.

    fill='zero': non-waypoints become 0.
    fill='linear': interpolate between hard waypoints; if no hard waypoints, leave input.
    """
    if _is_tensor(trajectory):
        traj = trajectory
        if mask is None:
            m_np = build_sparse_waypoint_mask(traj.shape[-2])
            mask = torch.from_numpy(m_np).to(device=traj.device, dtype=traj.dtype)
        mask = mask.to(device=traj.device, dtype=traj.dtype)
        while mask.ndim < traj.ndim:
            mask = mask.unsqueeze(0)
        if fill == "zero":
            return traj * mask, mask
        # torch linear interpolation for sparse hard points is cumbersome; leave dense and provide mask.
        return traj, mask

    traj = np.asarray(trajectory, dtype=np.float32)
    squeeze = traj.ndim == 2
    if squeeze:
        traj = traj[None]
    B, T, C = traj.shape
    if mask is None:
        mask = build_sparse_waypoint_mask(T)
    mask = np.asarray(mask, dtype=np.float32).reshape(T, 1)
    if fill == "zero":
        out = traj * mask[None]
    else:
        out = traj.copy()
        hard = np.where(mask[:, 0] >= 0.99)[0]
        if len(hard) >= 2:
            x = np.arange(T)
            for b in range(B):
                for c in range(C):
                    out[b, :, c] = np.interp(x, hard, traj[b, hard, c])
    return (out[0] if squeeze else out).astype(np.float32), mask.astype(np.float32)


def trajectory_to_bev_heatmap(
    trajectory,
    size: int = 32,
    sigma: float = 1.5,
    temporal_decay: float = 0.96,
    margin: float = 0.25,
):
    """Render [T,2] or [B,T,2] trajectory to a BEV Gaussian heatmap [B,1,H,W].

    This is an optional Stage-4 feature. It gives a coarse stage-map summary,
    not per-frame control.
    """
    traj = np.asarray(trajectory, dtype=np.float32)[..., :2]
    squeeze = traj.ndim == 2
    if squeeze:
        traj = traj[None]
    B, T, _ = traj.shape
    H = W = int(size)
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = np.zeros((B, 1, H, W), dtype=np.float32)
    for b in range(B):
        xy = traj[b]
        lo = xy.min(axis=0) - margin
        hi = xy.max(axis=0) + margin
        span = np.maximum(hi - lo, 1e-6)
        pix = (xy - lo) / span
        px = pix[:, 0] * (W - 1)
        py = pix[:, 1] * (H - 1)
        for t in range(T):
            amp = float(temporal_decay ** (T - 1 - t))
            g = np.exp(-((grid_x - px[t]) ** 2 + (grid_y - py[t]) ** 2) / (2.0 * sigma * sigma))
            out[b, 0] += amp * g.astype(np.float32)
        m = out[b, 0].max()
        if m > 1e-8:
            out[b, 0] /= m
    return out[0] if squeeze else out


def dynamic_traj_cfg_weights(trajectory, base=1.5, speed_w=2.0, curvature_w=1.0, min_w=1.0, max_w=5.0):
    phys = trajectory_physics_features(trajectory)
    if _is_tensor(phys):
        speed = phys[..., 0]
        curv = phys[..., 3]
        w = float(base) + float(speed_w) * speed + float(curvature_w) * curv
        return w.clamp(float(min_w), float(max_w)).unsqueeze(-1)
    speed = phys[..., 0]
    curv = phys[..., 3]
    w = float(base) + float(speed_w) * speed + float(curvature_w) * curv
    return np.clip(w, float(min_w), float(max_w)).astype(np.float32)[..., None]
