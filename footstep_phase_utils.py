"""Footstep / gait-phase utilities for EDGE 151-D motion.

Representation contract:
  [0:4] foot contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotation
  Ground-plane trajectory is always root X/Z => feature dims [4, 6].

This file is intentionally dependency-light so it can be used from dataset,
planner, inference and post-processing scripts.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

REPR_DIM = 151
CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROOT_XZ_IDX = [ROOT_X_IDX, ROOT_Z_IDX]
ROT_START = 7
ROT_DIM = 6
ROT_SLICE = slice(7, 151)
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def to_numpy(x) -> np.ndarray:
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def maybe_torch(x_np: np.ndarray, like):
    if torch is not None and torch.is_tensor(like):
        return torch.from_numpy(x_np).to(device=like.device, dtype=like.dtype)
    return x_np


def as_t151(motion) -> np.ndarray:
    arr = to_numpy(motion).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == REPR_DIM:
        return arr
    if arr.shape[0] == REPR_DIM:
        return arr.T.astype(np.float32)
    raise ValueError(f"Expected one dimension to be 151, got {arr.shape}")


def joint6d_indices(joints) -> np.ndarray:
    idx = []
    for j in joints:
        j = int(j)
        start = ROT_START + ROT_DIM * j
        idx.extend(range(start, start + ROT_DIM))
    return np.asarray([i for i in idx if 0 <= i < REPR_DIM], dtype=np.int64)


LOWER_ROT_INDEX = joint6d_indices(LOWER_JOINTS)
TORSO_ROT_INDEX = joint6d_indices(TORSO_JOINTS)
UPPER_ROT_INDEX = joint6d_indices(UPPER_JOINTS)


def robust_norm(values: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values, 0.0, 1.0
    if lo is None or hi is None:
        lo, hi = np.percentile(values, [10, 90])
    if float(hi - lo) <= 1e-8:
        lo, hi = float(values.min()), float(values.max() + 1e-6)
    norm = np.clip((values - float(lo)) / max(float(hi - lo), 1e-8), 0.0, 1.0).astype(np.float32)
    return norm, float(lo), float(hi)


def unit_basic_stats(unit) -> Dict[str, float]:
    unit = as_t151(unit)
    if len(unit) <= 1:
        return dict(
            motion_energy=0.0,
            root_speed=0.0,
            upper_activity=0.0,
            lower_activity=0.0,
            spatial_range=0.0,
            turning=0.0,
            contact_stability=1.0,
            contact_switch=0.0,
            alternating_foot_phase=0.0,
            root_lower_sync=0.0,
        )

    diff = unit[1:] - unit[:-1]
    root = unit[:, ROOT_XZ_IDX]
    root_vel = root[1:] - root[:-1]
    root_speed_per = np.linalg.norm(root_vel, axis=-1) if len(root_vel) else np.zeros((0,), dtype=np.float32)
    root_speed = float(root_speed_per.mean()) if root_speed_per.size else 0.0

    contacts = (unit[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_switch = float(np.abs(contacts[1:] - contacts[:-1]).mean()) if len(contacts) > 1 else 0.0
    contact_stability = float(np.clip(1.0 - contact_switch, 0.0, 1.0))

    left = contacts[:, 0:2].max(axis=1)
    right = contacts[:, 2:4].max(axis=1)
    # Alternating support is higher when only one side dominates at a time.
    alternating = float(np.mean(np.abs(left - right))) if len(left) else 0.0

    lower_diff = diff[:, LOWER_ROT_INDEX] if LOWER_ROT_INDEX.size else diff[:, :0]
    upper_diff = diff[:, UPPER_ROT_INDEX] if UPPER_ROT_INDEX.size else diff[:, :0]
    lower_activity = float(np.sqrt(np.mean(lower_diff ** 2))) if lower_diff.size else 0.0
    upper_activity = float(np.sqrt(np.mean(upper_diff ** 2))) if upper_diff.size else 0.0
    motion_energy = float(np.sqrt(np.mean(diff[:, ROT_SLICE] ** 2)))

    if len(root_vel) >= 2:
        v1, v2 = root_vel[:-1], root_vel[1:]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
        turning = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    else:
        turning = 0.0

    # Encourage clips where lower-body motion rises when root speed rises.
    if root_speed_per.size and lower_diff.size:
        lower_frame = np.sqrt(np.mean(lower_diff ** 2, axis=1))
        if float(root_speed_per.std()) > 1e-8 and float(lower_frame.std()) > 1e-8:
            root_lower_sync = float(np.corrcoef(root_speed_per, lower_frame)[0, 1])
            root_lower_sync = float(np.clip((root_lower_sync + 1.0) * 0.5, 0.0, 1.0))
        else:
            root_lower_sync = float(lower_activity > 1e-5 and root_speed > 1e-5)
    else:
        root_lower_sync = 0.0

    return dict(
        motion_energy=motion_energy,
        root_speed=root_speed,
        upper_activity=upper_activity,
        lower_activity=lower_activity,
        spatial_range=float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
        turning=float(max(0.0, turning)),
        contact_stability=contact_stability,
        contact_switch=float(np.clip(contact_switch, 0.0, 1.0)),
        alternating_foot_phase=float(np.clip(alternating, 0.0, 1.0)),
        root_lower_sync=float(np.clip(root_lower_sync, 0.0, 1.0)),
    )


def add_dual_scores(stats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Add expressiveness / locomotion / footstep / mobile scores in-place."""
    n = 0
    for v in stats.values():
        v = np.asarray(v)
        if v.ndim >= 1:
            n = len(v)
            break
    if n == 0:
        return stats

    def field(name: str, default: float = 0.0) -> np.ndarray:
        if name in stats:
            return np.nan_to_num(np.asarray(stats[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)[:n]
        return np.full((n,), float(default), dtype=np.float32)

    # Normalize missing raw fields.
    for raw in ["motion_energy", "root_speed", "upper_activity", "lower_activity", "spatial_range", "turning", "contact_switch", "root_lower_sync"]:
        norm = raw + "_norm"
        if norm not in stats:
            stats[norm] = robust_norm(field(raw))[0]

    contact_stability = field("contact_stability", 1.0)
    expr = field("expressiveness_score", np.nan)
    if not np.isfinite(expr).all() or np.allclose(expr, 0.0):
        expr = np.clip(
            0.30 * field("motion_energy_norm")
            + 0.30 * field("upper_activity_norm")
            + 0.20 * field("spatial_range_norm")
            + 0.15 * field("turning_norm")
            + 0.05 * field("root_speed_norm"),
            0.0,
            1.0,
        ).astype(np.float32)
        expr = expr * np.clip(0.50 + 0.50 * contact_stability, 0.0, 1.0)
    stats["expressiveness_score"] = expr.astype(np.float32)

    locomotion = np.clip(
        0.45 * field("root_speed_norm")
        + 0.35 * field("lower_activity_norm")
        + 0.15 * field("spatial_range_norm")
        + 0.05 * field("turning_norm"),
        0.0,
        1.0,
    ).astype(np.float32)

    footstep = np.clip(
        0.35 * field("contact_switch_norm")
        + 0.30 * field("alternating_foot_phase")
        + 0.20 * field("root_lower_sync_norm")
        + 0.15 * contact_stability,
        0.0,
        1.0,
    ).astype(np.float32)

    mobile = np.clip(0.60 * locomotion + 0.40 * footstep, 0.0, 1.0).astype(np.float32)
    stats["locomotion_score"] = locomotion
    stats["footstep_score"] = footstep
    stats["mobile_score"] = mobile
    return stats


def _unnormalize_if_needed(motion_np: np.ndarray, normalizer=None) -> np.ndarray:
    if normalizer is None:
        return motion_np.astype(np.float32)
    if torch is None:
        try:
            return normalizer.unnormalize(motion_np).astype(np.float32)
        except Exception:
            return motion_np.astype(np.float32)
    mt = torch.as_tensor(motion_np, dtype=torch.float32)
    try:
        if mt.ndim == 2:
            out = normalizer.unnormalize(mt[None])[0]
        else:
            out = normalizer.unnormalize(mt)
        return to_numpy(out).astype(np.float32)
    except Exception:
        return motion_np.astype(np.float32)


def gait_phase_from_motion(motion, normalizer=None, speed_threshold: float = 0.01, stride_length: float = 0.35):
    """Build [T,6] gait phase: sin, cos, left_prior, right_prior, move_gate, speed_norm.

    Training path: contacts come from motion channels [0:4] after optional unnormalization.
    The phase angle is derived from cumulative root X/Z distance so it remains usable
    even when contact channels are noisy.
    """
    original = motion
    motion_np = as_t151(motion)
    physical = _unnormalize_if_needed(motion_np, normalizer=normalizer)
    T = physical.shape[0]
    root = physical[:, ROOT_XZ_IDX].astype(np.float32)
    vel = np.zeros((T,), dtype=np.float32)
    if T > 1:
        vel[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1)
        vel[0] = vel[1]
    speed_norm, _, _ = robust_norm(vel)
    move_gate = (vel > float(speed_threshold)).astype(np.float32)

    contacts = np.clip(physical[:, CONTACT_SLICE], 0.0, 1.0)
    left_prior = contacts[:, 0:2].max(axis=1).astype(np.float32)
    right_prior = contacts[:, 2:4].max(axis=1).astype(np.float32)

    dist = np.zeros((T,), dtype=np.float32)
    if T > 1:
        dist[1:] = np.cumsum(np.linalg.norm(root[1:] - root[:-1], axis=-1))
    stride = max(float(stride_length), 1e-4)
    angle = 2.0 * math.pi * dist / stride
    phase_sin = np.sin(angle).astype(np.float32)
    phase_cos = np.cos(angle).astype(np.float32)

    # In stationary clips, suppress phase oscillation so upper-body dance is not forced to step.
    phase_sin *= np.maximum(move_gate, 0.05)
    phase_cos = phase_cos * np.maximum(move_gate, 0.05) + (1.0 - np.maximum(move_gate, 0.05))

    out = np.stack([phase_sin, phase_cos, left_prior, right_prior, move_gate, speed_norm], axis=-1).astype(np.float32)
    return maybe_torch(out, original)


def gait_phase_from_trajectory(trajectory, speed_threshold: float = 0.01, stride_length: float = 0.35):
    """Inference path. trajectory may be [T,2] or [B,T,2] in normalized or physical X/Z."""
    original = trajectory
    traj = to_numpy(trajectory).astype(np.float32)
    batched = traj.ndim == 3
    if not batched:
        traj = traj[None]
    outs = []
    for root in traj[..., :2]:
        T = root.shape[0]
        vel = np.zeros((T,), dtype=np.float32)
        if T > 1:
            vel[1:] = np.linalg.norm(root[1:] - root[:-1], axis=-1)
            vel[0] = vel[1]
        speed_norm, _, _ = robust_norm(vel)
        move_gate = (vel > float(speed_threshold)).astype(np.float32)
        dist = np.zeros((T,), dtype=np.float32)
        if T > 1:
            dist[1:] = np.cumsum(np.linalg.norm(root[1:] - root[:-1], axis=-1))
        stride = max(float(stride_length), 1e-4)
        angle = 2.0 * math.pi * dist / stride
        phase_sin = np.sin(angle).astype(np.float32) * np.maximum(move_gate, 0.05)
        phase_cos = np.cos(angle).astype(np.float32) * np.maximum(move_gate, 0.05) + (1.0 - np.maximum(move_gate, 0.05))
        left_prior = ((np.sin(angle) >= 0.0).astype(np.float32) * move_gate)
        right_prior = ((np.sin(angle) < 0.0).astype(np.float32) * move_gate)
        outs.append(np.stack([phase_sin, phase_cos, left_prior, right_prior, move_gate, speed_norm], axis=-1))
    out = np.stack(outs, axis=0).astype(np.float32)
    if not batched:
        out = out[0]
    return maybe_torch(out, original)
