#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-plausible post-optimization for V34 151D motion.

The previous "smooth everything" style of post-processing can hide jitter but
also destroys landing impulse and makes the root look buoyant.  This module is
structured as a conservative physical gateway:

1. contact-aware root X/Z lock for support-foot sliding;
2. contact-segmented root-Y gravity arc and landing shock absorption;
3. lightweight collision-aware IK on upper-body rotations only.

The original *_v26.npy is never overwritten unless the caller explicitly points
--out to the same path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - CPU-only environments still get root fixes.
    torch = None

try:
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton
except Exception:  # pragma: no cover - fallback FK below remains available.
    ax_from_6v = None
    SMPLSkeleton = None


PARENTS = np.array([
    -1, 0, 0, 0,
     1, 2, 3,
     4, 5, 6,
     7, 8, 9,
     9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21,
], dtype=np.int64)

OFFSETS = np.array([
    [ 0.00,  0.00,  0.00],
    [-0.10, -0.10,  0.00],
    [ 0.10, -0.10,  0.00],
    [ 0.00,  0.13,  0.00],
    [ 0.00, -0.42,  0.00],
    [ 0.00, -0.42,  0.00],
    [ 0.00,  0.14,  0.00],
    [ 0.00, -0.40,  0.00],
    [ 0.00, -0.40,  0.00],
    [ 0.00,  0.14,  0.00],
    [ 0.00, -0.08,  0.12],
    [ 0.00, -0.08,  0.12],
    [ 0.00,  0.14,  0.00],
    [-0.10,  0.08,  0.00],
    [ 0.10,  0.08,  0.00],
    [ 0.00,  0.16,  0.00],
    [-0.18,  0.00,  0.00],
    [ 0.18,  0.00,  0.00],
    [-0.28,  0.00,  0.00],
    [ 0.28,  0.00,  0.00],
    [-0.25,  0.00,  0.00],
    [ 0.25,  0.00,  0.00],
    [-0.08,  0.00,  0.00],
    [ 0.08,  0.00,  0.00],
], dtype=np.float32)

FOOT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)
UPPER_BODY_JOINTS = np.array([13, 14, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
COLLISION_PAIRS = (
    (18, 3), (19, 3), (20, 3), (21, 3), (22, 3), (23, 3),
    (18, 6), (19, 6), (20, 6), (21, 6), (22, 6), (23, 6),
    (18, 9), (19, 9), (20, 9), (21, 9), (22, 9), (23, 9),
    (20, 12), (21, 12), (22, 12), (23, 12),
    (20, 21), (22, 23),
)


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _load_motion(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        obj = x.item()
        x = obj.get("motion", obj.get("pose", x))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x


def _rot6d_to_matrix_np(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _fk_from_t151_np(motion: np.ndarray) -> np.ndarray:
    if (
        _enabled("V34_POSTPROCESS_USE_SMPL_FK", "1")
        and torch is not None
        and ax_from_6v is not None
        and SMPLSkeleton is not None
    ):
        try:
            device_name = os.getenv(
                "V34_POSTPROCESS_FK_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
            device = torch.device(device_name)
            root = torch.tensor(
                motion[:, [4, 5, 6]],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            q_6d = torch.tensor(
                motion[:, 7:151],
                dtype=torch.float32,
                device=device,
            ).reshape(1, motion.shape[0], 24, 6)
            with torch.no_grad():
                q_ax = ax_from_6v(q_6d)
                joints = SMPLSkeleton(device=device).forward(q_ax, root)
            return joints[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass

    t = motion.shape[0]
    root = motion[:, [4, 5, 6]].astype(np.float32)
    local_r = _rot6d_to_matrix_np(motion[:, 7:151].reshape(t, 24, 6))
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
    joints[:, 0] = root
    global_r[:, 0] = local_r[:, 0]
    for j in range(1, 24):
        p = int(PARENTS[j])
        global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
        off = OFFSETS[j][None, :, None]
        joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], off)[..., 0]
    return joints


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    flat = x.reshape(x.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return out.reshape(x.shape).astype(np.float32)


def _segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.astype(bool).tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= int(min_len):
                rows.append((start, i))
            start = None
    return rows


def _remove_short_contacts(mask: np.ndarray, min_len: int) -> np.ndarray:
    clean = np.zeros_like(mask, dtype=bool)
    for foot in range(mask.shape[1]):
        for start, end in _segments(mask[:, foot], int(min_len)):
            clean[start:end, foot] = True
    return clean


def _contact_mask(
    motion: np.ndarray,
    threshold: float,
    *,
    min_contact_frames: int = 1,
) -> np.ndarray:
    c = np.asarray(motion[:, 0:4], dtype=np.float32)
    if c.shape[1] != 4:
        label_contact = np.zeros((len(motion), 4), dtype=bool)
    else:
        label_contact = c >= float(threshold)
    if not _enabled("V34_KINEMATIC_CONTACT_INFER", "1") or len(motion) < 2:
        return _remove_short_contacts(label_contact, max(1, int(min_contact_frames)))
    try:
        joints = _fk_from_t151_np(motion)
        feet = joints[:, FOOT_JOINTS, :]
        foot_y = feet[:, :, 1]
        q = float(np.clip(_env_float("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
        floor_y = float(np.quantile(foot_y.reshape(-1), q))
        height_gate = foot_y <= floor_y + _env_float("V34_KIN_CONTACT_HEIGHT", 0.045)
        speed = np.zeros(foot_y.shape, dtype=np.float32)
        speed[1:] = np.linalg.norm(
            feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]],
            axis=-1,
        )
        speed_gate = speed <= _env_float("V34_KIN_CONTACT_SPEED", 0.035)
        vote_threshold = int(
            np.clip(_env_float("V34_KIN_CONTACT_VOTE_THRESHOLD", 2.0), 1.0, 3.0)
        )
        votes = (
            label_contact.astype(np.int32)
            + height_gate.astype(np.int32)
            + speed_gate.astype(np.int32)
        )
        inferred = votes >= vote_threshold
        return _remove_short_contacts(
            inferred,
            max(1, int(_env_float("V34_KIN_CONTACT_MIN_FRAMES", min_contact_frames))),
        )
    except Exception:
        return _remove_short_contacts(label_contact, max(1, int(min_contact_frames)))


def _mean_contact_speed(joints: np.ndarray, contact: np.ndarray) -> float:
    feet = joints[:, FOOT_JOINTS, :]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    active = contact.astype(bool)
    if not np.any(active):
        return 0.0
    return float(np.mean(speed[active]))


def _collision_stats(joints: np.ndarray, radius: float) -> Dict[str, float]:
    dists = []
    for a, b in COLLISION_PAIRS:
        dists.append(np.linalg.norm(joints[:, a] - joints[:, b], axis=-1))
    if not dists:
        return {"risk": 0.0, "min_distance": 1.0, "bad_frames": 0}
    dist = np.stack(dists, axis=1)
    pen = np.maximum(0.0, float(radius) - dist)
    return {
        "risk": float(np.mean(np.square(pen / max(float(radius), 1e-8)))),
        "min_distance": float(np.min(dist)),
        "bad_frames": int(np.sum(np.any(pen > 0, axis=1))),
    }


def _quality_audit(
    motion: np.ndarray,
    *,
    contact_threshold: float,
    min_contact_frames: int,
    floor_margin: float,
    collision_radius: float,
) -> Dict[str, float]:
    joints = _fk_from_t151_np(motion)
    feet = joints[:, FOOT_JOINTS, :]
    feet_y = feet[:, :, 1]
    q = float(np.clip(_env_float("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
    floor_y = float(
        _env_float("V34_FLOOR_Y", float(np.quantile(feet_y.reshape(-1), q)))
    )
    contact = _contact_mask(
        motion,
        contact_threshold,
        min_contact_frames=int(min_contact_frames),
    )
    feet_xz = feet[:, :, [0, 2]]
    skate_vals: List[np.ndarray] = []
    for foot in range(4):
        for start, end in _segments(contact[:, foot], max(2, int(min_contact_frames))):
            if end - start > 1:
                skate_vals.append(
                    np.linalg.norm(
                        feet_xz[start + 1:end, foot] - feet_xz[start:end - 1, foot],
                        axis=-1,
                    )
                )
    skate = float(np.mean(np.concatenate(skate_vals))) if skate_vals else 0.0

    root_y = motion[:, 5]
    root_v = np.diff(root_y, prepend=root_y[:1])
    root_a = np.diff(root_v, prepend=root_v[:1])
    vel = np.diff(joints, axis=0, prepend=joints[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    jerk = np.diff(acc, axis=0, prepend=acc[:1])
    mean_jerk = np.linalg.norm(jerk, axis=-1).mean(axis=-1)
    collision = _collision_stats(joints, collision_radius)

    penetration = feet_y - (floor_y + float(floor_margin))
    return {
        "floor_y": float(floor_y),
        "contact_ratio": float(np.mean(contact)) if contact.size else 0.0,
        "foot_skate_mean_mpf": float(skate),
        "foot_penetration_min_m": float(np.min(penetration)) if penetration.size else 0.0,
        "root_y_range_m": float(np.max(root_y) - np.min(root_y)) if len(root_y) else 0.0,
        "root_y_acc_mean": float(np.mean(np.abs(root_a))) if len(root_a) else 0.0,
        "root_y_acc_p95": float(np.percentile(np.abs(root_a), 95)) if len(root_a) else 0.0,
        "mean_joint_jerk_max": float(np.max(mean_jerk)) if len(mean_jerk) else 0.0,
        "mean_joint_jerk_p95": float(np.percentile(mean_jerk, 95)) if len(mean_jerk) else 0.0,
        "collision_risk": float(collision["risk"]),
        "collision_min_distance": float(collision["min_distance"]),
        "collision_bad_frames": float(collision["bad_frames"]),
    }


def _quality_rejection_signal(metrics: Dict[str, float]) -> Dict[str, object]:
    reasons: List[str] = []
    if metrics.get("foot_skate_mean_mpf", 0.0) > _env_float("V34_REJECT_MAX_SKATE_MPF", 0.035):
        reasons.append("foot_sliding")
    if metrics.get("foot_penetration_min_m", 0.0) < _env_float("V34_REJECT_MIN_FOOT_PENETRATION", -0.015):
        reasons.append("floor_penetration")
    if metrics.get("mean_joint_jerk_p95", 0.0) > _env_float("V34_REJECT_MAX_JERK_P95", 1200.0):
        reasons.append("high_jitter")
    if metrics.get("collision_risk", 0.0) > _env_float("V34_REJECT_MAX_COLLISION_RISK", 0.20):
        reasons.append("self_collision_proxy")
    return {
        "accepted": not reasons,
        "reject_reasons": reasons,
        "planner_action": (
            "accept"
            if not reasons
            else "reroute_local_phrase_or_mask_failed_edges"
        ),
    }


def contact_lock_root(
    motion: np.ndarray,
    *,
    contact_threshold: float,
    min_contact_frames: int,
    strength: float,
    smooth_window: int,
    max_correction: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = motion.astype(np.float32, copy=True)
    joints = _fk_from_t151_np(out)
    contact = _contact_mask(
        out,
        contact_threshold,
        min_contact_frames=int(min_contact_frames),
    )
    correction = np.zeros((len(out), 2), dtype=np.float32)
    counts = np.zeros((len(out), 1), dtype=np.float32)
    segment_count = 0
    feet = joints[:, FOOT_JOINTS, :]
    for local_foot in range(4):
        for start, end in _segments(contact[:, local_foot], min_contact_frames):
            segment = feet[start:end, local_foot, :][:, [0, 2]]
            anchor = np.median(segment, axis=0)
            correction[start:end] += anchor[None, :] - segment
            counts[start:end] += 1.0
            segment_count += 1
    active = counts[:, 0] > 0
    correction[active] /= np.maximum(counts[active], 1.0)
    if max_correction > 0:
        norm = np.linalg.norm(correction, axis=1, keepdims=True)
        correction *= np.minimum(1.0, float(max_correction) / np.maximum(norm, 1e-8))
    correction = _moving_average(correction, smooth_window)
    out[:, 4] += float(strength) * correction[:, 0]
    out[:, 6] += float(strength) * correction[:, 1]
    after_joints = _fk_from_t151_np(out)
    return out, {
        "enabled": True,
        "contact_segments": int(segment_count),
        "active_contact_frames": int(np.sum(active)),
        "mean_contact_speed_before": _mean_contact_speed(joints, contact),
        "mean_contact_speed_after": _mean_contact_speed(after_joints, contact),
        "max_root_xz_correction": float(np.max(np.linalg.norm(correction, axis=1))) if len(out) else 0.0,
        "contact_threshold": float(contact_threshold),
        "contact_vote_threshold": float(_env_float("V34_KIN_CONTACT_VOTE_THRESHOLD", 2.0)),
        "kinematic_contact_min_frames": int(
            _env_float("V34_KIN_CONTACT_MIN_FRAMES", min_contact_frames)
        ),
        "strength": float(strength),
    }


def enforce_floor_clearance(
    motion: np.ndarray,
    *,
    enabled: bool,
    margin: float,
    strength: float,
    smooth_window: int,
    max_lift: float,
    contact_threshold: float,
    min_contact_frames: int,
    support_damping: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    joints = _fk_from_t151_np(motion)
    feet_y = joints[:, FOOT_JOINTS, 1]
    q = float(np.clip(_env_float("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
    floor_y = float(
        _env_float("V34_FLOOR_Y", float(np.quantile(feet_y.reshape(-1), q)))
    )
    min_foot_y = np.min(feet_y, axis=1)
    penetration = np.maximum(0.0, floor_y + float(margin) - min_foot_y)
    before = {
        "floor_y": float(floor_y),
        "min_foot_y": float(np.min(feet_y)) if feet_y.size else 0.0,
        "penetrating_frames": int(np.sum(penetration > 1e-6)),
        "max_penetration": float(np.max(penetration)) if len(penetration) else 0.0,
        "mean_penetration": float(np.mean(penetration)) if len(penetration) else 0.0,
    }
    if not enabled or len(motion) == 0:
        return motion.astype(np.float32, copy=True), {
            "enabled": bool(enabled),
            "skipped": True,
            "before": before,
        }
    lift = penetration.astype(np.float32)
    if max_lift > 0:
        lift = np.minimum(lift, float(max_lift))
    lift = _moving_average(lift[:, None], int(smooth_window))[:, 0]
    out = motion.astype(np.float32, copy=True)
    out[:, 5] += float(strength) * lift
    if float(support_damping) > 0.0 and len(out) > 2:
        contact = _contact_mask(
            out,
            contact_threshold,
            min_contact_frames=int(min_contact_frames),
        )
        support = np.any(contact, axis=1)
        y = out[:, 5].copy()
        velocity = np.zeros_like(y)
        velocity[1:] = y[1:] - y[:-1]
        damped_velocity = velocity.copy()
        damped_velocity[support] *= max(0.0, 1.0 - float(support_damping))
        y_damped = y.copy()
        y_damped[1:] = y[0] + np.cumsum(damped_velocity[1:])
        out[:, 5] = (0.65 * y + 0.35 * y_damped).astype(np.float32)
    else:
        support = np.zeros((len(out),), dtype=bool)
    after_joints = _fk_from_t151_np(out)
    after_feet_y = after_joints[:, FOOT_JOINTS, 1]
    after_pen = np.maximum(0.0, floor_y + float(margin) - np.min(after_feet_y, axis=1))
    return out, {
        "enabled": True,
        "skipped": False,
        "margin": float(margin),
        "strength": float(strength),
        "smooth_window": int(smooth_window),
        "max_lift": float(max_lift),
        "support_damping": float(support_damping),
        "support_frame_ratio": float(np.mean(support)) if len(support) else 0.0,
        "before": before,
        "after": {
            "min_foot_y": float(np.min(after_feet_y)) if after_feet_y.size else 0.0,
            "penetrating_frames": int(np.sum(after_pen > 1e-6)),
            "max_penetration": float(np.max(after_pen)) if len(after_pen) else 0.0,
            "mean_penetration": float(np.mean(after_pen)) if len(after_pen) else 0.0,
        },
        "max_root_y_lift": float(np.max(lift)) if len(lift) else 0.0,
    }


def enforce_root_y_physics(
    motion: np.ndarray,
    *,
    contact_threshold: float,
    min_flight_frames: int,
    parabola_strength: float,
    min_arc_lift: float,
    max_arc_lift: float,
    landing_frames: int,
    landing_max_drop: float,
    landing_strength: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = motion.astype(np.float32, copy=True)
    contact = _contact_mask(
        out,
        contact_threshold,
        min_contact_frames=max(2, int(_env_float("V34_KIN_CONTACT_MIN_FRAMES", 3))),
    )
    support = np.any(contact, axis=1)
    flight_segments = _segments(~support, min_flight_frames)
    y = out[:, 5].copy()
    corrected = y.copy()

    for start, end in flight_segments:
        n = end - start
        if n < 2:
            continue
        u = np.linspace(0.0, 1.0, n, dtype=np.float32)
        y0 = float(y[start])
        y1 = float(y[end - 1])
        linear = (1.0 - u) * y0 + u * y1
        observed_lift = float(np.max(y[start:end]) - max(y0, y1))
        lift = float(np.clip(max(observed_lift, min_arc_lift), 0.0, max_arc_lift))
        parabola = linear + lift * 4.0 * u * (1.0 - u)
        corrected[start:end] = (
            (1.0 - float(parabola_strength)) * corrected[start:end]
            + float(parabola_strength) * parabola
        )

    landing_count = 0
    if landing_frames > 2 and landing_strength > 0:
        transitions = np.flatnonzero((support[1:] == True) & (support[:-1] == False)) + 1
        for t in transitions:
            end = min(len(out), t + int(landing_frames))
            n = end - t
            if n < 3:
                continue
            pre = max(0, t - 3)
            downward_speed = max(0.0, float(np.mean(y[pre:t]) - y[t]))
            amp = float(np.clip(0.35 * downward_speed + 0.012, 0.0, landing_max_drop))
            u = np.linspace(0.0, 1.0, n, dtype=np.float32)
            shock = -amp * np.sin(np.pi * u) * np.exp(-1.35 * u)
            corrected[t:end] += float(landing_strength) * shock
            landing_count += 1

    out[:, 5] = corrected.astype(np.float32)
    return out, {
        "enabled": True,
        "flight_segments": int(len(flight_segments)),
        "landing_events": int(landing_count),
        "root_y_delta_mean": float(np.mean(np.abs(corrected - y))) if len(y) else 0.0,
        "root_y_delta_max": float(np.max(np.abs(corrected - y))) if len(y) else 0.0,
        "parabola_strength": float(parabola_strength),
        "landing_strength": float(landing_strength),
    }


def smooth_rotations_only(
    motion: np.ndarray,
    *,
    rotation_window: int,
    strength: float,
) -> np.ndarray:
    out = motion.astype(np.float32, copy=True)
    strength = float(np.clip(strength, 0.0, 1.0))
    if rotation_window > 1 and strength > 0:
        smoothed = _moving_average(out[:, 7:151], rotation_window)
        out[:, 7:151] = (1.0 - strength) * out[:, 7:151] + strength * smoothed
    return out


def _rot6d_to_matrix_torch(x: "torch.Tensor") -> "torch.Tensor":
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / torch.clamp(torch.linalg.norm(a1, dim=-1, keepdim=True), min=1e-8)
    proj = torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = b2 / torch.clamp(torch.linalg.norm(b2, dim=-1, keepdim=True), min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _fk_torch(motion: "torch.Tensor") -> "torch.Tensor":
    t = motion.shape[0]
    root = motion[:, [4, 5, 6]]
    local_r = _rot6d_to_matrix_torch(motion[:, 7:151].reshape(t, 24, 6))
    joints = []
    global_r = []
    offsets = torch.as_tensor(OFFSETS, dtype=motion.dtype, device=motion.device)
    for j in range(24):
        if j == 0:
            joints.append(root)
            global_r.append(local_r[:, 0])
        else:
            p = int(PARENTS[j])
            r = torch.matmul(global_r[p], local_r[:, j])
            off = offsets[j].view(1, 3, 1)
            pos = joints[p] + torch.matmul(global_r[p], off).squeeze(-1)
            joints.append(pos)
            global_r.append(r)
    return torch.stack(joints, dim=1)


def collision_aware_ik(
    motion: np.ndarray,
    *,
    enabled: bool,
    radius: float,
    steps: int,
    lr: float,
    collision_weight: float,
    reg_weight: float,
    temporal_weight: float,
    device: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
    before = _collision_stats(_fk_from_t151_np(motion), radius)
    if not enabled or torch is None or before["bad_frames"] == 0 or steps <= 0:
        before["enabled"] = bool(enabled and torch is not None)
        before["skipped"] = True
        return motion.astype(np.float32, copy=True), before

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    base = torch.as_tensor(motion.astype(np.float32), device=dev)
    upper_idx = torch.as_tensor(UPPER_BODY_JOINTS, dtype=torch.long, device=dev)
    original_upper = base[:, 7:151].reshape(-1, 24, 6)[:, upper_idx].detach()
    var = original_upper.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([var], lr=float(lr))
    pairs = [(int(a), int(b)) for a, b in COLLISION_PAIRS]

    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        full_rot = base[:, 7:151].reshape(-1, 24, 6).clone()
        full_rot[:, upper_idx] = var
        candidate = base.clone()
        candidate[:, 7:151] = full_rot.reshape(-1, 144)
        joints = _fk_torch(candidate)
        losses = []
        for a, b in pairs:
            d = torch.linalg.norm(joints[:, a] - joints[:, b], dim=-1)
            losses.append(torch.relu(float(radius) - d) ** 2)
        collision_loss = torch.stack(losses, dim=1).mean()
        reg_loss = torch.mean((var - original_upper) ** 2)
        if var.shape[0] > 2:
            temporal_loss = torch.mean((var[1:] - var[:-1]) ** 2)
        else:
            temporal_loss = torch.zeros((), dtype=var.dtype, device=var.device)
        loss = (
            float(collision_weight) * collision_loss
            + float(reg_weight) * reg_loss
            + float(temporal_weight) * temporal_loss
        )
        loss.backward()
        opt.step()

    out = motion.astype(np.float32, copy=True)
    final_upper = var.detach().cpu().numpy()
    rot = out[:, 7:151].reshape(-1, 24, 6)
    rot[:, UPPER_BODY_JOINTS] = final_upper
    out[:, 7:151] = rot.reshape(-1, 144)
    after = _collision_stats(_fk_from_t151_np(out), radius)
    return out, {
        "enabled": True,
        "skipped": False,
        "device": str(dev),
        "steps": int(steps),
        "radius": float(radius),
        "before": before,
        "after": after,
    }


def process_file(args: argparse.Namespace) -> Dict[str, object]:
    src = Path(args.motion)
    out_path = Path(args.out)
    motion = _load_motion(src)
    original = motion.copy()
    pre_audit = _quality_audit(
        original,
        contact_threshold=args.contact_threshold,
        min_contact_frames=args.min_contact_frames,
        floor_margin=args.floor_margin,
        collision_radius=args.collision_radius,
    )

    physics_summary: Dict[str, object] = {"enabled": False}
    if args.root_y_physics:
        motion, physics_summary = enforce_root_y_physics(
            motion,
            contact_threshold=args.contact_threshold,
            min_flight_frames=args.min_flight_frames,
            parabola_strength=args.parabola_strength,
            min_arc_lift=args.min_arc_lift,
            max_arc_lift=args.max_arc_lift,
            landing_frames=args.landing_frames,
            landing_max_drop=args.landing_max_drop,
            landing_strength=args.landing_strength,
        )

    collision_summary: Dict[str, object]
    motion, collision_summary = collision_aware_ik(
        motion,
        enabled=bool(args.collision_ik),
        radius=args.collision_radius,
        steps=args.collision_steps,
        lr=args.collision_lr,
        collision_weight=args.collision_weight,
        reg_weight=args.collision_reg_weight,
        temporal_weight=args.collision_temporal_weight,
        device=args.device,
    )

    contact_summary: Dict[str, object] = {"enabled": False}
    if args.contact_lock:
        motion, contact_summary = contact_lock_root(
            motion,
            contact_threshold=args.contact_threshold,
            min_contact_frames=args.min_contact_frames,
            strength=args.contact_lock_strength,
            smooth_window=args.contact_smooth_window,
            max_correction=args.max_root_correction,
        )

    floor_summary: Dict[str, object]
    motion, floor_summary = enforce_floor_clearance(
        motion,
        enabled=bool(args.floor_clearance),
        margin=args.floor_margin,
        strength=args.floor_strength,
        smooth_window=args.floor_smooth_window,
        max_lift=args.floor_max_lift,
        contact_threshold=args.contact_threshold,
        min_contact_frames=args.min_contact_frames,
        support_damping=args.floor_support_damping,
    )

    if args.smooth:
        motion = smooth_rotations_only(
            motion,
            rotation_window=args.rotation_smooth_window,
            strength=args.smooth_strength,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, motion.astype(np.float32))
    post_audit = _quality_audit(
        motion,
        contact_threshold=args.contact_threshold,
        min_contact_frames=args.min_contact_frames,
        floor_margin=args.floor_margin,
        collision_radius=args.collision_radius,
    )
    rejection_signal = _quality_rejection_signal(post_audit)

    summary = {
        "version": "v34_physics_plausible_postprocess",
        "input": str(src),
        "output": str(out_path),
        "frames": int(len(motion)),
        "pre_audit": pre_audit,
        "post_audit": post_audit,
        "planner_feedback": rejection_signal,
        "audit_improvement": {
            "foot_skate_mean_delta": float(
                pre_audit["foot_skate_mean_mpf"] - post_audit["foot_skate_mean_mpf"]
            ),
            "foot_penetration_min_delta": float(
                post_audit["foot_penetration_min_m"] - pre_audit["foot_penetration_min_m"]
            ),
            "jerk_p95_delta": float(
                pre_audit["mean_joint_jerk_p95"] - post_audit["mean_joint_jerk_p95"]
            ),
            "collision_risk_delta": float(
                pre_audit["collision_risk"] - post_audit["collision_risk"]
            ),
        },
        "root_y_physics": physics_summary,
        "collision_aware_ik": collision_summary,
        "contact_lock": contact_summary,
        "floor_clearance": floor_summary,
        "rotation_smooth": {
            "enabled": bool(args.smooth),
            "rotation_window": int(args.rotation_smooth_window),
            "strength": float(args.smooth_strength),
            "note": "root_y and endpoint coordinates are not blindly smoothed",
        },
        "root_xz_delta_mean": float(np.mean(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
        "root_xz_delta_max": float(np.max(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
        "root_y_delta_mean": float(np.mean(np.abs(motion[:, 5] - original[:, 5]))) if len(motion) else 0.0,
        "root_y_delta_max": float(np.max(np.abs(motion[:, 5] - original[:, 5]))) if len(motion) else 0.0,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--contact_threshold", type=float, default=0.65)

    parser.add_argument("--root_y_physics", type=int, default=1)
    parser.add_argument("--min_flight_frames", type=int, default=6)
    parser.add_argument("--parabola_strength", type=float, default=0.60)
    parser.add_argument("--min_arc_lift", type=float, default=0.012)
    parser.add_argument("--max_arc_lift", type=float, default=0.10)
    parser.add_argument("--landing_frames", type=int, default=8)
    parser.add_argument("--landing_max_drop", type=float, default=0.035)
    parser.add_argument("--landing_strength", type=float, default=0.75)

    parser.add_argument("--collision_ik", type=int, default=1)
    parser.add_argument("--collision_radius", type=float, default=0.16)
    parser.add_argument("--collision_steps", type=int, default=24)
    parser.add_argument("--collision_lr", type=float, default=0.025)
    parser.add_argument("--collision_weight", type=float, default=8.0)
    parser.add_argument("--collision_reg_weight", type=float, default=0.45)
    parser.add_argument("--collision_temporal_weight", type=float, default=0.02)

    parser.add_argument("--contact_lock", type=int, default=1)
    parser.add_argument("--min_contact_frames", type=int, default=8)
    parser.add_argument("--contact_lock_strength", type=float, default=0.85)
    parser.add_argument("--contact_smooth_window", type=int, default=11)
    parser.add_argument("--max_root_correction", type=float, default=0.18)
    parser.add_argument("--floor_clearance", type=int, default=1)
    parser.add_argument("--floor_margin", type=float, default=0.006)
    parser.add_argument("--floor_strength", type=float, default=0.95)
    parser.add_argument("--floor_smooth_window", type=int, default=5)
    parser.add_argument("--floor_max_lift", type=float, default=0.12)
    parser.add_argument("--floor_support_damping", type=float, default=0.18)

    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--rotation_smooth_window", type=int, default=3)
    parser.add_argument("--smooth_strength", type=float, default=0.20)
    args = parser.parse_args()
    process_file(args)


if __name__ == "__main__":
    main()
