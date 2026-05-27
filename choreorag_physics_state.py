"""
Physics-aware boundary features and safe stitching utilities for EDGE-Dunhuang.

Drop this file at repository root:
    /home/disk/lsm/storage/EDGE/choreorag_physics_state.py

Design goals
------------
1. Fast feature-space boundary matching without running SMPL/FK by default.
2. Prefer physical continuity over exact music onset timing.
3. Work with EDGE 151D motion representation:
      [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotations.
4. Provide robust prior stitching for 45-frame units into 120/150-frame phrases.

The implementation intentionally does not depend on model internals. It can be
used before diffusion generation, as a safer long-prior composer, or to build
training data for v3_unit_recon.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import math
import pickle

import numpy as np

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_START = 7
ROT_DIM = 6
NFEATS = 151

# SMPL-ish joint groups used by the existing project convention.
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
PELVIS_JOINTS = [0]


def _as_float_array(x, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def ensure_motion_2d(motion: np.ndarray, name: str = "motion") -> np.ndarray:
    motion = _as_float_array(motion, name)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 2 or motion.shape[-1] != NFEATS:
        raise ValueError(f"{name} must be [T,151], got {motion.shape}")
    return motion.astype(np.float32, copy=False)


def ensure_motion_bank(motions: np.ndarray, name: str = "motions") -> np.ndarray:
    motions = _as_float_array(motions, name)
    if motions.ndim == 2 and motions.shape[-1] == NFEATS:
        motions = motions[None]
    if motions.ndim != 3 or motions.shape[-1] != NFEATS:
        raise ValueError(f"{name} must be [N,T,151] or [T,151], got {motions.shape}")
    return motions.astype(np.float32, copy=False)


def rot_indices(joints: Sequence[int]) -> List[int]:
    out: List[int] = []
    for joint in joints:
        start = ROT_START + ROT_DIM * int(joint)
        out.extend(range(start, min(start + ROT_DIM, NFEATS)))
    return out


LOWER_ROT_IDXS = rot_indices(LOWER_JOINTS)
TORSO_ROT_IDXS = rot_indices(TORSO_JOINTS)
UPPER_ROT_IDXS = rot_indices(UPPER_JOINTS)
PELVIS_ROT_IDXS = rot_indices(PELVIS_JOINTS)
ALL_ROT_IDXS = list(range(ROT_START, NFEATS))


def frame_velocity(motion: np.ndarray, frame: int, window: int = 1) -> np.ndarray:
    """Centered finite-difference velocity for a frame."""
    motion = ensure_motion_2d(motion)
    t = int(np.clip(frame, 0, len(motion) - 1))
    w = max(1, int(window))
    left = max(0, t - w)
    right = min(len(motion) - 1, t + w)
    if right == left:
        return np.zeros((motion.shape[-1],), dtype=np.float32)
    return ((motion[right] - motion[left]) / float(right - left)).astype(np.float32)


def contact_lr(contact4: np.ndarray) -> np.ndarray:
    """Compress 4 contact channels into [left,right] support probabilities.

    EDGE convention generally stores four foot-contact channels. The exact
    ordering can differ across preprocesses, so we use a conservative grouping:
    first two channels = left side, last two channels = right side.
    """
    c = np.asarray(contact4, dtype=np.float32).reshape(-1)
    if c.size < 4:
        out = np.zeros((2,), dtype=np.float32)
        if c.size == 1:
            out[:] = c[0]
        elif c.size >= 2:
            out[:] = c[:2]
        return out.clip(0.0, 1.0)
    return np.array([float(np.max(c[:2])), float(np.max(c[2:4]))], dtype=np.float32).clip(0.0, 1.0)


def normalize_6d_rot(motion: np.ndarray) -> np.ndarray:
    """Project every 6D rotation block to a valid-ish orthonormal 6D frame.

    This is a light-weight fix for linear blending of 6D rotations. It follows
    the Gram-Schmidt idea used by common 6D rotation representations.
    """
    m = ensure_motion_2d(motion).copy()
    rot = m[:, ROT_START:NFEATS].reshape(len(m), 24, 6)
    a1 = rot[..., 0:3]
    a2 = rot[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    rot[..., 0:3] = b1
    rot[..., 3:6] = b2
    m[:, ROT_START:NFEATS] = rot.reshape(len(m), -1)
    return m.astype(np.float32)


@dataclass
class BoundaryState:
    root_y: float
    root_vx: float
    root_vy: float
    root_vz: float
    root_speed_xz: float
    contact_l: float
    contact_r: float
    contact_any: float
    lower_vel: float
    torso_vel: float
    upper_vel: float
    pelvis_vel: float
    frame_activity: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.root_y,
            self.root_vx,
            self.root_vy,
            self.root_vz,
            self.root_speed_xz,
            self.contact_l,
            self.contact_r,
            self.contact_any,
            self.lower_vel,
            self.torso_vel,
            self.upper_vel,
            self.pelvis_vel,
            self.frame_activity,
        ], dtype=np.float32)

    @staticmethod
    def from_vector(v: Sequence[float]) -> "BoundaryState":
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size < 13:
            arr = np.pad(arr, (0, 13 - arr.size), mode="constant")
        return BoundaryState(*map(float, arr[:13]))


def boundary_state(motion: np.ndarray, frame: int, velocity_window: int = 1) -> BoundaryState:
    """Compute a fast explicit physics-aware state vector at one boundary."""
    motion = ensure_motion_2d(motion)
    t = int(np.clip(frame, 0, len(motion) - 1))
    vel = frame_velocity(motion, t, velocity_window)
    root_v = vel[4:7]
    root_speed_xz = float(np.linalg.norm(root_v[[0, 2]]))
    contact = motion[t, CONTACT_SLICE].clip(0.0, 1.0)
    lr = contact_lr(contact)

    def group_vel(idxs: Sequence[int]) -> float:
        if not idxs:
            return 0.0
        return float(np.sqrt(np.mean(np.square(vel[list(idxs)]))) + 1e-8)

    lower = group_vel(LOWER_ROT_IDXS)
    torso = group_vel(TORSO_ROT_IDXS)
    upper = group_vel(UPPER_ROT_IDXS)
    pelvis = group_vel(PELVIS_ROT_IDXS)
    frame_activity = float(0.25 * root_speed_xz + 0.30 * lower + 0.25 * torso + 0.20 * upper)
    return BoundaryState(
        root_y=float(motion[t, ROOT_Y_IDX]),
        root_vx=float(root_v[0]),
        root_vy=float(root_v[1]),
        root_vz=float(root_v[2]),
        root_speed_xz=root_speed_xz,
        contact_l=float(lr[0]),
        contact_r=float(lr[1]),
        contact_any=float(np.max(contact)),
        lower_vel=lower,
        torso_vel=torso,
        upper_vel=upper,
        pelvis_vel=pelvis,
        frame_activity=frame_activity,
    )


def boundary_state_bank(motions: np.ndarray, frame: str = "entry") -> np.ndarray:
    motions = ensure_motion_bank(motions)
    out = []
    for m in motions:
        idx = 0 if frame == "entry" else len(m) - 1
        out.append(boundary_state(m, idx).to_vector())
    return np.stack(out, axis=0).astype(np.float32)


@dataclass
class TransitionWeights:
    # Hard-mask thresholds. They are conservative because the project chose to
    # sacrifice exact musical hit timing to preserve physical plausibility.
    max_root_y_delta: float = 0.30
    max_root_speed_delta: float = 0.35
    max_contact_l1: float = 1.60
    max_frame_activity_delta: float = 0.55

    # Soft penalty weights.
    root_y: float = 12.0
    root_vel: float = 4.0
    root_speed: float = 8.0
    contact: float = 10.0
    lower_vel: float = 5.0
    torso_vel: float = 2.0
    upper_vel: float = 1.0
    pelvis_vel: float = 3.0
    activity: float = 3.0

    # Nonlinear distance scale.
    exp_k: float = 1.5

    # Score weights. beta dominates by design.
    alpha_music: float = 0.25
    beta_physics: float = 1.0
    quality: float = 0.40
    novelty: float = 0.25

    @staticmethod
    def from_env() -> "TransitionWeights":
        import os
        w = TransitionWeights()
        for key in asdict(w):
            env_key = "EDGE_ONSET_" + key.upper()
            if env_key in os.environ:
                try:
                    setattr(w, key, float(os.environ[env_key]))
                except Exception:
                    pass
        return w


def transition_distance(
    current: BoundaryState | Sequence[float],
    entry: BoundaryState | Sequence[float],
    weights: Optional[TransitionWeights] = None,
    return_parts: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    """Physical distance between current boundary and candidate entry.

    Lower is better. Extremely incompatible candidates return +inf via hard
    masks to avoid airborne-to-ground or support-phase clashes.
    """
    weights = weights or TransitionWeights()
    a = current if isinstance(current, BoundaryState) else BoundaryState.from_vector(current)
    b = entry if isinstance(entry, BoundaryState) else BoundaryState.from_vector(entry)

    dy = abs(a.root_y - b.root_y)
    rv_a = np.array([a.root_vx, a.root_vy, a.root_vz], dtype=np.float32)
    rv_b = np.array([b.root_vx, b.root_vy, b.root_vz], dtype=np.float32)
    root_v = float(np.linalg.norm(rv_a - rv_b))
    root_speed_delta = abs(a.root_speed_xz - b.root_speed_xz)
    contact_a = np.array([a.contact_l, a.contact_r], dtype=np.float32)
    contact_b = np.array([b.contact_l, b.contact_r], dtype=np.float32)
    contact_l1 = float(np.sum(np.abs(contact_a - contact_b)))
    activity_delta = abs(a.frame_activity - b.frame_activity)

    hard_fail = (
        dy > weights.max_root_y_delta
        or root_speed_delta > weights.max_root_speed_delta
        or contact_l1 > weights.max_contact_l1
        or activity_delta > weights.max_frame_activity_delta
    )
    if hard_fail:
        parts = {
            "hard_fail": 1.0,
            "root_y_delta": float(dy),
            "root_vel_delta": float(root_v),
            "root_speed_delta": float(root_speed_delta),
            "contact_l1": float(contact_l1),
            "activity_delta": float(activity_delta),
        }
        return (float("inf"), parts) if return_parts else float("inf")

    contact_bce = float(-np.mean(
        contact_a * np.log(np.clip(contact_b, 1e-4, 1.0))
        + (1.0 - contact_a) * np.log(np.clip(1.0 - contact_b, 1e-4, 1.0))
    ))
    raw = 0.0
    raw += weights.root_y * dy * dy
    raw += weights.root_vel * root_v * root_v
    raw += weights.root_speed * root_speed_delta * root_speed_delta
    raw += weights.contact * contact_bce
    raw += weights.lower_vel * (a.lower_vel - b.lower_vel) ** 2
    raw += weights.torso_vel * (a.torso_vel - b.torso_vel) ** 2
    raw += weights.upper_vel * (a.upper_vel - b.upper_vel) ** 2
    raw += weights.pelvis_vel * (a.pelvis_vel - b.pelvis_vel) ** 2
    raw += weights.activity * activity_delta * activity_delta
    dist = float(math.expm1(min(20.0, weights.exp_k * raw)))
    parts = {
        "hard_fail": 0.0,
        "raw": float(raw),
        "dist": float(dist),
        "root_y_delta": float(dy),
        "root_vel_delta": float(root_v),
        "root_speed_delta": float(root_speed_delta),
        "contact_l1": float(contact_l1),
        "contact_bce": float(contact_bce),
        "activity_delta": float(activity_delta),
    }
    return (dist, parts) if return_parts else dist


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    window = min(window, len(x) if len(x) % 2 else len(x) - 1)
    if window <= 1:
        return x.astype(np.float32)
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    y = np.empty_like(x, dtype=np.float32)
    flat = x.reshape(len(x), -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(np.pad(flat[:, c], (pad, pad), mode="edge"), kernel, mode="valid")
    return out.reshape(x.shape).astype(np.float32)


def smooth_boundary_region(motion: np.ndarray, center: int, radius: int = 4) -> np.ndarray:
    motion = ensure_motion_2d(motion).copy()
    if radius <= 1:
        return motion
    lo = max(0, int(center) - int(radius))
    hi = min(len(motion), int(center) + int(radius) + 1)
    if hi - lo < 3:
        return motion
    region = motion[lo:hi].copy()
    # Do not smooth contact too aggressively; use short average and clip.
    region[:, CONTACT_SLICE] = moving_average(region[:, CONTACT_SLICE], 3).clip(0.0, 1.0)
    region[:, 4:7] = moving_average(region[:, 4:7], 5)
    region[:, ROT_START:NFEATS] = moving_average(region[:, ROT_START:NFEATS], 5)
    motion[lo:hi] = region
    motion = normalize_6d_rot(motion)
    return motion.astype(np.float32)


def blend_motions(
    prev: np.ndarray,
    next_motion: np.ndarray,
    cut_frame: int,
    blend_frames: int = 10,
    inplace_root: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Cut prev at cut_frame and blend into next_motion.

    This is a deterministic prior-level stitching step. Diffusion/SDEdit can be
    applied after this file if desired; the rough prior itself should already be
    physically plausible enough to pass safety gates.
    """
    prev = ensure_motion_2d(prev, "prev")
    nxt = ensure_motion_2d(next_motion, "next_motion")
    cut = int(np.clip(cut_frame, 1, len(prev)))
    bf = int(max(0, min(blend_frames, cut, len(nxt))))

    head = prev[:cut].copy()
    if bf <= 0:
        out = np.concatenate([head, nxt], axis=0)
        return out.astype(np.float32), {"blend_frames": 0.0, "cut_frame": float(cut)}

    # Align next root X/Z to current cut point before optional root lock.
    aligned = nxt.copy()
    root_delta = head[-1, [ROOT_X_IDX, ROOT_Z_IDX]] - aligned[0, [ROOT_X_IDX, ROOT_Z_IDX]]
    aligned[:, ROOT_X_IDX] += root_delta[0]
    aligned[:, ROOT_Z_IDX] += root_delta[1]

    prev_tail = head[-bf:].copy()
    next_head = aligned[:bf].copy()
    weights = np.linspace(0.0, 1.0, bf + 2, dtype=np.float32)[1:-1][:, None]
    blended = (1.0 - weights) * prev_tail + weights * next_head
    # Contacts should remain plausible; do not create half contacts too long.
    blended[:, CONTACT_SLICE] = np.where(
        weights < 0.5,
        prev_tail[:, CONTACT_SLICE],
        next_head[:, CONTACT_SLICE],
    )
    stitched = np.concatenate([head[:-bf], blended, aligned[bf:]], axis=0)
    if inplace_root:
        stitched[:, ROOT_X_IDX] -= stitched[0, ROOT_X_IDX]
        stitched[:, ROOT_Z_IDX] -= stitched[0, ROOT_Z_IDX]
        # Conservative root drift guard for in-place demo.
        stitched[:, ROOT_X_IDX] *= 0.05
        stitched[:, ROOT_Z_IDX] *= 0.05
    stitched = normalize_6d_rot(stitched)
    stitched = smooth_boundary_region(stitched, max(0, cut - bf // 2), radius=max(3, bf // 2))
    debug = {
        "blend_frames": float(bf),
        "cut_frame": float(cut),
        "root_align_dx": float(root_delta[0]),
        "root_align_dz": float(root_delta[1]),
    }
    return stitched.astype(np.float32), debug


def local_jump_metrics(motion: np.ndarray, boundaries: Sequence[int], radius: int = 2) -> Dict[str, float]:
    motion = ensure_motion_2d(motion)
    metrics: Dict[str, float] = {}
    diffs = np.linalg.norm(motion[1:, ROT_START:NFEATS] - motion[:-1, ROT_START:NFEATS], axis=-1)
    root_diffs = np.linalg.norm(motion[1:, [ROOT_X_IDX, ROOT_Z_IDX]] - motion[:-1, [ROOT_X_IDX, ROOT_Z_IDX]], axis=-1)
    for b in boundaries:
        lo = max(0, int(b) - radius)
        hi = min(len(diffs), int(b) + radius + 1)
        if hi > lo:
            metrics[f"boundary_{int(b)}_local_rot_jump_max"] = float(np.max(diffs[lo:hi]))
            metrics[f"boundary_{int(b)}_local_root_jump_max"] = float(np.max(root_diffs[lo:hi]))
    metrics["global_rot_jump_p95"] = float(np.percentile(diffs, 95)) if len(diffs) else 0.0
    metrics["global_root_jump_p95"] = float(np.percentile(root_diffs, 95)) if len(root_diffs) else 0.0
    root_xz = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    metrics["root_max_radius"] = float(np.linalg.norm(root_xz - root_xz[:1], axis=-1).max())
    metrics["root_final_x"] = float(root_xz[-1, 0] - root_xz[0, 0])
    metrics["root_final_z"] = float(root_xz[-1, 1] - root_xz[0, 1])
    activity = np.linalg.norm(motion[1:, ROT_START:NFEATS] - motion[:-1, ROT_START:NFEATS], axis=-1)
    metrics["segment_activity_mean"] = float(np.mean(activity)) if len(activity) else 0.0
    return metrics


def save_motion_pkl(motion: np.ndarray, path: str | Path, metadata: Optional[Dict] = None) -> None:
    motion = ensure_motion_2d(motion)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"motion": motion.astype(np.float32)}
    if metadata:
        payload.update(metadata)
    with p.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(payload: Dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
