#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V42.2 FINAL Root-Footplant Physics Optimizer for EDGE 151D
==========================================================

Purpose
-------
Fix visually obvious lower-body instability in accepted EDGE whole-song outputs,
especially support-foot sliding and linear/root-weightless motion, without
pretending that global foot XYZ can be written directly into EDGE 151D.

EDGE 151D convention used here
-----------------------------
- motion[:, 4:7] is global root translation [x, y, z]
- motion[:, 5] is root height / Root-Y
- motion[:, 7:151].reshape(T, 24, 6) is local joint rotation in 6D form
- FK foot joint ids are [7, 8, 10, 11]

Hard safety fixes included
--------------------------
1. No origin snap: foot IK targets are initialized from native FK positions,
   never zeros.
2. No floating anchor: stance anchors are sampled from contact-internal frames
   only, never start-1 / pre-contact flight frames.
3. No fake writeback: the .npy output only changes legal root channels [4,5,6].
   Foot IK targets are saved in a .npz file for a downstream lower-body IK.
4. C1-safe root-Y: flight parabola is mixed by a bell gate with zero weight at
   flight boundaries.
5. Damping fuse-off: landing damping stops as soon as the motion re-enters flight.
6. Rollback-if-worse: if safety metrics degrade, the output motion is rolled
   back to the input while still saving diagnostics and targets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import scipy.ndimage as ndi
except Exception:  # pragma: no cover - scipy is expected in the EDGE env
    ndi = None

ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT6D_START = 7
ROT6D_END = 151
NUM_JOINTS = 24
DEFAULT_FOOT_JOINTS = (7, 8, 10, 11)

# SMPL-like default kinematic tree. If the repository's V41 FK injector is
# available, its PARENTS/OFFSETS are used instead.
FALLBACK_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
FALLBACK_OFFSETS = np.array(
    [
        [0.0000, 0.0000, 0.0000],
        [0.0586, -0.0823, 0.0177],
        [-0.0603, -0.0905, 0.0135],
        [0.0044, 0.1244, -0.0384],
        [0.0435, -0.3865, 0.0080],
        [-0.0433, -0.3837, -0.0048],
        [0.0045, 0.1379, 0.0268],
        [-0.0148, -0.4269, -0.0374],
        [0.0195, -0.4200, -0.0346],
        [-0.0023, 0.0560, 0.0029],
        [0.0411, -0.0603, 0.1220],
        [-0.0348, -0.0621, 0.1309],
        [0.0264, 0.2146, -0.0375],
        [0.0714, 0.1138, -0.0189],
        [-0.0824, 0.1125, -0.0237],
        [0.0103, 0.0889, 0.0504],
        [0.1229, 0.0452, -0.0190],
        [-0.1132, 0.0469, -0.0085],
        [0.2553, -0.0156, -0.0229],
        [-0.2601, -0.0144, -0.0319],
        [0.2657, 0.0127, -0.0074],
        [-0.2691, 0.0067, -0.0060],
        [0.0867, -0.0106, -0.0156],
        [-0.0888, -0.0087, -0.0101],
    ],
    dtype=np.float32,
)


def _load_repo_fk_tree() -> Tuple[np.ndarray, np.ndarray, str]:
    """Try to reuse V41 FK constants if installed; otherwise use fallback."""
    try:
        # Works when EDGE root is in PYTHONPATH or current working directory.
        from tools import v41b_inject_min_foot_y_to_db as v41fk  # type: ignore

        parents = np.asarray(getattr(v41fk, "PARENTS"), dtype=np.int64)
        offsets = np.asarray(getattr(v41fk, "OFFSETS"), dtype=np.float32)
        if parents.shape[0] == NUM_JOINTS and offsets.shape == (NUM_JOINTS, 3):
            return parents, offsets, "repo.tools.v41b_inject_min_foot_y_to_db"
    except Exception:
        pass
    return FALLBACK_PARENTS, FALLBACK_OFFSETS, "fallback_smpl_like_tree"


PARENTS, OFFSETS, FK_TREE_SOURCE = _load_repo_fk_tree()


def rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    """Convert (...,6) 6D rotation representation to (...,3,3)."""
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def fk_24_from_edge151(motion: np.ndarray) -> np.ndarray:
    """Forward kinematics from EDGE 151D to global joints [T,24,3]."""
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < ROT6D_END:
        raise ValueError(f"Expected motion [T,>=151], got {motion.shape}")
    t = motion.shape[0]
    root = motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]].astype(np.float32)
    rot6d = motion[:, ROT6D_START:ROT6D_END].reshape(t, NUM_JOINTS, 6)
    local_r = rot6d_to_matrix(rot6d)
    global_r = np.zeros((t, NUM_JOINTS, 3, 3), dtype=np.float32)
    joints = np.zeros((t, NUM_JOINTS, 3), dtype=np.float32)
    global_r[:, 0] = local_r[:, 0]
    joints[:, 0] = root
    for j in range(1, NUM_JOINTS):
        p = int(PARENTS[j])
        if p < 0:
            global_r[:, j] = local_r[:, j]
            joints[:, j] = root
        else:
            global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
            offset = OFFSETS[j].astype(np.float32)[None, :, None]
            joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], offset)[..., 0]
    return joints


def contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return [start,end) regions where mask is True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    diff = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def smoothstep(edge0: float, edge1: float, x: np.ndarray | float) -> np.ndarray | float:
    denom = max(float(edge1 - edge0), 1e-8)
    y = np.clip((np.asarray(x) - edge0) / denom, 0.0, 1.0)
    out = y * y * (3.0 - 2.0 * y)
    if np.isscalar(x):
        return float(out)
    return out


def median_bool_filter(x: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return x.astype(bool)
    if ndi is None:
        return x.astype(bool)
    return ndi.median_filter(x.astype(np.uint8), size=size).astype(bool)


def smooth_array(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or ndi is None:
        return x
    return ndi.gaussian_filter1d(x, sigma=float(sigma), axis=0, mode="nearest")


@dataclass
class V42Config:
    fps: float = 30.0
    foot_joint_ids: Tuple[int, ...] = DEFAULT_FOOT_JOINTS
    floor_quantile: float = 0.05
    height_margin: float = 0.045
    speed_gate_mpf: float = 0.035
    contact_high: float = 0.55
    contact_low: float = 0.35
    contact_median_size: int = 5
    min_stance_frames: int = 3
    anchor_frames: int = 3
    ramp_in_frames: int = 6
    ramp_out_frames: int = 4
    root_xz_strength: float = 0.65
    root_xz_max_correction: float = 0.075
    root_xz_smooth_sigma: float = 2.0
    root_y_strength: float = 0.35
    gravity: float = 9.81
    min_flight_frames: int = 5
    damping_seconds: float = 0.28
    damping_max_dip: float = 0.018
    floor_clearance_margin: float = 0.004
    floor_lift_max: float = 0.08
    rollback_if_worse: bool = True
    max_skate_ratio: float = 1.08
    max_jerk_ratio: float = 1.20
    min_accepted_penetration: float = -0.030
    max_accepted_skate_p95: float = 0.018
    max_accepted_root_xz_delta: float = 0.12
    max_accepted_root_y_delta: float = 0.20

    @staticmethod
    def from_json(path: Optional[str]) -> "V42Config":
        cfg = V42Config()
        if not path:
            return cfg
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)
        data = json.load(open(p, "r", encoding="utf-8"))
        for k, v in data.items():
            if not hasattr(cfg, k):
                continue
            if k == "foot_joint_ids":
                v = tuple(int(x) for x in v)
            setattr(cfg, k, v)
        return cfg


def derive_contact_confidence(
    foot_pos: np.ndarray,
    cfg: V42Config,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Derive contact confidence from FK foot height and velocity."""
    foot_y = foot_pos[..., 1]
    floor_y = float(np.percentile(foot_y.reshape(-1), cfg.floor_quantile * 100.0))
    height_score = 1.0 - (foot_y - floor_y) / max(cfg.height_margin, 1e-8)
    height_score = np.clip(height_score, 0.0, 1.0)

    vel = np.zeros(foot_pos.shape[:2], dtype=np.float32)
    vel[1:] = np.linalg.norm(foot_pos[1:, :, [0, 2]] - foot_pos[:-1, :, [0, 2]], axis=-1)
    speed_score = 1.0 - vel / max(cfg.speed_gate_mpf, 1e-8)
    speed_score = np.clip(speed_score, 0.0, 1.0)

    confidence = 0.62 * height_score + 0.38 * speed_score

    clean = np.zeros_like(confidence, dtype=bool)
    for f in range(confidence.shape[1]):
        state = False
        for t, p in enumerate(confidence[:, f]):
            if p >= cfg.contact_high:
                state = True
            elif p <= cfg.contact_low:
                state = False
            clean[t, f] = state
        clean[:, f] = median_bool_filter(clean[:, f], cfg.contact_median_size)

    meta = {
        "floor_y": floor_y,
        "confidence_mean": float(confidence.mean()),
        "clean_contact_ratio": float(clean.mean()),
        "height_margin": float(cfg.height_margin),
        "speed_gate_mpf": float(cfg.speed_gate_mpf),
        "contact_high": float(cfg.contact_high),
        "contact_low": float(cfg.contact_low),
    }
    return confidence.astype(np.float32), clean, floor_y, meta


def generate_foot_ik_targets(
    foot_pos: np.ndarray,
    clean_contacts: np.ndarray,
    cfg: V42Config,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Generate footplant IK targets. This function intentionally does NOT write
    global foot XYZ into EDGE 151D. It only returns targets for downstream IK.
    """
    # Critical safety base: never zeros. Non-contact and skipped segments remain native.
    targets = foot_pos.copy()
    segment_preview: List[Dict[str, object]] = []
    skipped_short = 0
    edited_segments = 0

    for f in range(foot_pos.shape[1]):
        regions = contiguous_regions(clean_contacts[:, f])
        for start, end in regions:
            length = end - start
            if length < cfg.min_stance_frames:
                skipped_short += 1
                continue

            # Critical fix: anchor only from contact-internal frames.
            anchor_end = min(start + cfg.anchor_frames, end)
            anchor = np.mean(foot_pos[start:anchor_end, f, :], axis=0)
            edited_segments += 1

            for i, t in enumerate(range(start, end)):
                win = max(length - 1, 1)
                in_w = smoothstep(0, cfg.ramp_in_frames, i)
                out_w = smoothstep(0, cfg.ramp_out_frames, length - 1 - i)
                w = float(min(in_w, out_w))
                targets[t, f, :] = (1.0 - w) * foot_pos[t, f, :] + w * anchor

            if len(segment_preview) < 24:
                segment_preview.append(
                    {
                        "foot": int(f),
                        "start": int(start),
                        "end": int(end),
                        "frames": int(length),
                        "anchor_source": "contact_internal_only",
                        "anchor_start": int(start),
                        "anchor_end": int(anchor_end),
                        "anchor_xyz": [float(x) for x in anchor.tolist()],
                    }
                )

    diff = np.linalg.norm(targets - foot_pos, axis=-1)
    non_contact = ~clean_contacts
    zero = np.linalg.norm(targets.reshape(-1, 3), axis=1) < 1e-8
    meta = {
        "version": "v42_2_final_target_generator",
        "no_zero_initialization": True,
        "target_initialized_from_native_foot_global_pos": True,
        "anchor_sampling_contact_internal_only": True,
        "no_precontact_flight_anchor": True,
        "skipped_short_segments": int(skipped_short),
        "edited_segments": int(edited_segments),
        "zero_target_count": int(zero.sum()),
        "non_contact_diff_max": float(diff[non_contact].max()) if non_contact.any() else 0.0,
        "contact_diff_p95": float(np.percentile(diff[clean_contacts], 95)) if clean_contacts.any() else 0.0,
        "segment_preview": segment_preview,
    }
    return targets.astype(np.float32), meta


def compute_root_xz_countermotion(
    foot_pos: np.ndarray,
    targets: np.ndarray,
    clean_contacts: np.ndarray,
    confidence: np.ndarray,
    cfg: V42Config,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Aggregate footplant target errors into legal root XZ counter-motion."""
    t = foot_pos.shape[0]
    corr_sum = np.zeros((t, 2), dtype=np.float32)
    weight_sum = np.zeros((t, 1), dtype=np.float32)

    delta_xz = targets[:, :, [0, 2]] - foot_pos[:, :, [0, 2]]
    for f in range(foot_pos.shape[1]):
        # Contact confidence weights; zero outside clean contacts.
        w = confidence[:, f:f+1].astype(np.float32) * clean_contacts[:, f:f+1].astype(np.float32)
        corr_sum += w * delta_xz[:, f, :]
        weight_sum += w

    corr = corr_sum / np.maximum(weight_sum, 1e-8)
    corr[weight_sum[:, 0] <= 1e-8] = 0.0
    corr = smooth_array(corr, cfg.root_xz_smooth_sigma)

    mag = np.linalg.norm(corr, axis=1)
    scale = np.minimum(1.0, cfg.root_xz_max_correction / np.maximum(mag, 1e-8))
    corr = corr * scale[:, None]
    corr *= float(cfg.root_xz_strength)

    meta = {
        "enabled": True,
        "mode": "legal_root_xz_countermotion_from_footplant_targets",
        "applied_corr_mean": float(np.linalg.norm(corr, axis=1).mean()),
        "applied_corr_p95": float(np.percentile(np.linalg.norm(corr, axis=1), 95)),
        "applied_corr_max": float(np.linalg.norm(corr, axis=1).max()),
        "strength": float(cfg.root_xz_strength),
        "max_correction": float(cfg.root_xz_max_correction),
        "smooth_sigma": float(cfg.root_xz_smooth_sigma),
    }
    return corr.astype(np.float32), meta


def apply_root_y_weight(
    root_y: np.ndarray,
    clean_contacts: np.ndarray,
    cfg: V42Config,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """C1-safe flight parabola and landing damping on root-y only."""
    out = root_y.astype(np.float32).copy()
    any_contact = clean_contacts.any(axis=1)
    is_flight = ~any_contact
    flight_regions = contiguous_regions(is_flight)
    applied_flights = 0

    for start, end in flight_regions:
        n = end - start
        if n < cfg.min_flight_frames:
            continue
        left_idx = max(0, start - 1)
        right_idx = min(len(root_y) - 1, end)
        y0 = float(root_y[left_idx])
        y1 = float(root_y[right_idx])
        duration = max((right_idx - left_idx) / cfg.fps, 1.0 / cfg.fps)
        v0 = (y1 - y0 + 0.5 * cfg.gravity * duration * duration) / duration

        for k, ti in enumerate(range(start, end)):
            if n <= 1:
                phase = 0.0
            else:
                phase = k / float(n - 1)
            # C1-safe gate: exactly zero at first and last flight frames.
            bell = math.sin(math.pi * phase) ** 2
            blend = float(cfg.root_y_strength) * bell
            tau = (ti - left_idx) / cfg.fps
            parabola = y0 + v0 * tau - 0.5 * cfg.gravity * tau * tau
            out[ti] = (1.0 - blend) * out[ti] + blend * parabola
        applied_flights += 1

    landings = np.where((any_contact[1:] == True) & (any_contact[:-1] == False))[0] + 1
    damping_frames = max(1, int(round(cfg.damping_seconds * cfg.fps)))
    applied_landings = 0
    for land_idx in landings:
        applied = False
        for i in range(damping_frames):
            ti = land_idx + i
            if ti >= len(out):
                break
            # Fuse-off: stop damping if the dancer re-enters flight.
            if is_flight[ti]:
                break
            phase = i / max(damping_frames - 1, 1)
            dip = cfg.damping_max_dip * math.exp(-8.0 * phase) * math.sin(math.pi * phase)
            out[ti] -= dip
            applied = True
        if applied:
            applied_landings += 1

    meta = {
        "enabled": True,
        "root_y_idx": ROOT_Y_IDX,
        "c1_bell_weight": True,
        "damping_fuse_off_on_flight": True,
        "flight_segments_total": int(len(flight_regions)),
        "flight_segments_applied": int(applied_flights),
        "landings_total": int(len(landings)),
        "landings_applied": int(applied_landings),
        "root_y_strength": float(cfg.root_y_strength),
        "damping_seconds": float(cfg.damping_seconds),
        "damping_max_dip": float(cfg.damping_max_dip),
        "delta_mean": float(np.mean(out - root_y)),
        "delta_max_abs": float(np.max(np.abs(out - root_y))) if len(out) else 0.0,
    }
    return out.astype(np.float32), meta


def apply_floor_clearance_guard(
    motion: np.ndarray,
    foot_joint_ids: Sequence[int],
    cfg: V42Config,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Prevent root-level edits from creating new floor penetration."""
    out = motion.copy()
    joints = fk_24_from_edge151(out)
    foot_y = joints[:, list(foot_joint_ids), 1]
    floor_y = float(np.percentile(foot_y.reshape(-1), cfg.floor_quantile * 100.0))
    min_clearance = foot_y.min(axis=1) - floor_y
    needed = np.maximum(0.0, cfg.floor_clearance_margin - min_clearance)
    needed = np.minimum(needed, cfg.floor_lift_max)
    needed = smooth_array(needed[:, None], 2.0)[:, 0]
    out[:, ROOT_Y_IDX] += needed.astype(np.float32)
    meta = {
        "enabled": True,
        "floor_y": floor_y,
        "margin": float(cfg.floor_clearance_margin),
        "lift_mean": float(needed.mean()),
        "lift_p95": float(np.percentile(needed, 95)),
        "lift_max": float(needed.max()),
        "lift_frames": int((needed > 1e-8).sum()),
    }
    return out, meta


def audit_motion(
    motion: np.ndarray,
    cfg: V42Config,
    clean_contacts: Optional[np.ndarray] = None,
    floor_y: Optional[float] = None,
) -> Dict[str, float]:
    joints = fk_24_from_edge151(motion)
    foot = joints[:, list(cfg.foot_joint_ids), :]
    foot_y = foot[..., 1]
    if floor_y is None:
        floor_y = float(np.percentile(foot_y.reshape(-1), cfg.floor_quantile * 100.0))
    if clean_contacts is None:
        _, clean_contacts, _, _ = derive_contact_confidence(foot, cfg)

    vel = np.zeros(foot.shape[:2], dtype=np.float32)
    vel[1:] = np.linalg.norm(foot[1:, :, [0, 2]] - foot[:-1, :, [0, 2]], axis=-1)
    if clean_contacts.any():
        skate_values = vel[clean_contacts]
    else:
        skate_values = vel.reshape(-1)

    if joints.shape[0] >= 4:
        jerk = np.diff(joints, n=3, axis=0)
        jerk_frame = np.linalg.norm(jerk, axis=-1).mean(axis=-1)
        jerk_p95 = float(np.percentile(jerk_frame, 95))
        jerk_max = float(np.max(jerk_frame))
    else:
        jerk_p95 = 0.0
        jerk_max = 0.0

    return {
        "floor_y": float(floor_y),
        "contact_ratio": float(clean_contacts.mean()) if clean_contacts is not None else 0.0,
        "foot_skate_mean_mpf": float(np.mean(skate_values)) if skate_values.size else 0.0,
        "foot_skate_p95_mpf": float(np.percentile(skate_values, 95)) if skate_values.size else 0.0,
        "foot_skate_max_mpf": float(np.max(skate_values)) if skate_values.size else 0.0,
        "foot_penetration_min_m": float(np.min(foot_y - floor_y)),
        "root_y_range_m": float(np.max(motion[:, ROOT_Y_IDX]) - np.min(motion[:, ROOT_Y_IDX])),
        "mean_joint_jerk_p95": jerk_p95,
        "mean_joint_jerk_max": jerk_max,
    }


def accepted_from_audit(audit: Dict[str, float], cfg: V42Config) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if audit["foot_penetration_min_m"] < cfg.min_accepted_penetration:
        reasons.append("floor_penetration")
    if audit["foot_skate_p95_mpf"] > cfg.max_accepted_skate_p95:
        reasons.append("foot_skate")
    return len(reasons) == 0, reasons


def optimize_motion(motion: np.ndarray, cfg: V42Config) -> Tuple[np.ndarray, Dict[str, object], Dict[str, np.ndarray]]:
    input_motion = np.asarray(motion, dtype=np.float32)
    out = input_motion.copy()

    joints0 = fk_24_from_edge151(input_motion)
    foot0 = joints0[:, list(cfg.foot_joint_ids), :].astype(np.float32)
    confidence, clean_contacts, floor_y, contact_meta = derive_contact_confidence(foot0, cfg)
    targets, target_meta = generate_foot_ik_targets(foot0, clean_contacts, cfg)

    pre = audit_motion(input_motion, cfg, clean_contacts=clean_contacts, floor_y=floor_y)

    corr_xz, root_xz_meta = compute_root_xz_countermotion(foot0, targets, clean_contacts, confidence, cfg)
    out[:, ROOT_X_IDX] += corr_xz[:, 0]
    out[:, ROOT_Z_IDX] += corr_xz[:, 1]

    root_y_new, root_y_meta = apply_root_y_weight(out[:, ROOT_Y_IDX].copy(), clean_contacts, cfg)
    out[:, ROOT_Y_IDX] = root_y_new

    out, clearance_meta = apply_floor_clearance_guard(out, cfg.foot_joint_ids, cfg)

    post = audit_motion(out, cfg, clean_contacts=clean_contacts, floor_y=floor_y)

    root_delta = out[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - input_motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
    root_xz_delta = np.linalg.norm(root_delta[:, [0, 2]], axis=1)
    root_y_delta = np.abs(root_delta[:, 1])

    rollback_reasons: List[str] = []
    if cfg.rollback_if_worse:
        if post["foot_skate_p95_mpf"] > max(pre["foot_skate_p95_mpf"] * cfg.max_skate_ratio, pre["foot_skate_p95_mpf"] + 1e-4):
            rollback_reasons.append("foot_skate_worse")
        if post["mean_joint_jerk_p95"] > max(pre["mean_joint_jerk_p95"] * cfg.max_jerk_ratio, pre["mean_joint_jerk_p95"] + 1e-4):
            rollback_reasons.append("jerk_worse")
        if root_xz_delta.max() > cfg.max_accepted_root_xz_delta:
            rollback_reasons.append("root_xz_delta_too_large")
        if root_y_delta.max() > cfg.max_accepted_root_y_delta:
            rollback_reasons.append("root_y_delta_too_large")

    rolled_back = bool(rollback_reasons)
    if rolled_back:
        final_motion = input_motion.copy()
        final_post = audit_motion(final_motion, cfg, clean_contacts=clean_contacts, floor_y=floor_y)
        final_root_delta = final_motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]] - input_motion[:, [ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX]]
        root_xz_delta = np.linalg.norm(final_root_delta[:, [0, 2]], axis=1)
        root_y_delta = np.abs(final_root_delta[:, 1])
    else:
        final_motion = out
        final_post = post

    accepted, reject_reasons = accepted_from_audit(final_post, cfg)

    report: Dict[str, object] = {
        "version": "v42_2_final_root_footplant_target_generator",
        "raw_column_mode": False,
        "fk_source": "FK(root[4:7] + rot6d[7:151] -> joints[7,8,10,11])",
        "fk_tree_source": FK_TREE_SOURCE,
        "root_indices": {"x": ROOT_X_IDX, "y": ROOT_Y_IDX, "z": ROOT_Z_IDX},
        "foot_joint_ids": list(map(int, cfg.foot_joint_ids)),
        "config": asdict(cfg),
        "contact_detector": contact_meta,
        "foot_ik_target_generator": target_meta,
        "root_xz_footplant": root_xz_meta,
        "root_y_weight": root_y_meta,
        "floor_clearance": clearance_meta,
        "pre_audit": pre,
        "candidate_post_audit_before_rollback": post,
        "post_audit": final_post,
        "planner_feedback": {"accepted": accepted, "reject_reasons": reject_reasons},
        "rollback": {"enabled": bool(cfg.rollback_if_worse), "triggered": rolled_back, "reasons": rollback_reasons},
        "root_xz_delta_mean": float(root_xz_delta.mean()) if root_xz_delta.size else 0.0,
        "root_xz_delta_p95": float(np.percentile(root_xz_delta, 95)) if root_xz_delta.size else 0.0,
        "root_xz_delta_max": float(root_xz_delta.max()) if root_xz_delta.size else 0.0,
        "root_y_delta_mean": float(root_y_delta.mean()) if root_y_delta.size else 0.0,
        "root_y_delta_p95": float(np.percentile(root_y_delta, 95)) if root_y_delta.size else 0.0,
        "root_y_delta_max": float(root_y_delta.max()) if root_y_delta.size else 0.0,
    }

    arrays = {
        "foot_ik_targets": targets.astype(np.float32),
        "foot_global_pos_native": foot0.astype(np.float32),
        "clean_contacts": clean_contacts.astype(np.uint8),
        "contact_confidence": confidence.astype(np.float32),
        "root_xz_correction": corr_xz.astype(np.float32),
        "root_y_before": input_motion[:, ROOT_Y_IDX].astype(np.float32),
        "root_y_after_candidate": out[:, ROOT_Y_IDX].astype(np.float32),
        "root_y_after_final": final_motion[:, ROOT_Y_IDX].astype(np.float32),
    }
    return final_motion.astype(np.float32), report, arrays


def render_if_requested(args: argparse.Namespace, motion_path: str) -> None:
    if not args.render_output:
        return
    if not args.audio:
        print("[V42.2 WARN] --render_output set but --audio missing; skip render", file=sys.stderr)
        return
    audio = Path(args.audio)
    if not audio.exists():
        print(f"[V42.2 WARN] audio does not exist: {audio}; skip render", file=sys.stderr)
        return
    render_script = Path(args.render_script or "render_from_npy.py")
    if not render_script.exists():
        print(f"[V42.2 WARN] render script missing: {render_script}; skip render", file=sys.stderr)
        return
    cmd = [
        sys.executable,
        str(render_script),
        "--motion",
        motion_path,
        "--audio",
        str(audio),
        "--output",
        args.render_output,
        "--camera_mode",
        args.camera_mode,
        "--render_smooth_window",
        str(args.render_smooth_window),
    ]
    print("[V42.2 RENDER]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V42.2 FINAL EDGE Root-Footplant Physics Optimizer")
    p.add_argument("--input", required=True, help="Input EDGE motion .npy")
    p.add_argument("--output", required=True, help="Output corrected EDGE motion .npy")
    p.add_argument("--json", required=True, help="Output V42.2 audit json")
    p.add_argument("--targets", required=True, help="Output V42.2 target arrays .npz")
    p.add_argument("--config", default=None, help="Optional JSON config")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--root_xz_strength", type=float, default=None)
    p.add_argument("--root_y_strength", type=float, default=None)
    p.add_argument("--contact_high", type=float, default=None)
    p.add_argument("--contact_low", type=float, default=None)
    p.add_argument("--height_margin", type=float, default=None)
    p.add_argument("--speed_gate_mpf", type=float, default=None)
    p.add_argument("--max_correction", type=float, default=None)
    p.add_argument("--rollback_if_worse", type=int, default=None, help="1/0")
    p.add_argument("--audio", default=None)
    p.add_argument("--render_output", default=None)
    p.add_argument("--render_script", default="render_from_npy.py")
    p.add_argument("--camera_mode", default="follow", choices=["fixed", "follow"])
    p.add_argument("--render_smooth_window", type=int, default=5)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    cfg = V42Config.from_json(args.config)
    override_map = {
        "fps": args.fps,
        "root_xz_strength": args.root_xz_strength,
        "root_y_strength": args.root_y_strength,
        "contact_high": args.contact_high,
        "contact_low": args.contact_low,
        "height_margin": args.height_margin,
        "speed_gate_mpf": args.speed_gate_mpf,
        "root_xz_max_correction": args.max_correction,
    }
    for k, v in override_map.items():
        if v is not None:
            setattr(cfg, k, v)
    if args.rollback_if_worse is not None:
        cfg.rollback_if_worse = bool(args.rollback_if_worse)

    motion = np.load(args.input)
    final_motion, report, arrays = optimize_motion(motion, cfg)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.targets).parent.mkdir(parents=True, exist_ok=True)

    np.save(args.output, final_motion.astype(np.float32))
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    np.savez_compressed(args.targets, **arrays)

    print(json.dumps({
        "version": report["version"],
        "input": args.input,
        "output": args.output,
        "json": args.json,
        "targets": args.targets,
        "accepted": report["planner_feedback"]["accepted"],
        "reject": report["planner_feedback"]["reject_reasons"],
        "pre_skate_p95": report["pre_audit"]["foot_skate_p95_mpf"],
        "post_skate_p95": report["post_audit"]["foot_skate_p95_mpf"],
        "post_foot_pen": report["post_audit"]["foot_penetration_min_m"],
        "rollback": report["rollback"],
        "no_zero_initialization": report["foot_ik_target_generator"]["no_zero_initialization"],
        "anchor_sampling_contact_internal_only": report["foot_ik_target_generator"]["anchor_sampling_contact_internal_only"],
        "root_indices": report["root_indices"],
        "raw_column_mode": report["raw_column_mode"],
    }, ensure_ascii=False, indent=2))

    render_if_requested(args, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
