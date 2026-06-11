#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Risk metrics and conservative acceptance gate for V31 transitions."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from tools.v29_motion_geometry import (
    CONTACT,
    NUM_JOINTS,
    angular_velocity_np,
    motion_to_joint_positions_np,
    motion_rotation_matrices_np,
)
from pytorch3d.transforms import matrix_to_axis_angle
import torch

FOOT_JOINTS = (7, 8, 10, 11)


def _rms(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _safe_ratio(value: float, reference: float, floor: float) -> float:
    return float(value / max(reference, floor))


def _rotation_step(motion: np.ndarray) -> np.ndarray:
    matrices = motion_rotation_matrices_np(motion)
    if len(matrices) < 2:
        return np.zeros((0, NUM_JOINTS), np.float32)
    with torch.no_grad():
        first = torch.from_numpy(matrices[:-1])
        second = torch.from_numpy(matrices[1:])
        relative = torch.matmul(first.transpose(-1, -2), second)
        angle = torch.linalg.vector_norm(
            matrix_to_axis_angle(relative), dim=-1
        )
    return angle.cpu().numpy().astype(np.float32)


def transition_risk(
    previous: np.ndarray,
    transition: np.ndarray,
    following: np.ndarray,
    fps: float = 30.0,
) -> Dict[str, float]:
    prev = np.asarray(previous, np.float32)
    trans = np.asarray(transition, np.float32)
    nxt = np.asarray(following, np.float32)
    if len(trans) == 0:
        return {
            "total": 1e9,
            "boundary_velocity": 1e9,
            "boundary_acceleration": 1e9,
            "joint_jerk": 1e9,
            "angular_jerk": 1e9,
            "foot_slip": 1e9,
            "max_rotation_step_rad": 1e9,
        }
    context = np.concatenate([
        prev[-3:] if len(prev) >= 3 else prev,
        trans,
        nxt[:3] if len(nxt) >= 3 else nxt,
    ], axis=0)
    positions = motion_to_joint_positions_np(context)
    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps

    left = max(len(prev[-3:] if len(prev) >= 3 else prev) - 1, 0)
    right = min(left + len(trans), len(velocity) - 1)
    boundary_velocity = _rms(
        np.concatenate([
            (velocity[left] - velocity[max(left - 1, 0)]).reshape(-1),
            (velocity[min(right, len(velocity)-1)]
             - velocity[max(right - 1, 0)]).reshape(-1),
        ])
    )
    if len(acceleration):
        a_left = min(max(left - 1, 0), len(acceleration) - 1)
        a_right = min(max(right - 1, 0), len(acceleration) - 1)
        boundary_acceleration = _rms(
            np.concatenate([
                acceleration[a_left].reshape(-1),
                acceleration[a_right].reshape(-1),
            ])
        )
    else:
        boundary_acceleration = 0.0

    transition_positions = motion_to_joint_positions_np(trans)
    tv = np.diff(transition_positions, axis=0) * fps
    ta = np.diff(tv, axis=0) * fps
    tj = np.diff(ta, axis=0) * fps
    joint_jerk = _rms(tj)

    angular = angular_velocity_np(trans) * fps
    angular_acc = np.diff(angular, axis=0) * fps
    angular_jerk_values = np.diff(angular_acc, axis=0) * fps
    angular_jerk = _rms(angular_jerk_values)

    feet = transition_positions[:, FOOT_JOINTS]
    foot_velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    foot_speed = np.linalg.norm(foot_velocity, axis=-1)
    contacts = np.asarray(trans[:, CONTACT] > 0.5)
    foot_slip = (
        float(np.mean(foot_speed[contacts]))
        if np.any(contacts) else 0.0
    )

    rotation_step = _rotation_step(trans)
    max_rotation_step = (
        float(np.max(rotation_step)) if rotation_step.size else 0.0
    )
    high_frequency = _rms(np.diff(tj, axis=0)) if len(tj) > 1 else joint_jerk

    # The score is used only for within-boundary candidate ranking.  Reported
    # components remain available separately and should be used in papers.
    total = (
        1.00 * boundary_velocity
        + 0.15 * boundary_acceleration
        + 0.08 * joint_jerk
        + 0.04 * angular_jerk
        + 1.50 * foot_slip
        + 2.00 * max_rotation_step
        + 0.05 * high_frequency
    )
    return {
        "total": float(total),
        "boundary_velocity": float(boundary_velocity),
        "boundary_acceleration": float(boundary_acceleration),
        "joint_jerk": float(joint_jerk),
        "angular_jerk": float(angular_jerk),
        "foot_slip": float(foot_slip),
        "max_rotation_step_rad": float(max_rotation_step),
        "high_frequency": float(high_frequency),
    }


def accept_candidate(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    max_total_ratio: float = 1.02,
    max_boundary_ratio: float = 1.04,
    max_jerk_ratio: float = 1.03,
    max_foot_ratio: float = 1.05,
    max_rotation_step_rad: float = 0.22,
) -> Tuple[bool, Dict[str, object]]:
    finite = all(np.isfinite(float(value)) for value in candidate.values())
    total_ratio = _safe_ratio(candidate["total"], baseline["total"], 1e-5)
    boundary_ratio = _safe_ratio(
        candidate["boundary_velocity"],
        baseline["boundary_velocity"],
        1e-4,
    )
    jerk_ratio = _safe_ratio(
        candidate["joint_jerk"], baseline["joint_jerk"], 1e-3
    )
    angular_ratio = _safe_ratio(
        candidate["angular_jerk"], baseline["angular_jerk"], 1e-3
    )
    foot_ratio = _safe_ratio(
        candidate["foot_slip"], baseline["foot_slip"], 1e-4
    )
    checks = {
        "finite": finite,
        "total_ratio_ok": total_ratio <= max_total_ratio,
        "boundary_ratio_ok": boundary_ratio <= max_boundary_ratio,
        "joint_jerk_ratio_ok": jerk_ratio <= max_jerk_ratio,
        "angular_jerk_ratio_ok": angular_ratio <= max_jerk_ratio,
        "foot_ratio_ok": foot_ratio <= max_foot_ratio,
        "rotation_step_ok": (
            candidate["max_rotation_step_rad"] <= max_rotation_step_rad
        ),
    }
    accepted = bool(all(checks.values()))
    return accepted, {
        "accepted": accepted,
        "checks": checks,
        "ratios": {
            "total": total_ratio,
            "boundary_velocity": boundary_ratio,
            "joint_jerk": jerk_ratio,
            "angular_jerk": angular_ratio,
            "foot_slip": foot_ratio,
        },
    }
