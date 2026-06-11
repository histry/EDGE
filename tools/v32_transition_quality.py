#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context-aware transition risk and acceptance gate for EDGE V32."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from pytorch3d.transforms import matrix_to_axis_angle

from tools.v29_motion_geometry import (
    CONTACT,
    NUM_JOINTS,
    angular_velocity_np,
    motion_rotation_matrices_np,
    motion_to_joint_positions_np,
)

FOOT_JOINTS = (7, 8, 10, 11)


def _rms(values: np.ndarray) -> float:
    x = np.asarray(values, np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def _ratio(value: float, reference: float, floor: float) -> float:
    return float(value / max(reference, floor))


def _rotation_step(motion: np.ndarray) -> np.ndarray:
    matrix = motion_rotation_matrices_np(motion)
    if len(matrix) < 2:
        return np.zeros((0, NUM_JOINTS), np.float32)
    with torch.no_grad():
        first = torch.from_numpy(matrix[:-1])
        second = torch.from_numpy(matrix[1:])
        relative = torch.matmul(first.transpose(-1, -2), second)
        angle = torch.linalg.norm(
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
            key: 1e9 for key in (
                "total", "entry_velocity", "exit_velocity",
                "boundary_acceleration", "joint_jerk",
                "angular_jerk", "foot_slip", "foot_penetration",
                "contact_switch", "max_rotation_step_rad",
                "high_frequency",
            )
        }

    prev_context = prev[-4:] if len(prev) >= 4 else prev
    next_context = nxt[:4] if len(nxt) >= 4 else nxt
    context = np.concatenate([prev_context, trans, next_context], axis=0)
    positions = motion_to_joint_positions_np(context)
    velocity = np.diff(positions, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps

    left_frames = len(prev_context)
    entry_index = max(left_frames - 1, 0)
    exit_index = min(left_frames + len(trans) - 1, len(velocity) - 1)

    entry_velocity = _rms(
        velocity[entry_index]
        - velocity[max(entry_index - 1, 0)]
    )
    exit_velocity = _rms(
        velocity[min(exit_index + 1, len(velocity) - 1)]
        - velocity[exit_index]
    )

    if len(acceleration):
        a_entry = min(max(entry_index - 1, 0), len(acceleration) - 1)
        a_exit = min(max(exit_index - 1, 0), len(acceleration) - 1)
        boundary_acceleration = _rms(np.concatenate([
            acceleration[a_entry].reshape(-1),
            acceleration[a_exit].reshape(-1),
        ]))
    else:
        boundary_acceleration = 0.0

    transition_positions = motion_to_joint_positions_np(trans)
    tv = np.diff(transition_positions, axis=0) * fps
    ta = np.diff(tv, axis=0) * fps
    tj = np.diff(ta, axis=0) * fps
    joint_jerk = _rms(tj)
    high_frequency = _rms(np.diff(tj, axis=0)) if len(tj) > 1 else joint_jerk

    angular = angular_velocity_np(trans) * fps
    angular_acc = np.diff(angular, axis=0) * fps
    angular_jerk_values = np.diff(angular_acc, axis=0) * fps
    angular_jerk = _rms(angular_jerk_values)

    feet = transition_positions[:, FOOT_JOINTS]
    foot_velocity = np.diff(feet, axis=0, prepend=feet[:1]) * fps
    horizontal_speed = np.linalg.norm(
        foot_velocity[..., (0, 2)], axis=-1
    )
    contact_prob = np.asarray(
        trans[:, CONTACT], np.float32
    ).clip(0.0, 1.0)
    foot_slip = float(
        np.sum(horizontal_speed * contact_prob)
        / max(float(contact_prob.sum()), 1e-6)
    )

    context_positions = motion_to_joint_positions_np(
        np.concatenate([prev_context[-1:], trans, next_context[:1]], axis=0)
    )
    context_feet_y = context_positions[..., FOOT_JOINTS, 1]
    ground = float(np.percentile(context_feet_y, 5))
    penetration = np.maximum(ground - feet[..., 1] - 0.008, 0.0)
    foot_penetration = float(np.mean(penetration**2))

    contact_switch = (
        float(np.abs(np.diff(contact_prob, axis=0)).mean())
        if len(contact_prob) > 1 else 0.0
    )
    rotation_step = _rotation_step(trans)
    max_rotation_step = (
        float(np.max(rotation_step)) if rotation_step.size else 0.0
    )

    total = (
        1.20 * entry_velocity
        + 1.40 * exit_velocity
        + 0.15 * boundary_acceleration
        + 0.08 * joint_jerk
        + 0.04 * angular_jerk
        + 2.00 * foot_slip
        + 6.00 * foot_penetration
        + 0.25 * contact_switch
        + 2.00 * max_rotation_step
        + 0.05 * high_frequency
    )
    return {
        "total": float(total),
        "entry_velocity": float(entry_velocity),
        "exit_velocity": float(exit_velocity),
        "boundary_acceleration": float(boundary_acceleration),
        "joint_jerk": float(joint_jerk),
        "angular_jerk": float(angular_jerk),
        "foot_slip": float(foot_slip),
        "foot_penetration": float(foot_penetration),
        "contact_switch": float(contact_switch),
        "max_rotation_step_rad": float(max_rotation_step),
        "high_frequency": float(high_frequency),
    }


def accept_candidate(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    max_total_ratio: float = 1.02,
    max_entry_ratio: float = 1.05,
    max_exit_ratio: float = 1.03,
    max_jerk_ratio: float = 1.03,
    max_foot_ratio: float = 1.02,
    max_penetration_ratio: float = 1.02,
    max_rotation_step_rad: float = 0.20,
) -> Tuple[bool, Dict[str, object]]:
    finite = all(np.isfinite(float(v)) for v in candidate.values())
    ratios = {
        "total": _ratio(candidate["total"], baseline["total"], 1e-5),
        "entry_velocity": _ratio(
            candidate["entry_velocity"],
            baseline["entry_velocity"],
            1e-4,
        ),
        "exit_velocity": _ratio(
            candidate["exit_velocity"],
            baseline["exit_velocity"],
            1e-4,
        ),
        "joint_jerk": _ratio(
            candidate["joint_jerk"], baseline["joint_jerk"], 1e-3
        ),
        "angular_jerk": _ratio(
            candidate["angular_jerk"], baseline["angular_jerk"], 1e-3
        ),
        "foot_slip": _ratio(
            candidate["foot_slip"], baseline["foot_slip"], 1e-4
        ),
        "foot_penetration": _ratio(
            candidate["foot_penetration"],
            baseline["foot_penetration"],
            1e-7,
        ),
    }
    checks = {
        "finite": finite,
        "total": ratios["total"] <= max_total_ratio,
        "entry": ratios["entry_velocity"] <= max_entry_ratio,
        "exit": ratios["exit_velocity"] <= max_exit_ratio,
        "joint_jerk": ratios["joint_jerk"] <= max_jerk_ratio,
        "angular_jerk": ratios["angular_jerk"] <= max_jerk_ratio,
        "foot_slip": ratios["foot_slip"] <= max_foot_ratio,
        "penetration": (
            ratios["foot_penetration"] <= max_penetration_ratio
        ),
        "rotation_step": (
            candidate["max_rotation_step_rad"]
            <= max_rotation_step_rad
        ),
    }
    accepted = bool(all(checks.values()))
    return accepted, {
        "accepted": accepted,
        "checks": checks,
        "ratios": ratios,
    }
