#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boundary compatibility gate for V34 Event-RAG retrieval.

V34 strict warp guarantees that an event can fit a music slot.  It does not
guarantee that the previous event and the next event can be joined without a
large visual jump.  This module adds a retrieval-time edge gate so obviously
incompatible event pairs are rejected before Contact-INR is asked to repair
them.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping

import numpy as np


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _array_value(hierarchy: Mapping[str, Any] | None, name: str, index: int, default: float = 0.0) -> float:
    if not hierarchy or name not in hierarchy:
        return float(default)
    arr = np.asarray(hierarchy[name])
    if arr.ndim == 0 or int(index) < 0 or int(index) >= len(arr):
        return float(default)
    return float(arr[int(index)])


def _soft_ratio(value: float, limit: float) -> float:
    return float(max(0.0, value / max(limit, 1e-8) - 1.0))


def evaluate_boundary_compatibility(
    *,
    previous_index: int,
    candidate_index: int,
    candidate_boundary: Mapping[str, float],
    transition_cost: float,
    phrase: Any,
    args: Any,
    hierarchy: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a retrieval-time compatibility score and hard-reject decision.

    The thresholds are deliberately stricter than Contact-INR's post-generation
    safety gate because this runs before synthesis.  A pair that exceeds these
    coarse endpoint limits is unlikely to be repaired by a short transition.
    Strong musical boundaries are allowed slightly larger semantic/body resets,
    but not unbounded pose or contact discontinuities.
    """
    pose = float(candidate_boundary.get("pose_jump", 0.0))
    velocity = float(candidate_boundary.get("velocity_jump", 0.0))
    acceleration = float(candidate_boundary.get("acceleration_jump", 0.0))
    contact = float(candidate_boundary.get("contact_jump", 0.0))
    yaw = float(candidate_boundary.get("yaw_gap_deg", 0.0))
    transition = float(transition_cost)

    boundary_strength = float(getattr(phrase, "boundary_accent_strength", 0.0))
    music_event = str(getattr(phrase, "music_event", "neutral_flow"))
    tension = float(getattr(phrase, "tension", 0.0))
    calm = float(getattr(phrase, "calmness", 0.0))
    reset_allow = float(np.clip(
        0.12 + 0.55 * boundary_strength + 0.20 * tension - 0.12 * calm
        + (0.18 if music_event == "section_change" else 0.0),
        0.0,
        0.85,
    ))

    # Defaults are conservative enough to block the observed hard jumps while
    # still letting section changes use a broader reset budget.
    pose_limit = _env_float("V34_COMPAT_MAX_POSE_JUMP", 0.42) * (1.0 + 0.25 * reset_allow)
    velocity_limit = _env_float("V34_COMPAT_MAX_VELOCITY_JUMP", 0.060) * (1.0 + 0.25 * reset_allow)
    acceleration_limit = _env_float("V34_COMPAT_MAX_ACCELERATION_JUMP", 0.120) * (1.0 + 0.25 * reset_allow)
    contact_limit = _env_float("V34_COMPAT_MAX_CONTACT_JUMP", 0.62) + 0.12 * reset_allow
    yaw_limit = _env_float("V34_COMPAT_MAX_YAW_GAP_DEG", 62.0) + 28.0 * reset_allow
    transition_limit = _env_float("V34_COMPAT_MAX_TRANSITION_COST", 0.95) * (1.0 + 0.35 * reset_allow)

    prev_body = _array_value(hierarchy, "body_code", previous_index, 2.0)
    next_body = _array_value(hierarchy, "body_code", candidate_index, 2.0)
    prev_activity = _array_value(hierarchy, "activity01", previous_index, 0.5)
    next_activity = _array_value(hierarchy, "activity01", candidate_index, 0.5)
    prev_turn = _array_value(hierarchy, "turn01", previous_index, 0.0)
    next_turn = _array_value(hierarchy, "turn01", candidate_index, 0.0)

    body_jump = abs(next_body - prev_body) / 5.0
    activity_jump = abs(next_activity - prev_activity)
    turn_jump = abs(next_turn - prev_turn)
    body_allow = 0.30 + 0.55 * reset_allow
    activity_allow = 0.22 + 0.45 * reset_allow
    turn_allow = 0.25 + 0.45 * reset_allow

    terms = {
        "pose": _soft_ratio(pose, pose_limit),
        "velocity": _soft_ratio(velocity, velocity_limit),
        "acceleration": _soft_ratio(acceleration, acceleration_limit),
        "contact": _soft_ratio(contact, contact_limit),
        "yaw": _soft_ratio(yaw, yaw_limit),
        "transition": _soft_ratio(transition, transition_limit),
        "body": _soft_ratio(body_jump, body_allow),
        "activity": _soft_ratio(activity_jump, activity_allow),
        "turn": _soft_ratio(turn_jump, turn_allow),
    }
    score = (
        1.25 * terms["pose"]
        + 1.10 * terms["velocity"]
        + 1.05 * terms["acceleration"]
        + 0.95 * terms["yaw"]
        + 0.80 * terms["transition"]
        + 0.70 * terms["contact"]
        + 0.45 * terms["body"]
        + 0.35 * terms["activity"]
        + 0.25 * terms["turn"]
    )

    hard_checks = {
        "pose": pose <= pose_limit,
        "velocity": velocity <= velocity_limit,
        "acceleration": acceleration <= acceleration_limit,
        "contact": contact <= contact_limit,
        "yaw": yaw <= yaw_limit,
        "transition": transition <= transition_limit,
    }
    if _enabled("V34_COMPAT_SEMANTIC_HARD_PRUNE", "0"):
        hard_checks.update({
            "body": body_jump <= body_allow,
            "activity": activity_jump <= activity_allow,
            "turn": turn_jump <= turn_allow,
        })
    hard_reject = not all(bool(x) for x in hard_checks.values())

    return {
        "enabled": True,
        "score": float(score),
        "hard_reject": bool(hard_reject),
        "checks": hard_checks,
        "terms": {key: float(value) for key, value in terms.items()},
        "metrics": {
            "pose_jump": pose,
            "velocity_jump": velocity,
            "acceleration_jump": acceleration,
            "contact_jump": contact,
            "yaw_gap_deg": yaw,
            "transition_cost": transition,
            "body_jump": float(body_jump),
            "activity_jump": float(activity_jump),
            "turn_jump": float(turn_jump),
        },
        "limits": {
            "pose_jump": float(pose_limit),
            "velocity_jump": float(velocity_limit),
            "acceleration_jump": float(acceleration_limit),
            "contact_jump": float(contact_limit),
            "yaw_gap_deg": float(yaw_limit),
            "transition_cost": float(transition_limit),
            "body_jump": float(body_allow),
            "activity_jump": float(activity_allow),
            "turn_jump": float(turn_allow),
        },
        "context": {
            "reset_allow": float(reset_allow),
            "boundary_accent_strength": float(boundary_strength),
            "music_event": music_event,
            "tension": float(tension),
            "calmness": float(calm),
        },
    }
