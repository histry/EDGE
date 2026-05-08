"""Formal planner patch for V10 ChoreoRAG.

Add this file to the EDGE repository root.  ``generate_v10_choreo.py`` in this
bundle imports it automatically when ``EDGE_V10_JERK_PENALTY=1``.

Purpose
-------
The original planner already has transition cost.  This patch adds a nonlinear
threshold penalty based on a transition-jerk proxy between adjacent retrieved
motion units.  It discourages paths where a very expressive unit creates an
abrupt unit-to-unit transition.

Environment
-----------
    EDGE_V10_JERK_PENALTY=1
    EDGE_V10_JERK_THRESHOLD=0.18
    EDGE_V10_JERK_PENALTY_WEIGHT=0.35
    EDGE_V10_JERK_PENALTY_SCALE=8.0
    EDGE_V10_JERK_VERBOSE=0
"""
from __future__ import annotations

import os
from functools import wraps

import numpy as np

ROT_SLICE = slice(7, 151)
CONTACT_SLICE = slice(0, 4)
ROOT_XZ_IDX = [4, 6]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _as_unit_t151(unit) -> np.ndarray:
    arr = np.asarray(unit, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"unit must be [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == 151:
        return arr
    if arr.shape[0] == 151:
        return arr.T.astype(np.float32)
    raise ValueError(f"unit must be [T,151] or [151,T], got {arr.shape}")


def estimated_transition_jerk(unit_a, unit_b) -> float:
    """Proxy for generated transition jerk before diffusion generation.

    We cannot know final generated jerk during planning, so we estimate it from:
    - pose jump between unit_a exit and unit_b entry;
    - velocity mismatch around the boundary;
    - contact phase discontinuity;
    - root direction discontinuity.
    """
    a = _as_unit_t151(unit_a)
    b = _as_unit_t151(unit_b)
    if len(a) < 2 or len(b) < 2:
        return 0.0

    exit_pose = a[-1, ROT_SLICE]
    prev_pose = a[-2, ROT_SLICE]
    entry_pose = b[0, ROT_SLICE]
    next_pose = b[1, ROT_SLICE]

    v_a = exit_pose - prev_pose
    v_b = next_pose - entry_pose
    pose_jump = float(np.sqrt(np.mean((entry_pose - exit_pose) ** 2)))
    vel_mismatch = float(np.sqrt(np.mean((v_b - v_a) ** 2)))

    contact_jump = float(np.mean(np.abs((a[-1, CONTACT_SLICE] > 0.5).astype(np.float32) - (b[0, CONTACT_SLICE] > 0.5).astype(np.float32))))

    ra = a[-1, ROOT_XZ_IDX] - a[-2, ROOT_XZ_IDX]
    rb = b[1, ROOT_XZ_IDX] - b[0, ROOT_XZ_IDX]
    na = float(np.linalg.norm(ra))
    nb = float(np.linalg.norm(rb))
    if na > 1e-8 and nb > 1e-8:
        root_dir = float(1.0 - np.clip(np.dot(ra, rb) / (na * nb + 1e-8), -1.0, 1.0))
    else:
        root_dir = 0.0

    return float(pose_jump + 0.50 * vel_mismatch + 0.10 * contact_jump + 0.05 * root_dir)


def threshold_penalty(x: float, threshold: float, scale: float) -> float:
    x = float(x)
    threshold = float(threshold)
    if x <= threshold:
        return 0.0
    # Clip exponent to avoid numerical explosions in pathological DB entries.
    z = min(20.0, float(scale) * (x - threshold))
    return float(np.exp(z) - 1.0)


def install_v10_choreo_planner_formal_patch(verbose: bool = True) -> bool:
    if not _env_bool("EDGE_V10_JERK_PENALTY", False):
        return True
    try:
        import v10_choreo_planner as planner
    except Exception as exc:
        if verbose:
            print(f"⚠️ V10 formal planner patch skipped: {exc}")
        return False

    if getattr(planner, "_edge_v10_formal_planner_patch_installed", False):
        return True

    original_transition_cost = getattr(planner, "transition_cost", None)
    if original_transition_cost is None:
        if verbose:
            print("⚠️ V10 formal planner patch skipped: transition_cost not found.")
        return False

    @wraps(original_transition_cost)
    def patched_transition_cost(unit_a, unit_b, weights):
        base = float(original_transition_cost(unit_a, unit_b, weights))
        if not _env_bool("EDGE_V10_JERK_PENALTY", False):
            return base
        jerk = estimated_transition_jerk(unit_a, unit_b)
        threshold = _env_float("EDGE_V10_JERK_THRESHOLD", 0.18)
        scale = _env_float("EDGE_V10_JERK_PENALTY_SCALE", 8.0)
        weight = _env_float("EDGE_V10_JERK_PENALTY_WEIGHT", 0.35)
        penalty = weight * threshold_penalty(jerk, threshold=threshold, scale=scale)
        if _env_bool("EDGE_V10_JERK_VERBOSE", False):
            print(
                "   V10 jerk penalty: "
                f"base={base:.6f}, jerk_proxy={jerk:.6f}, threshold={threshold:.3f}, penalty={penalty:.6f}"
            )
        return float(base + penalty)

    planner.transition_cost = patched_transition_cost
    planner._edge_v10_formal_planner_patch_installed = True
    planner._edge_v10_original_transition_cost = original_transition_cost
    planner.estimated_transition_jerk = estimated_transition_jerk

    if verbose:
        print(
            "✅ Installed V10 formal planner patch: nonlinear transition-jerk penalty "
            f"threshold={_env_float('EDGE_V10_JERK_THRESHOLD', 0.18)}, "
            f"weight={_env_float('EDGE_V10_JERK_PENALTY_WEIGHT', 0.35)}, "
            f"scale={_env_float('EDGE_V10_JERK_PENALTY_SCALE', 8.0)}"
        )
    return True


def install():
    return install_v10_choreo_planner_formal_patch(verbose=True)
