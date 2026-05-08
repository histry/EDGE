"""Formal planner patch for V10/V11 ChoreoRAG.

Adds nonlinear transition penalty and an adaptive-keyframe scaffold.

Compared with the original linear ``transition_cost`` in
``v10_choreo_planner.py``, this patch adds a thresholded nonlinear penalty for
abrupt unit-to-unit transitions.  It is designed for the Activity-vs-Jerk
conflict in ChoreoRAG: expressive units remain selectable, but extreme
pose/velocity discontinuities become very costly.

V11 additions
-------------
The helper ``adaptive_transition_cost`` supports frame-aware dynamic weighting:
near a user keyframe, pose compatibility can be emphasized and jerk can be
relaxed; far from keyframes, jerk penalty returns to normal.  The current
planner calls ``transition_cost(unit_a, unit_b, weights)`` without frame
metadata, so the active default remains global.  If future planner code passes
``weights["target_frame"]`` or calls ``adaptive_transition_cost(...,
target_frame=...)``, the adaptive behavior becomes active immediately.

Environment
-----------
    EDGE_V10_JERK_PENALTY=1
    EDGE_V10_JERK_MODE=exp|quadratic|both
    EDGE_V10_JERK_THRESHOLD=0.18
    EDGE_V10_JERK_PENALTY_WEIGHT=0.35
    EDGE_V10_JERK_PENALTY_SCALE=8.0
    EDGE_V10_POSE_SAFE_THRESHOLD=0.40
    EDGE_V10_POSE_QUAD_WEIGHT=5.0

Adaptive scaffold:
    EDGE_V10_ADAPTIVE_PLANNER=1
    EDGE_V10_ADAPTIVE_KEYFRAME_RADIUS=8
    EDGE_V10_ADAPTIVE_NEAR_JERK_SCALE=0.35
    EDGE_V10_ADAPTIVE_NEAR_POSE_SCALE=2.0
    EDGE_V10_MID_FRAMES=50,100
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Any, Optional

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


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def _as_unit_t151(unit: Any) -> np.ndarray:
    arr = np.asarray(unit, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"unit must be [T,151] or [151,T], got {arr.shape}")
    if arr.shape[-1] == 151:
        return arr
    if arr.shape[0] == 151:
        return arr.T.astype(np.float32)
    raise ValueError(f"unit must be [T,151] or [151,T], got {arr.shape}")


def _parse_mid_frames() -> list[int]:
    text = os.environ.get("EDGE_V10_MID_FRAMES", "")
    if not text:
        return [50, 100]
    out = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(round(float(item))))
        except Exception:
            pass
    return out or [50, 100]


def _near_keyframe_factor(target_frame: Optional[int]) -> float:
    """Return 1 near a keyframe and 0 far away."""
    if target_frame is None:
        return 0.0
    radius = max(1, _env_int("EDGE_V10_ADAPTIVE_KEYFRAME_RADIUS", 8))
    mids = _parse_mid_frames()
    dist = min(abs(int(target_frame) - int(f)) for f in mids)
    return float(np.clip(1.0 - dist / float(radius), 0.0, 1.0))


def estimated_transition_jerk(unit_a: Any, unit_b: Any) -> float:
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

    contact_jump = float(
        np.mean(
            np.abs(
                (a[-1, CONTACT_SLICE] > 0.5).astype(np.float32)
                - (b[0, CONTACT_SLICE] > 0.5).astype(np.float32)
            )
        )
    )

    ra = a[-1, ROOT_XZ_IDX] - a[-2, ROOT_XZ_IDX]
    rb = b[1, ROOT_XZ_IDX] - b[0, ROOT_XZ_IDX]
    na = float(np.linalg.norm(ra))
    nb = float(np.linalg.norm(rb))
    if na > 1e-8 and nb > 1e-8:
        root_dir = float(1.0 - np.clip(np.dot(ra, rb) / (na * nb + 1e-8), -1.0, 1.0))
    else:
        root_dir = 0.0

    return float(pose_jump + 0.50 * vel_mismatch + 0.10 * contact_jump + 0.05 * root_dir)


def pose_boundary_jump(unit_a: Any, unit_b: Any) -> float:
    a = _as_unit_t151(unit_a)
    b = _as_unit_t151(unit_b)
    if len(a) < 1 or len(b) < 1:
        return 0.0
    return float(np.sqrt(np.mean((a[-1, ROT_SLICE] - b[0, ROT_SLICE]) ** 2)))


def threshold_penalty(x: float, threshold: float, scale: float, mode: str = "exp") -> float:
    x = float(x)
    if x <= float(threshold):
        return 0.0
    excess = x - float(threshold)
    mode = str(mode or "exp").lower()
    if mode in {"quad", "quadratic", "l2"}:
        return float((scale * excess) ** 2)
    if mode in {"both", "hybrid"}:
        z = min(20.0, float(scale) * excess)
        return float((scale * excess) ** 2 + np.exp(z) - 1.0)
    z = min(20.0, float(scale) * excess)
    return float(np.exp(z) - 1.0)


def adaptive_transition_cost(unit_a, unit_b, weights, base_cost: float, target_frame: Optional[int] = None) -> float:
    mode = _env_str("EDGE_V10_JERK_MODE", "exp").lower()
    jerk = estimated_transition_jerk(unit_a, unit_b)
    pose_jump = pose_boundary_jump(unit_a, unit_b)

    near = _near_keyframe_factor(target_frame) if _env_bool("EDGE_V10_ADAPTIVE_PLANNER", False) else 0.0
    jerk_scale_near = _env_float("EDGE_V10_ADAPTIVE_NEAR_JERK_SCALE", 0.35)
    pose_scale_near = _env_float("EDGE_V10_ADAPTIVE_NEAR_POSE_SCALE", 2.0)

    jerk_weight = _env_float("EDGE_V10_JERK_PENALTY_WEIGHT", 0.35)
    pose_quad_weight = _env_float("EDGE_V10_POSE_QUAD_WEIGHT", 0.0)

    # Near keyframes: relax jerk slightly so the selected pose is not erased;
    # amplify pose penalty so the planner still respects keyframe compatibility.
    jerk_weight = jerk_weight * ((1.0 - near) + near * jerk_scale_near)
    pose_quad_weight = pose_quad_weight * ((1.0 - near) + near * pose_scale_near)

    penalty = jerk_weight * threshold_penalty(
        jerk,
        threshold=_env_float("EDGE_V10_JERK_THRESHOLD", 0.18),
        scale=_env_float("EDGE_V10_JERK_PENALTY_SCALE", 8.0),
        mode=mode,
    )

    pose_safe_threshold = _env_float("EDGE_V10_POSE_SAFE_THRESHOLD", 0.40)
    pose_penalty = 0.0
    if pose_quad_weight > 0 and pose_jump > pose_safe_threshold:
        pose_penalty = pose_quad_weight * float((pose_jump - pose_safe_threshold) ** 2)
        penalty += pose_penalty

    if _env_bool("EDGE_V10_JERK_VERBOSE", False):
        print(
            "   V10/V11 transition penalty: "
            f"base={base_cost:.6f}, frame={target_frame}, near={near:.3f}, "
            f"jerk={jerk:.6f}, pose_jump={pose_jump:.6f}, "
            f"penalty={penalty:.6f}, pose_penalty={pose_penalty:.6f}"
        )
    return float(base_cost + penalty)


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
        target_frame = None
        try:
            if isinstance(weights, dict):
                if "target_frame" in weights:
                    target_frame = int(weights["target_frame"])
                elif "frame" in weights:
                    target_frame = int(weights["frame"])
        except Exception:
            target_frame = None
        return adaptive_transition_cost(unit_a, unit_b, weights, base_cost=base, target_frame=target_frame)

    planner.transition_cost = patched_transition_cost
    planner._edge_v10_formal_planner_patch_installed = True
    planner._edge_v10_original_transition_cost = original_transition_cost
    planner.estimated_transition_jerk = estimated_transition_jerk
    planner.pose_boundary_jump = pose_boundary_jump
    planner.adaptive_transition_cost = adaptive_transition_cost

    if verbose:
        print(
            "✅ Installed V10/V11 formal planner patch: nonlinear/adaptive transition penalty "
            f"mode={_env_str('EDGE_V10_JERK_MODE', 'exp')}, "
            f"threshold={_env_float('EDGE_V10_JERK_THRESHOLD', 0.18)}, "
            f"weight={_env_float('EDGE_V10_JERK_PENALTY_WEIGHT', 0.35)}, "
            f"scale={_env_float('EDGE_V10_JERK_PENALTY_SCALE', 8.0)}, "
            f"adaptive={_env_bool('EDGE_V10_ADAPTIVE_PLANNER', False)}"
        )
    return True


def install():
    return install_v10_choreo_planner_formal_patch(verbose=True)
