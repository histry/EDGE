"""Functional choreography metrics for EDGE 151D motion.

Goal:
  Evaluate whether trajectory, support, and expression happen as a coupled
  choreography event, instead of independent root translation + local motion.

Representation:
  [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotation.
  Ground-plane trajectory is root X/Z.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from footstep_phase_utils import (
    CONTACT_SLICE,
    LOWER_ROT_INDEX,
    TORSO_ROT_INDEX,
    UPPER_ROT_INDEX,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    as_t151,
    robust_norm,
)


def _rms_frame_delta(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros((len(x),), dtype=np.float32)
    out = np.zeros((len(x),), dtype=np.float32)
    out[1:] = np.sqrt(np.mean((x[1:] - x[:-1]) ** 2, axis=1))
    return out


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(len(a), len(b))
    if n < 3:
        return 0.0
    a = a[:n]
    b = b[:n]
    if float(a.std()) <= 1e-8 or float(b.std()) <= 1e-8:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip((c + 1.0) * 0.5, 0.0, 1.0))


def _window_response(signal: np.ndarray, event_frames: List[int], radius: int = 3) -> float:
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == 0:
        return 0.0
    base = float(signal.mean()) + 1e-8
    if not event_frames:
        return 0.0
    vals = []
    for f in event_frames:
        s = max(0, int(f) - radius)
        e = min(len(signal), int(f) + radius + 1)
        if e > s:
            vals.append(float(signal[s:e].mean()))
    if not vals:
        return 0.0
    # Ratio-style response. 1.0 means event-window energy is about 2x baseline.
    return float(np.clip((float(np.mean(vals)) / base - 1.0), 0.0, 1.0))


def root_speed_curvature(motion) -> Tuple[np.ndarray, np.ndarray]:
    m = as_t151(motion)
    root = m[:, [ROOT_X_IDX, ROOT_Z_IDX]].astype(np.float32)
    speed = np.zeros((len(m),), dtype=np.float32)
    curvature = np.zeros((len(m),), dtype=np.float32)
    if len(m) <= 2:
        return speed, curvature

    vel = root[1:] - root[:-1]
    speed[1:] = np.linalg.norm(vel, axis=-1)
    speed[0] = speed[1]

    if len(vel) >= 2:
        v1 = vel[:-1]
        v2 = vel[1:]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos = np.sum(v1 * v2, axis=1) / np.clip(n1 * n2, 1e-8, None)
        turn = 1.0 - np.clip(cos, -1.0, 1.0)
        curvature[2:] = turn
        curvature[1] = turn[0]
    return speed.astype(np.float32), curvature.astype(np.float32)


def event_frames_from_motion(motion, speed_q: float = 70.0, turn_q: float = 75.0) -> Dict[str, List[int]]:
    m = as_t151(motion)
    speed, curvature = root_speed_curvature(m)

    contacts = (m[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_delta = np.zeros((len(m),), dtype=np.float32)
    if len(m) > 1:
        contact_delta[1:] = np.abs(contacts[1:] - contacts[:-1]).sum(axis=1)

    support_switch = np.where(contact_delta > 0.0)[0].astype(int).tolist()

    if float(speed.max()) > 1e-8:
        speed_thr = np.percentile(speed, speed_q)
        speed_peak = np.where(speed >= speed_thr)[0].astype(int).tolist()
    else:
        speed_peak = []

    if float(curvature.max()) > 1e-8:
        turn_thr = np.percentile(curvature, turn_q)
        turn_peak = np.where(curvature >= turn_thr)[0].astype(int).tolist()
    else:
        turn_peak = []

    return {
        "support_switch": support_switch,
        "speed_peak": speed_peak,
        "turn_peak": turn_peak,
    }


def functional_choreo_stats(motion) -> Dict[str, float]:
    """Return coupling stats for a [T,151] motion or unit."""
    m = as_t151(motion)
    speed, curvature = root_speed_curvature(m)

    lower_e = _rms_frame_delta(m[:, LOWER_ROT_INDEX])
    torso_e = _rms_frame_delta(m[:, TORSO_ROT_INDEX])
    upper_e = _rms_frame_delta(m[:, UPPER_ROT_INDEX])
    expr_e = 0.5 * torso_e + 0.5 * upper_e

    events = event_frames_from_motion(m)

    support_expression_coupling = _window_response(
        expr_e,
        events["support_switch"],
        radius=3,
    )
    turn_expression_response = _window_response(
        torso_e + upper_e,
        events["turn_peak"],
        radius=4,
    )
    speed_lower_sync = _safe_corr(speed, lower_e)
    speed_torso_sync = _safe_corr(speed, torso_e)
    speed_upper_sync = _safe_corr(speed, upper_e)
    speed_expression_sync = _safe_corr(speed, expr_e)

    lower_torso_sync = _safe_corr(lower_e, torso_e)
    lower_upper_sync = _safe_corr(lower_e, upper_e)

    contacts = (m[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_switch = float(np.abs(contacts[1:] - contacts[:-1]).mean()) if len(m) > 1 else 0.0

    root_path = float(np.linalg.norm(m[1:, [ROOT_X_IDX, ROOT_Z_IDX]] - m[:-1, [ROOT_X_IDX, ROOT_Z_IDX]], axis=-1).sum()) if len(m) > 1 else 0.0
    root_max_step = float(np.linalg.norm(m[1:, [ROOT_X_IDX, ROOT_Z_IDX]] - m[:-1, [ROOT_X_IDX, ROOT_Z_IDX]], axis=-1).max()) if len(m) > 1 else 0.0

    return {
        "root_speed_mean": float(speed.mean()),
        "root_speed_max": float(speed.max()),
        "root_path": root_path,
        "root_max_step": root_max_step,
        "turning_mean": float(curvature.mean()),
        "turning_max": float(curvature.max()),
        "lower_activity": float(lower_e.mean()),
        "torso_activity": float(torso_e.mean()),
        "upper_activity": float(upper_e.mean()),
        "expression_activity": float(expr_e.mean()),
        "contact_switch": float(contact_switch),
        "support_switch_count": float(len(events["support_switch"])),
        "speed_peak_count": float(len(events["speed_peak"])),
        "turn_peak_count": float(len(events["turn_peak"])),
        "support_expression_coupling": float(support_expression_coupling),
        "turn_expression_response": float(turn_expression_response),
        "speed_lower_sync": float(speed_lower_sync),
        "speed_torso_sync": float(speed_torso_sync),
        "speed_upper_sync": float(speed_upper_sync),
        "speed_expression_sync": float(speed_expression_sync),
        "lower_torso_sync": float(lower_torso_sync),
        "lower_upper_sync": float(lower_upper_sync),
    }


def add_functional_scores(stats_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Add normalized functional scores to a DB stats dict in-place."""
    def arr(name: str, default: float = 0.0) -> np.ndarray:
        if name in stats_arrays:
            return np.nan_to_num(np.asarray(stats_arrays[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = 0
        for v in stats_arrays.values():
            if np.asarray(v).ndim >= 1:
                n = len(v)
                break
        return np.full((n,), float(default), dtype=np.float32)

    n = len(arr("root_speed_mean"))
    for k in [
        "root_speed_mean",
        "turning_mean",
        "lower_activity",
        "torso_activity",
        "upper_activity",
        "expression_activity",
        "contact_switch",
        "support_expression_coupling",
        "turn_expression_response",
        "speed_lower_sync",
        "speed_torso_sync",
        "speed_upper_sync",
        "speed_expression_sync",
        "lower_torso_sync",
        "lower_upper_sync",
    ]:
        if k in stats_arrays and k + "_norm" not in stats_arrays:
            stats_arrays[k + "_norm"] = robust_norm(arr(k))[0]

    support_context_score = np.clip(
        0.30 * arr("root_speed_mean_norm")
        + 0.25 * arr("lower_activity_norm")
        + 0.20 * arr("contact_switch_norm")
        + 0.15 * arr("speed_lower_sync_norm")
        + 0.10 * arr("turning_mean_norm"),
        0.0,
        1.0,
    ).astype(np.float32)

    expressive_mobile_score = np.clip(
        0.25 * arr("upper_activity_norm")
        + 0.20 * arr("torso_activity_norm")
        + 0.20 * arr("turn_expression_response_norm")
        + 0.15 * arr("speed_expression_sync_norm")
        + 0.10 * arr("root_speed_mean_norm")
        + 0.10 * arr("lower_torso_sync_norm"),
        0.0,
        1.0,
    ).astype(np.float32)

    functional_coupling_score = np.clip(
        0.25 * arr("support_expression_coupling_norm")
        + 0.25 * arr("turn_expression_response_norm")
        + 0.20 * arr("speed_lower_sync_norm")
        + 0.15 * arr("speed_torso_sync_norm")
        + 0.15 * (support_context_score * expressive_mobile_score),
        0.0,
        1.0,
    ).astype(np.float32)

    # Units that are both mobile/support-aware and expressive during motion.
    mobile_expressive_score = np.clip(
        0.45 * expressive_mobile_score
        + 0.35 * support_context_score
        + 0.20 * functional_coupling_score,
        0.0,
        1.0,
    ).astype(np.float32)

    stats_arrays["support_context_score"] = support_context_score
    stats_arrays["expressive_mobile_score"] = expressive_mobile_score
    stats_arrays["functional_coupling_score"] = functional_coupling_score
    stats_arrays["mobile_expressive_score"] = mobile_expressive_score
    return stats_arrays


def describe_unit_functional(stats: Dict[str, float]) -> str:
    """Human-readable functional description for one unit/stat row.

    This helper is only used for reporting/captioning. It must be robust even
    when the caller passes raw functional_choreo_stats() output that does not
    yet contain normalized DB-level scores.
    """
    speed_v = float(stats.get("root_speed_mean", 0.0))
    contact_v = float(stats.get("contact_switch", 0.0))
    lower_v = float(stats.get("lower_activity", 0.0))
    expr_v = float(stats.get("expression_activity", 0.0))
    turn_resp_v = float(stats.get("turn_expression_response", 0.0))
    coupling_v = float(stats.get("functional_coupling_score", 0.0))

    speed = "快速移动" if speed_v > 0.03 else ("平稳移动" if speed_v > 0.01 else "缓慢移动")
    support = "支撑切换明显" if contact_v > 0.02 else "支撑稳定"
    lower = "步态活跃" if lower_v > 0.006 else ("步态变化" if lower_v > 0.003 else "下肢稳定")
    upper = "上身表达明显" if expr_v > 0.006 else ("上身舒展" if expr_v > 0.003 else "上身含蓄")
    turn = "转向响应明显" if turn_resp_v > 0.25 else "方向平稳"
    coupling = "走位-支撑-表达耦合强" if coupling_v > 0.60 else ("走位-表达耦合" if coupling_v > 0.30 else "耦合一般")

    return "，".join([speed, support, lower, upper, turn, coupling, "敦煌舞"])

