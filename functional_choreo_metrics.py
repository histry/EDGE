"""Functional choreography metrics for EDGE 151D motion.

Replacement version adds optional target-trajectory event metrics while keeping
old functions used by build_functional_choreo_rag_db.py.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from footstep_phase_utils import (
    CONTACT_SLICE,
    LOWER_ROT_INDEX,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    TORSO_ROT_INDEX,
    UPPER_ROT_INDEX,
    as_t151,
    robust_norm,
)
from turn_aware_event_utils import detect_turn_events, interp_traj, parse_points


def _rms_frame_delta(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros((len(x),), dtype=np.float32)
    out = np.zeros((len(x),), dtype=np.float32)
    out[1:] = np.sqrt(np.mean((x[1:] - x[:-1]) ** 2, axis=1))
    return out.astype(np.float32)


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
    if len(signal) == 0 or not event_frames:
        return 0.0
    base = float(signal.mean()) + 1e-8
    vals = []
    for f in event_frames:
        s = max(0, int(f) - radius)
        e = min(len(signal), int(f) + radius + 1)
        if e > s:
            vals.append(float(signal[s:e].mean()))
    if not vals:
        return 0.0
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
        heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0])).astype(np.float32)
        turn = np.abs(np.diff(heading)).astype(np.float32)
        curvature[2:] = turn / np.clip(speed[2:], 1e-6, None)
        curvature[1] = curvature[2]
    return speed.astype(np.float32), np.nan_to_num(curvature).astype(np.float32)


def event_frames_from_motion(motion, speed_q: float = 70.0, turn_q: float = 75.0) -> Dict[str, List[int]]:
    m = as_t151(motion)
    speed, curvature = root_speed_curvature(m)
    contacts = (m[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_delta = np.zeros((len(m),), dtype=np.float32)
    if len(m) > 1:
        contact_delta[1:] = np.abs(contacts[1:] - contacts[:-1]).sum(axis=1)
    support_switch = np.where(contact_delta > 0.0)[0].astype(int).tolist()
    speed_peak = []
    turn_peak = []
    if float(speed.max()) > 1e-8:
        speed_peak = np.where(speed >= np.percentile(speed, speed_q))[0].astype(int).tolist()
    if float(curvature.max()) > 1e-8:
        turn_peak = np.where(curvature >= np.percentile(curvature, turn_q))[0].astype(int).tolist()
    return {"support_switch": support_switch, "speed_peak": speed_peak, "turn_peak": turn_peak}


def energy_streams(motion):
    m = as_t151(motion)
    lower_e = _rms_frame_delta(m[:, LOWER_ROT_INDEX])
    torso_e = _rms_frame_delta(m[:, TORSO_ROT_INDEX])
    upper_e = _rms_frame_delta(m[:, UPPER_ROT_INDEX])
    expr_e = 0.5 * torso_e + 0.5 * upper_e
    return lower_e, torso_e, upper_e, expr_e


def trajectory_event_frames(trajectory: Optional[str], seq_len: int, count: int = 5) -> Dict[str, List[int]]:
    if not trajectory:
        return {}
    rep = detect_turn_events(trajectory, seq_len=seq_len, count=count)
    return {
        "target_turn_center": list(map(int, rep["event_centers"])),
        "target_support_frames": list(map(int, rep["support_frames"])),
        "target_expressive_frames": list(map(int, rep["expressive_frames"])),
    }


def functional_choreo_stats(motion, target_trajectory: Optional[str] = None) -> Dict[str, float]:
    """Return coupling stats for [T,151] motion.

    If target_trajectory is given, additional target-event response scores are
    computed from the trajectory instead of the generated root. This is useful
    after root X/Z is preserved by compositor.
    """
    m = as_t151(motion)
    speed, curvature = root_speed_curvature(m)
    lower_e, torso_e, upper_e, expr_e = energy_streams(m)
    events = event_frames_from_motion(m)
    support_expression_coupling = _window_response(expr_e, events["support_switch"], radius=3)
    turn_expression_response = _window_response(torso_e + upper_e, events["turn_peak"], radius=4)
    speed_lower_sync = _safe_corr(speed, lower_e)
    speed_torso_sync = _safe_corr(speed, torso_e)
    speed_upper_sync = _safe_corr(speed, upper_e)
    speed_expression_sync = _safe_corr(speed, expr_e)
    lower_torso_sync = _safe_corr(lower_e, torso_e)
    lower_upper_sync = _safe_corr(lower_e, upper_e)
    contacts = (m[:, CONTACT_SLICE] > 0.5).astype(np.float32)
    contact_switch = float(np.abs(contacts[1:] - contacts[:-1]).mean()) if len(m) > 1 else 0.0
    root_step = np.linalg.norm(m[1:, [ROOT_X_IDX, ROOT_Z_IDX]] - m[:-1, [ROOT_X_IDX, ROOT_Z_IDX]], axis=-1) if len(m) > 1 else np.zeros((0,), dtype=np.float32)
    out = {
        "root_speed_mean": float(speed.mean()),
        "root_speed_max": float(speed.max()),
        "root_path": float(root_step.sum()) if len(root_step) else 0.0,
        "root_max_step": float(root_step.max()) if len(root_step) else 0.0,
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
    target_events = trajectory_event_frames(target_trajectory, len(m), count=5) if target_trajectory else {}
    if target_events:
        out["target_turn_expression_response"] = float(_window_response(torso_e + upper_e, target_events["target_turn_center"], radius=5))
        out["target_support_lower_response"] = float(_window_response(lower_e, target_events["target_support_frames"], radius=5))
        out["target_expressive_response"] = float(_window_response(expr_e, target_events["target_expressive_frames"], radius=5))
    return out


def add_functional_scores(stats_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    def arr(name: str, default: float = 0.0) -> np.ndarray:
        if name in stats_arrays:
            return np.nan_to_num(np.asarray(stats_arrays[name], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        n = 0
        for v in stats_arrays.values():
            if np.asarray(v).ndim >= 1:
                n = len(v)
                break
        return np.full((n,), float(default), dtype=np.float32)

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
        + 0.25 * arr("turn_expression_response_norm")
        + 0.15 * arr("speed_expression_sync_norm")
        + 0.15 * arr("lower_torso_sync_norm"),
        0.0,
        1.0,
    ).astype(np.float32)
    functional_coupling_score = np.clip(
        0.25 * arr("support_expression_coupling_norm")
        + 0.30 * arr("turn_expression_response_norm")
        + 0.15 * arr("speed_lower_sync_norm")
        + 0.15 * arr("speed_torso_sync_norm")
        + 0.15 * (support_context_score * expressive_mobile_score),
        0.0,
        1.0,
    ).astype(np.float32)
    mobile_expressive_score = np.clip(
        0.45 * expressive_mobile_score + 0.35 * support_context_score + 0.20 * functional_coupling_score,
        0.0,
        1.0,
    ).astype(np.float32)
    stats_arrays["support_context_score"] = support_context_score
    stats_arrays["expressive_mobile_score"] = expressive_mobile_score
    stats_arrays["functional_coupling_score"] = functional_coupling_score
    stats_arrays["mobile_expressive_score"] = mobile_expressive_score
    return stats_arrays


def describe_unit_functional(stats: Dict[str, float]) -> str:
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
