#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V39 contact-stability postprocess for EDGE / Dunhuang whole-song generation.

Drop-in replacement for tools/v34_motion_quality_postprocess.py.

Compared with V38, this version keeps the three proven fixes
(contact denoise, SmoothStep contact ramp, targeted low-pass), and adds:

1) confidence + hysteresis contact inference:
   label / height / foot-speed voting is converted to a continuous confidence;
   on/off thresholds and morphology remove label flicker and holes.
2) support-aware footplant root correction:
   root X/Z correction is solved from all active ankle/toe anchors with robust
   segment anchors, SmoothStep entrance/exit weights, and velocity/acceleration
   limits on the correction signal.
3) residual-only Butterworth filtering:
   low-pass only the IK/postprocess residual on selected joints, preserving the
   original low-frequency dance phrase while removing high-frequency IK jitter.

The file is intentionally self-contained so it can be copied directly into an
existing EDGE checkout.  All new behavior is controlled by V39_* environment
switches but remains backward-compatible with V34/V38 CLI flags.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton
except Exception:  # pragma: no cover
    ax_from_6v = None
    SMPLSkeleton = None

try:
    from scipy.signal import butter, filtfilt
except Exception:  # pragma: no cover
    butter = None
    filtfilt = None

PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int64,
)
OFFSETS = np.array(
    [
        [0, 0, 0],
        [-0.10, -0.10, 0],
        [0.10, -0.10, 0],
        [0, 0.13, 0],
        [0, -0.42, 0],
        [0, -0.42, 0],
        [0, 0.14, 0],
        [0, -0.40, 0],
        [0, -0.40, 0],
        [0, 0.14, 0],
        [0, -0.08, 0.12],
        [0, -0.08, 0.12],
        [0, 0.14, 0],
        [-0.10, 0.08, 0],
        [0.10, 0.08, 0],
        [0, 0.16, 0],
        [-0.18, 0, 0],
        [0.18, 0, 0],
        [-0.28, 0, 0],
        [0.28, 0, 0],
        [-0.25, 0, 0],
        [0.25, 0, 0],
        [-0.08, 0, 0],
        [0.08, 0, 0],
    ],
    dtype=np.float32,
)
FOOT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)  # lankle, rankle, ltoes, rtoes
UPPER_BODY_JOINTS = np.array([13, 14, 16, 17, 18, 19, 20, 21, 22, 23], dtype=np.int64)
COLLISION_PAIRS = (
    (18, 3), (19, 3), (20, 3), (21, 3), (22, 3), (23, 3),
    (18, 6), (19, 6), (20, 6), (21, 6), (22, 6), (23, 6),
    (18, 9), (19, 9), (20, 9), (21, 9), (22, 9), (23, 9),
    (20, 12), (21, 12), (22, 23), (20, 21),
)
JOINT_NAME_TO_ID = {
    "root": 0, "lhip": 1, "rhip": 2, "spine": 3, "lknee": 4, "rknee": 5,
    "belly": 6, "lankle": 7, "rankle": 8, "chest": 9, "ltoes": 10, "rtoes": 11,
    "neck": 12, "linshoulder": 13, "rinshoulder": 14, "head": 15,
    "lshoulder": 16, "rshoulder": 17, "lelbow": 18, "relbow": 19,
    "lwrist": 20, "rwrist": 21, "lhand": 22, "rhand": 23,
}


def _enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _ef(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _ei(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _load_motion(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if isinstance(x, np.ndarray) and x.ndim == 0 and isinstance(x.item(), dict):
        obj = x.item()
        x = obj.get("motion", obj.get("pose", obj.get("arr_0", x)))
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


def _normalize_6d(x: np.ndarray) -> np.ndarray:
    m = _rot6d_to_matrix_np(x)
    return np.concatenate([m[..., 0], m[..., 1]], axis=-1).astype(np.float32)


def _fk_from_t151_np(motion: np.ndarray) -> np.ndarray:
    if _enabled("V34_POSTPROCESS_USE_SMPL_FK", "1") and torch is not None and ax_from_6v is not None and SMPLSkeleton is not None:
        try:
            dev = torch.device(os.getenv("V34_POSTPROCESS_FK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
            root = torch.tensor(motion[:, [4, 5, 6]], dtype=torch.float32, device=dev).unsqueeze(0)
            q = torch.tensor(motion[:, 7:151], dtype=torch.float32, device=dev).reshape(1, motion.shape[0], 24, 6)
            with torch.no_grad():
                joints = SMPLSkeleton(device=dev).forward(ax_from_6v(q), root)
            return joints[0].detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
    t = motion.shape[0]
    root = motion[:, [4, 5, 6]].astype(np.float32)
    local = _rot6d_to_matrix_np(motion[:, 7:151].reshape(t, 24, 6))
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    glob = np.zeros((t, 24, 3, 3), dtype=np.float32)
    joints[:, 0] = root
    glob[:, 0] = local[:, 0]
    for j in range(1, 24):
        p = int(PARENTS[j])
        glob[:, j] = np.matmul(glob[:, p], local[:, j])
        joints[:, j] = joints[:, p] + np.matmul(glob[:, p], OFFSETS[j][None, :, None])[..., 0]
    return joints


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    k = np.ones((window,), dtype=np.float32) / float(window)
    flat = x.reshape(x.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.convolve(padded[:, i], k, mode="valid")
    return out.reshape(x.shape).astype(np.float32)


def _segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.astype(bool).tolist() + [False]):
        if flag and start is None:
            start = i
        elif (not flag) and start is not None:
            if i - start >= int(min_len):
                rows.append((start, i))
            start = None
    return rows


def _remove_short_runs(x: np.ndarray, min_run: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=bool)
    for s, e in _segments(x, 1):
        if e - s >= int(min_run):
            y[s:e] = True
    return y


def _median_bool(x: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    if size <= 1 or len(x) < 3:
        return x.astype(bool)
    if size % 2 == 0:
        size += 1
    pad = size // 2
    y = np.pad(x.astype(np.uint8), (pad, pad), mode="edge")
    out = np.zeros_like(x, dtype=bool)
    need = size // 2 + 1
    for i in range(len(x)):
        out[i] = int(np.sum(y[i:i + size])) >= need
    return out


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    y = x.astype(bool).copy()
    if max_gap <= 0:
        return y
    for s, e in _segments(~y, 1):
        if s > 0 and e < len(y) and y[s - 1] and y[e] and (e - s) <= int(max_gap):
            y[s:e] = True
    return y


def _hysteresis_from_confidence(conf: np.ndarray, on: float, off: float) -> np.ndarray:
    state = False
    y = np.zeros((len(conf),), dtype=bool)
    on = float(on)
    off = min(float(off), on - 1e-6)
    for i, c in enumerate(conf.astype(np.float32)):
        if not state and c >= on:
            state = True
        elif state and c <= off:
            state = False
        y[i] = state
    return y


def _denoise_contact(
    raw: np.ndarray,
    enabled: bool = True,
    median_size: int = 5,
    close_holes: int = 5,
    open_spikes: int = 3,
    min_contact_frames: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = raw.astype(bool)
    if raw.ndim != 2 or raw.shape[1] != 4:
        return raw, {"enabled": False, "reason": "bad_shape"}
    clean = np.zeros_like(raw, dtype=bool)
    rows = []
    if not enabled:
        for f in range(4):
            clean[:, f] = _remove_short_runs(raw[:, f], min_contact_frames)
            rows.append({"foot": f, "before_rate": float(np.mean(raw[:, f])), "after_rate": float(np.mean(clean[:, f]))})
        return clean, {"enabled": False, "per_foot": rows}
    for f in range(4):
        x = raw[:, f]
        y = _median_bool(x, median_size)
        y = _fill_short_gaps(y, close_holes)
        y = _remove_short_runs(y, max(open_spikes, min_contact_frames))
        clean[:, f] = y
        rows.append({
            "foot": f,
            "before_rate": float(np.mean(x)),
            "after_rate": float(np.mean(y)),
            "changed_frames": int(np.sum(x != y)),
        })
    return clean, {
        "enabled": True,
        "median_size": int(median_size),
        "close_holes": int(close_holes),
        "open_spikes": int(open_spikes),
        "min_contact_frames": int(min_contact_frames),
        "per_foot": rows,
    }


def _contact_confidence(
    motion: np.ndarray,
    threshold: float,
    min_contact_frames: int,
    median_size: int,
    close_holes: int,
    open_spikes: int,
    use_hysteresis: bool,
    on_threshold: float,
    off_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    label = np.asarray(motion[:, 0:4], dtype=np.float32)
    label_bin = label >= float(threshold)
    conf = label_bin.astype(np.float32)
    meta: Dict[str, Any] = {"source": "label_only", "label_rate": float(np.mean(label_bin))}

    if _enabled("V34_KINEMATIC_CONTACT_INFER", "1") and len(motion) >= 2:
        try:
            joints = _fk_from_t151_np(motion)
            feet = joints[:, FOOT_JOINTS, :]
            foot_y = feet[:, :, 1]
            q = float(np.clip(_ef("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
            floor = float(_ef("V34_FLOOR_Y", float(np.quantile(foot_y.reshape(-1), q))))
            h_margin = max(_ef("V34_KIN_CONTACT_HEIGHT", 0.045), 1e-5)
            height_conf = 1.0 - np.clip((foot_y - floor) / h_margin, 0.0, 1.0)
            speed = np.zeros_like(foot_y, dtype=np.float32)
            speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            s_gate = max(_ef("V34_KIN_CONTACT_SPEED", 0.035), 1e-6)
            speed_conf = 1.0 - np.clip(speed / s_gate, 0.0, 1.0)
            lw = _ef("V39_CONTACT_LABEL_WEIGHT", 0.45)
            hw = _ef("V39_CONTACT_HEIGHT_WEIGHT", 0.30)
            sw = _ef("V39_CONTACT_SPEED_WEIGHT", 0.25)
            denom = max(lw + hw + sw, 1e-6)
            conf = (lw * label_bin.astype(np.float32) + hw * height_conf + sw * speed_conf) / denom
            meta.update({
                "source": "label_height_speed_confidence",
                "floor_y": floor,
                "height_margin": h_margin,
                "speed_gate": s_gate,
                "label_weight": lw,
                "height_weight": hw,
                "speed_weight": sw,
                "confidence_mean": float(np.mean(conf)),
            })
        except Exception as exc:
            meta.update({"kinematic_infer_failed": repr(exc)})
            conf = label_bin.astype(np.float32)

    raw = np.zeros_like(label_bin, dtype=bool)
    if use_hysteresis:
        for f in range(4):
            raw[:, f] = _hysteresis_from_confidence(conf[:, f], on_threshold, off_threshold)
        meta.update({"hysteresis": True, "on_threshold": float(on_threshold), "off_threshold": float(off_threshold)})
    else:
        raw = conf >= float(threshold)
        meta.update({"hysteresis": False})

    clean, den = _denoise_contact(
        raw,
        enabled=True,
        median_size=median_size,
        close_holes=close_holes,
        open_spikes=open_spikes,
        min_contact_frames=max(1, int(min_contact_frames)),
    )
    meta["denoise"] = den
    meta["raw_rate"] = float(np.mean(raw))
    meta["clean_rate"] = float(np.mean(clean))
    return clean, conf.astype(np.float32), meta


def _contact_mask(
    motion: np.ndarray,
    threshold: float,
    min_contact_frames: int = 1,
    denoise: bool = True,
    median_size: int = 5,
    close_holes: int = 5,
    open_spikes: int = 3,
) -> np.ndarray:
    if _enabled("V39_CONTACT_HYSTERESIS", "1"):
        clean, _, _ = _contact_confidence(
            motion,
            threshold,
            min_contact_frames,
            median_size,
            close_holes,
            open_spikes,
            use_hysteresis=True,
            on_threshold=_ef("V39_CONTACT_ON_THRESHOLD", 0.58),
            off_threshold=_ef("V39_CONTACT_OFF_THRESHOLD", 0.42),
        )
        if not denoise:
            return clean
        return clean
    label = np.asarray(motion[:, 0:4], dtype=np.float32) >= float(threshold)
    raw = label
    if _enabled("V34_KINEMATIC_CONTACT_INFER", "1") and len(motion) >= 2:
        try:
            joints = _fk_from_t151_np(motion)
            feet = joints[:, FOOT_JOINTS, :]
            foot_y = feet[:, :, 1]
            q = float(np.clip(_ef("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
            floor = float(np.quantile(foot_y.reshape(-1), q))
            height = foot_y <= floor + _ef("V34_KIN_CONTACT_HEIGHT", 0.045)
            speed = np.zeros_like(foot_y, dtype=np.float32)
            speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            speed_gate = speed <= _ef("V34_KIN_CONTACT_SPEED", 0.035)
            votes = label.astype(np.int32) + height.astype(np.int32) + speed_gate.astype(np.int32)
            raw = votes >= int(np.clip(_ef("V34_KIN_CONTACT_VOTE_THRESHOLD", 2.0), 1, 3))
        except Exception:
            raw = label
    clean, _ = _denoise_contact(raw, enabled=denoise, median_size=median_size, close_holes=close_holes, open_spikes=open_spikes, min_contact_frames=max(1, int(min_contact_frames)))
    return clean


def _mean_contact_speed(joints: np.ndarray, contact: np.ndarray) -> float:
    feet = joints[:, FOOT_JOINTS, :]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    return float(np.mean(speed[contact.astype(bool)])) if np.any(contact) else 0.0


def _collision_stats(joints: np.ndarray, radius: float) -> Dict[str, float]:
    dists = [np.linalg.norm(joints[:, a] - joints[:, b], axis=-1) for a, b in COLLISION_PAIRS]
    dist = np.stack(dists, axis=1)
    pen = np.maximum(0.0, float(radius) - dist)
    return {
        "risk": float(np.mean(np.square(pen / max(float(radius), 1e-8)))),
        "min_distance": float(np.min(dist)),
        "bad_frames": int(np.sum(np.any(pen > 0, axis=1))),
    }


def _quality_audit(
    motion: np.ndarray,
    contact_threshold: float,
    min_contact_frames: int,
    floor_margin: float,
    collision_radius: float,
    denoise: bool,
    median_size: int,
    close_holes: int,
    open_spikes: int,
) -> Dict[str, float]:
    joints = _fk_from_t151_np(motion)
    feet = joints[:, FOOT_JOINTS, :]
    foot_y = feet[:, :, 1]
    q = float(np.clip(_ef("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
    floor = float(_ef("V34_FLOOR_Y", float(np.quantile(foot_y.reshape(-1), q))))
    contact = _contact_mask(motion, contact_threshold, min_contact_frames, denoise, median_size, close_holes, open_spikes)
    feet_xz = feet[:, :, [0, 2]]
    vals = []
    max_vals = []
    for f in range(4):
        for s, e in _segments(contact[:, f], max(2, int(min_contact_frames))):
            if e - s > 1:
                v = np.linalg.norm(feet_xz[s + 1:e, f] - feet_xz[s:e - 1, f], axis=-1)
                vals.append(v)
                max_vals.append(float(np.max(v)))
    skate = float(np.mean(np.concatenate(vals))) if vals else 0.0
    skate_p95 = float(np.percentile(np.concatenate(vals), 95)) if vals else 0.0
    root_y = motion[:, 5]
    root_v = np.diff(root_y, prepend=root_y[:1])
    root_a = np.diff(root_v, prepend=root_v[:1])
    vel = np.diff(joints, axis=0, prepend=joints[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    jerk = np.diff(acc, axis=0, prepend=acc[:1])
    mean_jerk = np.linalg.norm(jerk, axis=-1).mean(axis=-1)
    hand_ids = np.array([18, 19, 20, 21, 22, 23], dtype=np.int64)
    hand_jerk = np.linalg.norm(jerk[:, hand_ids], axis=-1).mean(axis=-1)
    col = _collision_stats(joints, collision_radius)
    penetration = foot_y - (floor + float(floor_margin))
    return {
        "floor_y": floor,
        "contact_ratio": float(np.mean(contact)),
        "foot_skate_mean_mpf": skate,
        "foot_skate_p95_mpf": skate_p95,
        "foot_skate_max_mpf": float(max(max_vals) if max_vals else 0.0),
        "foot_penetration_min_m": float(np.min(penetration)),
        "root_y_range_m": float(np.max(root_y) - np.min(root_y)),
        "root_y_acc_mean": float(np.mean(np.abs(root_a))),
        "root_y_acc_p95": float(np.percentile(np.abs(root_a), 95)),
        "mean_joint_jerk_max": float(np.max(mean_jerk)),
        "mean_joint_jerk_p95": float(np.percentile(mean_jerk, 95)),
        "hand_joint_jerk_max": float(np.max(hand_jerk)),
        "hand_joint_jerk_p95": float(np.percentile(hand_jerk, 95)),
        "collision_risk": float(col["risk"]),
        "collision_min_distance": float(col["min_distance"]),
        "collision_bad_frames": float(col["bad_frames"]),
    }


def _quality_rejection_signal(m: Dict[str, float]) -> Dict[str, Any]:
    reasons = []
    if m.get("foot_skate_mean_mpf", 0.0) > _ef("V34_REJECT_MAX_SKATE_MPF", 0.035):
        reasons.append("foot_sliding")
    if m.get("foot_penetration_min_m", 0.0) < _ef("V34_REJECT_MIN_FOOT_PENETRATION", -0.015):
        reasons.append("floor_penetration")
    if m.get("mean_joint_jerk_p95", 0.0) > _ef("V34_REJECT_MAX_JERK_P95", 1200.0):
        reasons.append("high_jitter")
    if m.get("collision_risk", 0.0) > _ef("V34_REJECT_MAX_COLLISION_RISK", 0.10):
        reasons.append("self_collision")
    return {
        "accepted": not reasons,
        "reject_reasons": reasons,
        "planner_action": "accept" if not reasons else "reroute_local_phrase_or_mask_failed_edges",
    }


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0, 1).astype(np.float32)
    return (3 * u * u - 2 * u * u * u).astype(np.float32)


def _seg_weights(n: int, blend: int) -> np.ndarray:
    w = np.ones((n,), dtype=np.float32)
    b = max(0, min(int(blend), max(0, n // 2)))
    if b > 0:
        ramp = _smoothstep((np.arange(b, dtype=np.float32) + 1.0) / float(b))
        w[:b] = np.minimum(w[:b], ramp)
        w[-b:] = np.minimum(w[-b:], ramp[::-1])
    return w


def _limit_signal_step_accel(delta: np.ndarray, max_step: float, max_accel: float, passes: int = 2) -> np.ndarray:
    out = delta.astype(np.float32, copy=True)
    if len(out) < 3:
        return out
    max_step = float(max_step)
    max_accel = float(max_accel)
    if max_step <= 0 and max_accel <= 0:
        return out
    def clamp_norm(v: np.ndarray, limit: float) -> np.ndarray:
        if limit <= 0:
            return v
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v * np.minimum(1.0, limit / np.maximum(n, 1e-8))
    for _ in range(max(1, int(passes))):
        prev_step = np.zeros((2,), dtype=np.float32)
        for i in range(1, len(out)):
            step = out[i] - out[i - 1]
            if max_accel > 0:
                step = prev_step + clamp_norm((step - prev_step)[None, :], max_accel)[0]
            step = clamp_norm(step[None, :], max_step)[0]
            out[i] = out[i - 1] + step
            prev_step = step
        next_step = np.zeros((2,), dtype=np.float32)
        for i in range(len(out) - 2, -1, -1):
            step = out[i] - out[i + 1]
            if max_accel > 0:
                step = next_step + clamp_norm((step - next_step)[None, :], max_accel)[0]
            step = clamp_norm(step[None, :], max_step)[0]
            out[i] = out[i + 1] + step
            next_step = step
    return out.astype(np.float32)


def contact_lock_root(
    motion: np.ndarray,
    contact_threshold: float,
    min_contact_frames: int,
    strength: float,
    smooth_window: int,
    max_correction: float,
    denoise: bool = True,
    median_size: int = 5,
    close_holes: int = 5,
    open_spikes: int = 3,
    blend_frames: int = 5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = motion.astype(np.float32, copy=True)
    joints = _fk_from_t151_np(out)
    contact, conf, cmeta = _contact_confidence(
        out,
        contact_threshold,
        min_contact_frames,
        median_size,
        close_holes,
        open_spikes,
        use_hysteresis=_enabled("V39_CONTACT_HYSTERESIS", "1"),
        on_threshold=_ef("V39_CONTACT_ON_THRESHOLD", 0.58),
        off_threshold=_ef("V39_CONTACT_OFF_THRESHOLD", 0.42),
    )
    if not denoise:
        contact = np.asarray(out[:, 0:4] >= float(contact_threshold), dtype=bool)
        cmeta["denoise_forced_off"] = True
    corr = np.zeros((len(out), 2), dtype=np.float32)
    weight = np.zeros((len(out), 1), dtype=np.float32)
    feet = joints[:, FOOT_JOINTS, :]
    segs = 0
    segment_rows = []
    rejected_rows = []
    use_solver = _enabled("V39_FOOTPLANT_SOLVER", "1")
    # V39B: a long contact segment with large anchor error is usually not a
    # true foot plant.  Locking it to a single median anchor creates pelvis
    # dragging and high-frequency jitter.  Reject those segments before they
    # enter the footplant solver.
    max_seg_mean = _ef("V39_MAX_FOOTPLANT_SEGMENT_ERROR", 0.055)
    max_seg_p95 = _ef("V39_MAX_FOOTPLANT_SEGMENT_P95", 0.110)
    max_seg_frames = _ei("V39_MAX_FOOTPLANT_SEGMENT_FRAMES", 120)
    hard_reject = _enabled("V39_REJECT_MOVING_FOOTPLANTS", "1")
    for f in range(4):
        for s, e in _segments(contact[:, f], min_contact_frames):
            seg = feet[s:e, f, :][:, [0, 2]]
            if e - s < max(2, int(min_contact_frames)):
                continue
            # Robust anchor: median is less sensitive to one-frame IK spikes than mean.
            anchor = np.median(seg, axis=0).astype(np.float32)
            err = anchor[None, :] - seg
            err_norm = np.linalg.norm(err, axis=-1)
            mean_err = float(np.mean(err_norm)) if len(err_norm) else 0.0
            p95_err = float(np.percentile(err_norm, 95)) if len(err_norm) else 0.0
            too_moving = (mean_err > max_seg_mean) or (p95_err > max_seg_p95)
            too_long_moving = (e - s > max_seg_frames) and (mean_err > 0.65 * max_seg_mean)
            if hard_reject and (too_moving or too_long_moving):
                rejected_rows.append({
                    "foot": int(f),
                    "start": int(s),
                    "end": int(e),
                    "frames": int(e - s),
                    "mean_confidence": float(np.mean(conf[s:e, f])),
                    "mean_raw_error": mean_err,
                    "p95_raw_error": p95_err,
                    "reason": "moving_contact_segment_not_footplant",
                })
                continue
            ramp = _seg_weights(e - s, blend_frames)[:, None]
            conf_w = np.clip(conf[s:e, f:f + 1], 0.15, 1.0)
            # Softly reduce the solver strength for marginally stable plants.
            err_gate = float(np.clip(max_seg_mean / max(mean_err, 1e-6), 0.35, 1.0))
            w = ramp * conf_w * err_gate
            if use_solver:
                corr[s:e] += w * err
                weight[s:e] += w
            else:
                corr[s:e] += ramp * err * err_gate
                weight[s:e] += ramp * err_gate
            segs += 1
            segment_rows.append({
                "foot": int(f),
                "start": int(s),
                "end": int(e),
                "frames": int(e - s),
                "anchor_x": float(anchor[0]),
                "anchor_z": float(anchor[1]),
                "mean_confidence": float(np.mean(conf[s:e, f])),
                "mean_raw_error": mean_err,
                "p95_raw_error": p95_err,
                "err_gate": err_gate,
            })
    active = weight[:, 0] > 1e-6
    corr[active] /= np.maximum(weight[active], 1e-6)
    if max_correction > 0:
        n = np.linalg.norm(corr, axis=1, keepdims=True)
        corr *= np.minimum(1.0, float(max_correction) / np.maximum(n, 1e-8))
    if int(smooth_window) > 1:
        corr = _moving_average(corr, smooth_window)
    lw = np.clip(weight, 0, 1)[:, 0]
    root_delta = float(strength) * lw[:, None] * corr
    root_delta = _limit_signal_step_accel(
        root_delta,
        max_step=_ef("V39_ROOT_CORR_MAX_STEP", 0.026),
        max_accel=_ef("V39_ROOT_CORR_MAX_ACCEL", 0.012),
        passes=_ei("V39_ROOT_CORR_LIMIT_PASSES", 2),
    )
    out[:, 4] += root_delta[:, 0]
    out[:, 6] += root_delta[:, 1]
    support_damping = _ef("V39_SUPPORT_ROOT_VELOCITY_DAMPING", 0.10)
    if support_damping > 0 and len(out) > 2:
        support = np.any(contact, axis=1).astype(np.float32)
        support = _moving_average(support[:, None], max(3, int(smooth_window)))[:, 0]
        xz = out[:, [4, 6]].copy()
        v = np.diff(xz, axis=0, prepend=xz[:1])
        v *= (1.0 - float(support_damping) * np.clip(support[:, None], 0.0, 1.0))
        xz2 = xz.copy()
        xz2[1:] = xz[0] + np.cumsum(v[1:], axis=0)
        out[:, [4, 6]] = (0.55 * xz + 0.45 * xz2).astype(np.float32)
    after = _fk_from_t151_np(out)
    return out, {
        "enabled": True,
        "version": "v39_confidence_hysteresis_smoothstep_footplant_solver",
        "contact_segments": int(segs),
        "active_contact_frames": int(np.sum(active)),
        "mean_contact_speed_before": _mean_contact_speed(joints, contact),
        "mean_contact_speed_after": _mean_contact_speed(after, contact),
        "max_root_xz_correction": float(np.max(np.linalg.norm(root_delta, axis=1))) if len(out) else 0.0,
        "root_corr_max_step": _ef("V39_ROOT_CORR_MAX_STEP", 0.026),
        "root_corr_max_accel": _ef("V39_ROOT_CORR_MAX_ACCEL", 0.012),
        "strength": float(strength),
        "smoothstep_blend_frames": int(blend_frames),
        "support_root_velocity_damping": float(support_damping),
        "contact_confidence": cmeta,
        "footplant_gate": {
            "enabled": bool(hard_reject),
            "max_segment_mean_error": float(max_seg_mean),
            "max_segment_p95_error": float(max_seg_p95),
            "max_segment_frames": int(max_seg_frames),
            "accepted_segments": int(len(segment_rows)),
            "rejected_segments": int(len(rejected_rows)),
        },
        "segment_preview": segment_rows[:20],
        "rejected_segment_preview": rejected_rows[:20],
    }


def enforce_root_y_physics(
    motion: np.ndarray,
    contact_threshold: float,
    min_flight_frames: int,
    parabola_strength: float,
    min_arc_lift: float,
    max_arc_lift: float,
    landing_frames: int,
    landing_max_drop: float,
    landing_strength: float,
    denoise: bool = True,
    median_size: int = 5,
    close_holes: int = 5,
    open_spikes: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = motion.astype(np.float32, copy=True)
    contact = _contact_mask(out, contact_threshold, max(2, _ei("V34_KIN_CONTACT_MIN_FRAMES", 3)), denoise, median_size, close_holes, open_spikes)
    support = np.any(contact, axis=1)
    flights = _segments(~support, min_flight_frames)
    y = out[:, 5].copy()
    corrected = y.copy()
    for s, e in flights:
        n = e - s
        if n < 2:
            continue
        u = np.linspace(0, 1, n, dtype=np.float32)
        y0 = float(y[s])
        y1 = float(y[e - 1])
        lin = (1 - u) * y0 + u * y1
        obs = float(np.max(y[s:e]) - max(y0, y1))
        lift = float(np.clip(max(obs, min_arc_lift), 0, max_arc_lift))
        corrected[s:e] = (1 - float(parabola_strength)) * corrected[s:e] + float(parabola_strength) * (lin + lift * 4 * u * (1 - u))
    land = 0
    if landing_frames > 2 and landing_strength > 0:
        trans = np.flatnonzero((support[1:] == True) & (support[:-1] == False)) + 1
        for t in trans:
            e = min(len(out), t + int(landing_frames))
            n = e - t
            if n < 3:
                continue
            pre = max(0, t - 3)
            ds = max(0.0, float(np.mean(y[pre:t]) - y[t]))
            amp = float(np.clip(0.35 * ds + 0.012, 0, landing_max_drop))
            u = np.linspace(0, 1, n, dtype=np.float32)
            corrected[t:e] += float(landing_strength) * (-amp * np.sin(np.pi * u) * np.exp(-1.35 * u))
            land += 1
    out[:, 5] = corrected.astype(np.float32)
    return out, {
        "enabled": True,
        "flight_segments": len(flights),
        "landing_events": land,
        "root_y_delta_mean": float(np.mean(np.abs(corrected - y))),
        "root_y_delta_max": float(np.max(np.abs(corrected - y))),
    }


def enforce_floor_clearance(
    motion: np.ndarray,
    enabled: bool,
    margin: float,
    strength: float,
    smooth_window: int,
    max_lift: float,
    contact_threshold: float,
    min_contact_frames: int,
    support_damping: float,
    denoise: bool = True,
    median_size: int = 5,
    close_holes: int = 5,
    open_spikes: int = 3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    joints = _fk_from_t151_np(motion)
    feet_y = joints[:, FOOT_JOINTS, 1]
    q = float(np.clip(_ef("V34_FLOOR_QUANTILE", 0.12), 0.01, 0.45))
    floor = float(_ef("V34_FLOOR_Y", float(np.quantile(feet_y.reshape(-1), q))))
    pen = np.maximum(0.0, floor + float(margin) - np.min(feet_y, axis=1))
    before = {
        "floor_y": floor,
        "penetrating_frames": int(np.sum(pen > 1e-6)),
        "max_penetration": float(np.max(pen)),
        "mean_penetration": float(np.mean(pen)),
    }
    if not enabled:
        return motion.astype(np.float32, copy=True), {"enabled": False, "before": before}
    lift = pen.astype(np.float32)
    if max_lift > 0:
        lift = np.minimum(lift, float(max_lift))
    lift = _moving_average(lift[:, None], smooth_window)[:, 0]
    out = motion.astype(np.float32, copy=True)
    out[:, 5] += float(strength) * lift
    support = np.zeros((len(out),), dtype=bool)
    if support_damping > 0 and len(out) > 2:
        contact = _contact_mask(out, contact_threshold, min_contact_frames, denoise, median_size, close_holes, open_spikes)
        support = np.any(contact, axis=1)
        y = out[:, 5].copy()
        vel = np.zeros_like(y)
        vel[1:] = y[1:] - y[:-1]
        dvel = vel.copy()
        dvel[support] *= max(0, 1 - float(support_damping))
        yd = y.copy()
        yd[1:] = y[0] + np.cumsum(dvel[1:])
        out[:, 5] = (0.65 * y + 0.35 * yd).astype(np.float32)
    aj = _fk_from_t151_np(out)
    ap = np.maximum(0.0, floor + float(margin) - np.min(aj[:, FOOT_JOINTS, 1], axis=1))
    return out, {
        "enabled": True,
        "margin": float(margin),
        "support_damping": float(support_damping),
        "support_frame_ratio": float(np.mean(support)),
        "before": before,
        "after": {
            "penetrating_frames": int(np.sum(ap > 1e-6)),
            "max_penetration": float(np.max(ap)),
            "mean_penetration": float(np.mean(ap)),
        },
        "max_root_y_lift": float(np.max(lift)) if len(lift) else 0.0,
    }


def smooth_rotations_only(motion: np.ndarray, rotation_window: int, strength: float) -> np.ndarray:
    out = motion.astype(np.float32, copy=True)
    strength = float(np.clip(strength, 0, 1))
    if rotation_window > 1 and strength > 0:
        sm = _moving_average(out[:, 7:151], rotation_window)
        out[:, 7:151] = (1 - strength) * out[:, 7:151] + strength * sm
        out[:, 7:151] = _normalize_6d(out[:, 7:151].reshape(-1, 24, 6)).reshape(-1, 144)
    return out


def _rot6d_to_matrix_torch(x):
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / torch.clamp(torch.linalg.norm(a1, dim=-1, keepdim=True), min=1e-8)
    proj = torch.sum(b1 * a2, dim=-1, keepdim=True) * b1
    b2 = a2 - proj
    b2 = b2 / torch.clamp(torch.linalg.norm(b2, dim=-1, keepdim=True), min=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _fk_torch(motion):
    t = motion.shape[0]
    root = motion[:, [4, 5, 6]]
    local = _rot6d_to_matrix_torch(motion[:, 7:151].reshape(t, 24, 6))
    joints = []
    glob = []
    offsets = torch.as_tensor(OFFSETS, dtype=motion.dtype, device=motion.device)
    for j in range(24):
        if j == 0:
            joints.append(root)
            glob.append(local[:, 0])
        else:
            p = int(PARENTS[j])
            glob.append(torch.matmul(glob[p], local[:, j]))
            joints.append(joints[p] + torch.matmul(glob[p], offsets[j].view(1, 3, 1)).squeeze(-1))
    return torch.stack(joints, dim=1)


def collision_aware_ik(
    motion: np.ndarray,
    enabled: bool,
    radius: float,
    steps: int,
    lr: float,
    collision_weight: float,
    reg_weight: float,
    temporal_weight: float,
    device: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    before = _collision_stats(_fk_from_t151_np(motion), radius)
    if not enabled or torch is None or before["bad_frames"] == 0 or steps <= 0:
        before.update({"enabled": bool(enabled and torch is not None), "skipped": True})
        return motion.astype(np.float32, copy=True), before
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    base = torch.as_tensor(motion.astype(np.float32), device=dev)
    upper = torch.as_tensor(UPPER_BODY_JOINTS, dtype=torch.long, device=dev)
    original = base[:, 7:151].reshape(-1, 24, 6)[:, upper].detach()
    var = original.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([var], lr=float(lr))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        full = base[:, 7:151].reshape(-1, 24, 6).clone()
        full[:, upper] = var
        cand = base.clone()
        cand[:, 7:151] = full.reshape(-1, 144)
        joints = _fk_torch(cand)
        losses = [torch.relu(float(radius) - torch.linalg.norm(joints[:, a] - joints[:, b], dim=-1)) ** 2 for a, b in COLLISION_PAIRS]
        coll = torch.stack(losses, dim=1).mean()
        reg = torch.mean((var - original) ** 2)
        temp = torch.mean((var[1:] - var[:-1]) ** 2) if var.shape[0] > 2 else torch.zeros((), dtype=var.dtype, device=var.device)
        loss = float(collision_weight) * coll + float(reg_weight) * reg + float(temporal_weight) * temp
        loss.backward()
        opt.step()
    out = motion.astype(np.float32, copy=True)
    rot = out[:, 7:151].reshape(-1, 24, 6)
    rot[:, UPPER_BODY_JOINTS] = var.detach().cpu().numpy()
    out[:, 7:151] = _normalize_6d(rot).reshape(-1, 144)
    after = _collision_stats(_fk_from_t151_np(out), radius)
    return out, {"enabled": True, "skipped": False, "device": str(dev), "steps": int(steps), "before": before, "after": after}


def _parse_joints(text: str) -> np.ndarray:
    vals = []
    for it in str(text or "").replace(";", ",").split(","):
        it = it.strip().lower()
        if not it:
            continue
        vals.append(JOINT_NAME_TO_ID[it] if it in JOINT_NAME_TO_ID else int(float(it)))
    return np.asarray(sorted(set([v for v in vals if 0 <= v < 24])) or [18, 19, 20, 21, 22, 23, 16, 17], dtype=np.int64)


def _fft_lowpass(x: np.ndarray, fps: float, cutoff: float) -> np.ndarray:
    freq = np.fft.rfftfreq(len(x), d=1.0 / max(float(fps), 1e-6))
    X = np.fft.rfft(x, axis=0)
    X[freq > float(cutoff)] = 0
    return np.fft.irfft(X, n=len(x), axis=0).astype(np.float32)


def _lowpass_array(x: np.ndarray, fps: float, cutoff_hz: float, order: int) -> Tuple[np.ndarray, str, float]:
    nyq = 0.5 * float(fps)
    cutoff = float(np.clip(cutoff_hz, 0.1, max(0.11, nyq * 0.95)))
    method = "fft_fallback"
    filt = None
    if butter is not None and filtfilt is not None and len(x) > 3 * (int(order) + 1):
        try:
            b, a = butter(int(order), cutoff / nyq, btype="low", analog=False)
            filt = filtfilt(b, a, x, axis=0).astype(np.float32)
            method = "butterworth_filtfilt"
        except Exception:
            filt = None
    if filt is None:
        filt = _fft_lowpass(x, fps, cutoff)
    return filt.astype(np.float32), method, cutoff


def butterworth_lowpass_rotations(
    motion: np.ndarray,
    enabled: bool,
    fps: float,
    cutoff_hz: float,
    order: int,
    strength: float,
    joints_text: str,
    reference_motion: np.ndarray | None = None,
    residual_only: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    out = motion.astype(np.float32, copy=True)
    if not enabled or len(out) < 8 or strength <= 0:
        return out, {"enabled": bool(enabled), "skipped": True}
    ids = _parse_joints(joints_text)
    rot = out[:, 7:151].reshape(len(out), 24, 6)
    target = rot[:, ids, :].reshape(len(out), -1).astype(np.float32)
    method = "butterworth_filtfilt"
    cutoff = float(cutoff_hz)
    if residual_only and reference_motion is not None and reference_motion.shape == motion.shape:
        ref = reference_motion[:, 7:151].reshape(len(out), 24, 6)[:, ids, :].reshape(len(out), -1).astype(np.float32)
        residual = target - ref
        filt_residual, method, cutoff = _lowpass_array(residual, fps, cutoff_hz, order)
        filtered = ref + ((1.0 - float(strength)) * residual + float(strength) * filt_residual)
        mode = "residual_only"
    else:
        filt, method, cutoff = _lowpass_array(target, fps, cutoff_hz, order)
        filtered = (1.0 - float(strength)) * target + float(strength) * filt
        mode = "absolute_rotation"
    rot[:, ids, :] = filtered.reshape(len(out), len(ids), 6)
    out[:, 7:151] = _normalize_6d(rot).reshape(len(out), 144)
    return out, {
        "enabled": True,
        "skipped": False,
        "version": "v39_residual_only_butterworth" if residual_only else "v38_absolute_butterworth",
        "mode": mode,
        "method": method,
        "fps": float(fps),
        "cutoff_hz": cutoff,
        "order": int(order),
        "strength": float(strength),
        "joint_ids": [int(x) for x in ids],
    }


def process_file(args: argparse.Namespace) -> Dict[str, Any]:
    src = Path(args.motion)
    out_path = Path(args.out)
    motion = _load_motion(src)
    original = motion.copy()
    den = bool(args.contact_denoise)
    qargs = dict(
        contact_threshold=args.contact_threshold,
        min_contact_frames=args.min_contact_frames,
        floor_margin=args.floor_margin,
        collision_radius=args.collision_radius,
        denoise=den,
        median_size=args.contact_median_size,
        close_holes=args.contact_close_holes,
        open_spikes=args.contact_open_spikes,
    )
    pre = _quality_audit(original, **qargs)
    physics: Dict[str, Any] = {"enabled": False}
    if args.root_y_physics:
        motion, physics = enforce_root_y_physics(
            motion,
            args.contact_threshold,
            args.min_flight_frames,
            args.parabola_strength,
            args.min_arc_lift,
            args.max_arc_lift,
            args.landing_frames,
            args.landing_max_drop,
            args.landing_strength,
            den,
            args.contact_median_size,
            args.contact_close_holes,
            args.contact_open_spikes,
        )
    motion, collision = collision_aware_ik(
        motion,
        bool(args.collision_ik),
        args.collision_radius,
        args.collision_steps,
        args.collision_lr,
        args.collision_weight,
        args.collision_reg_weight,
        args.collision_temporal_weight,
        args.device,
    )
    contact: Dict[str, Any] = {"enabled": False}
    if args.contact_lock:
        motion, contact = contact_lock_root(
            motion,
            args.contact_threshold,
            args.min_contact_frames,
            args.contact_lock_strength,
            args.contact_smooth_window,
            args.max_root_correction,
            den,
            args.contact_median_size,
            args.contact_close_holes,
            args.contact_open_spikes,
            args.contact_lock_blend_frames,
        )
    motion, floor = enforce_floor_clearance(
        motion,
        bool(args.floor_clearance),
        args.floor_margin,
        args.floor_strength,
        args.floor_smooth_window,
        args.floor_max_lift,
        args.contact_threshold,
        args.min_contact_frames,
        args.floor_support_damping,
        den,
        args.contact_median_size,
        args.contact_close_holes,
        args.contact_open_spikes,
    )
    before_butter = motion.copy()
    before_butter_audit = _quality_audit(before_butter, **qargs)
    motion, butter_sum = butterworth_lowpass_rotations(
        motion,
        bool(args.butterworth_filter),
        args.fps,
        args.butterworth_cutoff_hz,
        args.butterworth_order,
        args.butterworth_strength,
        args.butterworth_joints,
        reference_motion=original,
        residual_only=bool(args.butterworth_residual_only),
    )
    if bool(args.butterworth_filter) and _enabled("V39_BUTTERWORTH_ROLLBACK_IF_WORSE", "1"):
        after_butter_audit = _quality_audit(motion, **qargs)
        max_joint_ratio = _ef("V39_BUTTERWORTH_MAX_JERK_WORSEN", 1.15)
        max_hand_ratio = _ef("V39_BUTTERWORTH_MAX_HAND_WORSEN", 1.15)
        joint_ok = after_butter_audit.get("mean_joint_jerk_p95", 0.0) <= max_joint_ratio * max(before_butter_audit.get("mean_joint_jerk_p95", 0.0), 1e-8)
        hand_ok = after_butter_audit.get("hand_joint_jerk_p95", 0.0) <= max_hand_ratio * max(before_butter_audit.get("hand_joint_jerk_p95", 0.0), 1e-8)
        butter_sum["safety_audit_before"] = before_butter_audit
        butter_sum["safety_audit_after"] = after_butter_audit
        butter_sum["rollback_if_worse"] = True
        butter_sum["rollback_thresholds"] = {"joint_ratio": float(max_joint_ratio), "hand_ratio": float(max_hand_ratio)}
        if not (joint_ok and hand_ok):
            motion = before_butter
            butter_sum["rolled_back"] = True
            butter_sum["rollback_reason"] = "jitter_metric_worse"
        else:
            butter_sum["rolled_back"] = False
    if args.smooth:
        motion = smooth_rotations_only(motion, args.rotation_smooth_window, args.smooth_strength)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, motion.astype(np.float32))
    post = _quality_audit(motion, **qargs)
    reject = _quality_rejection_signal(post)
    summary = {
        "version": "v39b_footplant_gated_contact_stability",
        "input": str(src),
        "output": str(out_path),
        "frames": int(len(motion)),
        "pre_audit": pre,
        "post_audit": post,
        "planner_feedback": reject,
        "audit_improvement": {
            "foot_skate_mean_delta": float(pre["foot_skate_mean_mpf"] - post["foot_skate_mean_mpf"]),
            "foot_skate_p95_delta": float(pre.get("foot_skate_p95_mpf", 0.0) - post.get("foot_skate_p95_mpf", 0.0)),
            "foot_penetration_min_delta": float(post["foot_penetration_min_m"] - pre["foot_penetration_min_m"]),
            "jerk_p95_delta": float(pre["mean_joint_jerk_p95"] - post["mean_joint_jerk_p95"]),
            "hand_jerk_p95_delta": float(pre.get("hand_joint_jerk_p95", 0.0) - post.get("hand_joint_jerk_p95", 0.0)),
            "collision_risk_delta": float(pre["collision_risk"] - post["collision_risk"]),
        },
        "root_y_physics": physics,
        "collision_aware_ik": collision,
        "contact_lock": contact,
        "floor_clearance": floor,
        "butterworth_filter": butter_sum,
        "rotation_smooth": {"enabled": bool(args.smooth), "note": "global smooth kept optional; V39 prefers residual low-pass"},
        "root_xz_delta_mean": float(np.mean(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
        "root_xz_delta_max": float(np.max(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
        "root_y_delta_mean": float(np.mean(np.abs(motion[:, 5] - original[:, 5]))) if len(motion) else 0.0,
        "root_y_delta_max": float(np.max(np.abs(motion[:, 5] - original[:, 5]))) if len(motion) else 0.0,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motion", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary_json", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=_ef("V26_FPS", 30.0))
    p.add_argument("--contact_threshold", type=float, default=_ef("V34_CONTACT_LOCK_THRESHOLD", 0.65))
    p.add_argument("--root_y_physics", type=int, default=_ei("V34_ROOT_Y_PHYSICS", 1))
    p.add_argument("--min_flight_frames", type=int, default=6)
    p.add_argument("--parabola_strength", type=float, default=0.60)
    p.add_argument("--min_arc_lift", type=float, default=0.012)
    p.add_argument("--max_arc_lift", type=float, default=0.10)
    p.add_argument("--landing_frames", type=int, default=8)
    p.add_argument("--landing_max_drop", type=float, default=0.035)
    p.add_argument("--landing_strength", type=float, default=0.75)
    p.add_argument("--collision_ik", type=int, default=_ei("V39_COLLISION_IK", 0))
    p.add_argument("--collision_radius", type=float, default=0.16)
    p.add_argument("--collision_steps", type=int, default=_ei("V39_COLLISION_STEPS", 20))
    p.add_argument("--collision_lr", type=float, default=0.025)
    p.add_argument("--collision_weight", type=float, default=8.0)
    p.add_argument("--collision_reg_weight", type=float, default=0.45)
    p.add_argument("--collision_temporal_weight", type=float, default=_ef("V39_COLLISION_TEMPORAL_WEIGHT", 0.05))
    p.add_argument("--contact_lock", type=int, default=_ei("V34_CONTACT_LOCK_POSTPROCESS", 1))
    p.add_argument("--min_contact_frames", type=int, default=_ei("V34_MIN_CONTACT_FRAMES", 8))
    p.add_argument("--contact_lock_strength", type=float, default=_ef("V34_CONTACT_LOCK_STRENGTH", 0.85))
    p.add_argument("--contact_smooth_window", type=int, default=_ei("V34_CONTACT_LOCK_SMOOTH_WINDOW", 11))
    p.add_argument("--max_root_correction", type=float, default=_ef("V34_CONTACT_LOCK_MAX_ROOT_CORRECTION", 0.18))
    p.add_argument("--floor_clearance", type=int, default=_ei("V34_FLOOR_CLEARANCE_POSTPROCESS", 1))
    p.add_argument("--floor_margin", type=float, default=_ef("V34_FLOOR_MARGIN", 0.003))
    p.add_argument("--floor_strength", type=float, default=_ef("V34_FLOOR_STRENGTH", 0.90))
    p.add_argument("--floor_smooth_window", type=int, default=_ei("V34_FLOOR_SMOOTH_WINDOW", 7))
    p.add_argument("--floor_max_lift", type=float, default=_ef("V34_FLOOR_MAX_LIFT", 0.18))
    p.add_argument("--floor_support_damping", type=float, default=_ef("V34_FLOOR_SUPPORT_DAMPING", 0.25))
    p.add_argument("--smooth", type=int, default=_ei("V34_OUTPUT_SMOOTH", 0))
    p.add_argument("--rotation_smooth_window", type=int, default=3)
    p.add_argument("--smooth_strength", type=float, default=0.20)
    p.add_argument("--contact_denoise", type=int, default=_ei("V38_CONTACT_DENOISE", 1))
    p.add_argument("--contact_median_size", type=int, default=_ei("V38_CONTACT_MEDIAN_SIZE", 5))
    p.add_argument("--contact_close_holes", type=int, default=_ei("V38_CONTACT_CLOSE_HOLES", 7))
    p.add_argument("--contact_open_spikes", type=int, default=_ei("V38_CONTACT_OPEN_SPIKES", 4))
    p.add_argument("--contact_lock_blend_frames", type=int, default=_ei("V38_CONTACT_LOCK_BLEND_FRAMES", 6))
    p.add_argument("--butterworth_filter", type=int, default=_ei("V38_BUTTERWORTH_FILTER", 1))
    p.add_argument("--butterworth_cutoff_hz", type=float, default=_ef("V38_BUTTERWORTH_CUTOFF_HZ", 4.2))
    p.add_argument("--butterworth_order", type=int, default=_ei("V38_BUTTERWORTH_ORDER", 2))
    p.add_argument("--butterworth_strength", type=float, default=_ef("V38_BUTTERWORTH_STRENGTH", 0.78))
    p.add_argument("--butterworth_joints", default=os.getenv("V38_BUTTERWORTH_JOINTS", "lelbow,relbow,lwrist,rwrist,lhand,rhand,lshoulder,rshoulder"))
    p.add_argument("--butterworth_residual_only", type=int, default=_ei("V39_BUTTERWORTH_RESIDUAL_ONLY", 1))
    process_file(p.parse_args())


if __name__ == "__main__":
    main()
