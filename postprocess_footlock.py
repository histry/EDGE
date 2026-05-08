"""
Contact-aware foot-lock postprocess for EDGE 151-D motion.

V11.1 logic-gap fix:
- Keeps V3 contact-aware dynamic trajectory blending.
- Adds trajectory tolerance band so root X/Z is not forcibly dragged to the
  target during foot-contact frames when it is already close enough.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton

ROOT_X_IDX = 4
ROOT_Z_IDX = 6
FOOT_JOINTS = [7, 8, 10, 11]
_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


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


def load_motion(path: str) -> np.ndarray:
    motion = np.load(path, allow_pickle=True)
    if motion.ndim == 0 and isinstance(motion.item(), dict):
        data = motion.item()
        if "motion" not in data:
            raise ValueError(f"{path} is a dict .npy but has no 'motion' key")
        motion = data["motion"]
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape} from {path}")
    return motion.astype(np.float32)


def resample_trajectory(traj: np.ndarray, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[0]
    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"Expected trajectory [T,2] or [1,T,2], got {traj.shape}")
    traj = traj[:, :2]
    if len(traj) == target_len:
        return traj.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(traj))
    x_new = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(x_new, x_old, traj[:, 0]), np.interp(x_new, x_old, traj[:, 1])], axis=1).astype(np.float32)


def motion_to_joints(motion: np.ndarray, device: str = "cpu") -> np.ndarray:
    device_t = torch.device(device)
    root = torch.from_numpy(motion[:, 4:7]).float().to(device_t).unsqueeze(0)
    rot6d = torch.from_numpy(motion[:, 7:151]).float().to(device_t).reshape(1, motion.shape[0], 24, 6)
    with torch.no_grad():
        q_ax = ax_from_6v(rot6d)
        joints = SMPLSkeleton(device=device_t).forward(q_ax, root)
    return joints.detach().cpu().numpy()[0].astype(np.float32)


def smooth_signal(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    squeeze = False
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x[:, 0] if squeeze else x
    if window % 2 == 0:
        window += 1
    window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if window <= 1:
        return x[:, 0] if squeeze else x
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = np.convolve(np.pad(x[:, c], (pad, pad), mode="edge"), kernel, mode="valid")
    return y[:, 0].astype(np.float32) if squeeze else y.astype(np.float32)


def contiguous_segments(mask_1d: np.ndarray, min_len: int = 3) -> List[Tuple[int, int]]:
    mask_1d = np.asarray(mask_1d, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask_1d):
        if value and start is None:
            start = i
        elif (not value) and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None
    if start is not None and len(mask_1d) - start >= min_len:
        segments.append((start, len(mask_1d)))
    return segments


def height_contact_mask(
    joints: np.ndarray,
    height_threshold: float = 0.035,
    speed_threshold: Optional[float] = 0.08,
    fps: float = 30.0,
    min_contact_ratio: float = 0.02,
) -> np.ndarray:
    feet = joints[:, FOOT_JOINTS, :]
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    low = heights <= floor + float(height_threshold)
    if speed_threshold is None or float(speed_threshold) <= 0:
        return low
    horizontal_speed = np.zeros((len(feet), 4), dtype=np.float32)
    if len(feet) > 1:
        horizontal_speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1) * float(fps)
    slow = horizontal_speed <= float(speed_threshold)
    strict = low & slow
    if float(strict.mean()) < float(min_contact_ratio):
        return low
    out = strict.copy()
    for foot_i in range(4):
        if not out[:, foot_i].any() and low[:, foot_i].any():
            out[:, foot_i] = low[:, foot_i]
    return out


def foot_lock_root_correction(
    motion: np.ndarray,
    device: str = "cpu",
    fps: float = 30.0,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.08,
    min_contact_len: int = 3,
    lock_strength: float = 0.75,
    smooth_window: int = 9,
    max_correction_m: float = 0.35,
) -> Tuple[np.ndarray, Dict[str, float]]:
    motion = motion.copy().astype(np.float32)
    joints_before = motion_to_joints(motion, device=device)
    contacts = height_contact_mask(joints_before, height_threshold=height_threshold, speed_threshold=speed_threshold, fps=fps)
    feet_xz = joints_before[:, FOOT_JOINTS, :][:, :, [0, 2]]
    correction_sum = np.zeros((len(motion), 2), dtype=np.float32)
    correction_count = np.zeros((len(motion), 1), dtype=np.float32)
    segment_count = 0
    for foot_i in range(4):
        for start, end in contiguous_segments(contacts[:, foot_i], min_len=min_contact_len):
            segment_count += 1
            segment_pos = feet_xz[start:end, foot_i, :]
            lock_pos = np.median(segment_pos, axis=0)
            correction_sum[start:end] += (lock_pos[None, :] - segment_pos).astype(np.float32)
            correction_count[start:end] += 1.0
    correction = np.zeros((len(motion), 2), dtype=np.float32)
    valid = correction_count[:, 0] > 0
    if valid.any():
        correction[valid] = correction_sum[valid] / np.maximum(correction_count[valid], 1.0)
    correction = smooth_signal(correction, smooth_window)
    if float(max_correction_m) > 0:
        correction = np.clip(correction, -float(max_correction_m), float(max_correction_m))
    lock_strength = float(np.clip(lock_strength, 0.0, 1.0))
    root_before = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    motion[:, ROOT_X_IDX] += lock_strength * correction[:, 0]
    motion[:, ROOT_Z_IDX] += lock_strength * correction[:, 1]
    changed = np.abs(motion[:, [ROOT_X_IDX, ROOT_Z_IDX]] - root_before)
    debug = {
        "contact_ratio_used": float(contacts.mean()),
        "contact_segment_count": int(segment_count),
        "mean_abs_root_change_m": float(changed.mean()),
        "max_abs_root_change_m": float(changed.max()),
    }
    return motion.astype(np.float32), debug


def _apply_trajectory_tolerance(root_xz: np.ndarray, target: np.ndarray, contact_any: Optional[np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
    if not _env_bool("EDGE_TRAJ_TOLERANCE_ENABLE", True):
        return target, {"traj_tolerance_enabled": 0.0}
    tol_air = _env_float("EDGE_TRAJ_TOLERANCE_M", 0.04)
    tol_contact = _env_float("EDGE_TRAJ_TOLERANCE_CONTACT_M", 0.08)
    if contact_any is None:
        tol = np.full((len(root_xz), 1), tol_air, dtype=np.float32)
    else:
        tol = np.where(contact_any.astype(bool), tol_contact, tol_air).astype(np.float32)[:, None]
        tol = smooth_signal(tol, _env_int("EDGE_TRAJ_TOLERANCE_SMOOTH", 5))
        if tol.ndim == 1:
            tol = tol[:, None]
    delta = target - root_xz
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    direction = delta / np.maximum(dist, 1e-8)
    tolerated_target = root_xz + direction * np.maximum(dist - tol, 0.0)
    return tolerated_target.astype(np.float32), {
        "traj_tolerance_enabled": 1.0,
        "traj_tolerance_air_m": float(tol_air),
        "traj_tolerance_contact_m": float(tol_contact),
        "traj_target_delta_mean_before_m": float(dist.mean()),
        "traj_target_delta_mean_after_m": float(np.linalg.norm(tolerated_target - root_xz, axis=1).mean()),
    }


def _dynamic_keep_from_contacts(
    motion: np.ndarray,
    base_keep: float,
    joints: Optional[np.ndarray],
    device: str,
    height_threshold: float,
    speed_threshold: float,
    fps: float,
    smooth_window: int,
    contact_keep: Optional[float],
    air_keep: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if joints is None:
        joints = motion_to_joints(motion, device=device)
    contacts = height_contact_mask(joints, height_threshold=height_threshold, speed_threshold=speed_threshold, fps=fps)
    is_contact = contacts.any(axis=1)
    contact_keep_v = _env_float("EDGE_TRAJ_KEEP_CONTACT", 0.03 if contact_keep is None else float(contact_keep))
    air_keep_v = _env_float("EDGE_TRAJ_KEEP_AIR", float(base_keep) if air_keep is None else float(air_keep))
    dynamic_keep = np.where(is_contact, contact_keep_v, air_keep_v).astype(np.float32)[:, None]
    dynamic_keep = smooth_signal(dynamic_keep, window=smooth_window)
    dynamic_keep = np.clip(dynamic_keep, 0.0, 1.0).astype(np.float32)
    if dynamic_keep.ndim == 1:
        dynamic_keep = dynamic_keep[:, None]
    debug = {
        "dynamic_traj_blend": 1.0,
        "contact_ratio": float(is_contact.mean()),
        "traj_keep_contact": float(contact_keep_v),
        "traj_keep_air": float(air_keep_v),
        "traj_keep_mean": float(dynamic_keep.mean()),
    }
    return dynamic_keep, is_contact, debug


def blend_back_to_trajectory(
    motion: np.ndarray,
    target_traj: np.ndarray,
    traj_keep: float = 0.35,
    keep_endpoints: bool = True,
    joints: Optional[np.ndarray] = None,
    dynamic_contact_blend: Optional[bool] = None,
    contact_traj_keep: Optional[float] = None,
    air_traj_keep: Optional[float] = None,
    blend_smooth_window: Optional[int] = None,
    device: str = "cpu",
    fps: float = 30.0,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.08,
    return_debug: bool = False,
):
    motion = motion.copy().astype(np.float32)
    target = resample_trajectory(target_traj, len(motion))
    base_keep = float(np.clip(traj_keep, 0.0, 1.0))
    if dynamic_contact_blend is None:
        dynamic_contact_blend = _env_bool("EDGE_DYNAMIC_TRAJ_BLEND", False)
    smooth_window = _env_int("EDGE_TRAJ_BLEND_SMOOTH", 5 if blend_smooth_window is None else blend_smooth_window)
    debug: Dict[str, float] = {"dynamic_traj_blend": 0.0, "traj_keep_base": float(base_keep)}
    contact_any = None
    if dynamic_contact_blend:
        dynamic_keep, contact_any, debug_dyn = _dynamic_keep_from_contacts(
            motion, base_keep, joints, device, height_threshold, speed_threshold, fps,
            smooth_window, contact_traj_keep, air_traj_keep
        )
        debug.update(debug_dyn)
    else:
        dynamic_keep = np.full((len(motion), 1), base_keep, dtype=np.float32)
    root_xz = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    tolerated_target, tol_debug = _apply_trajectory_tolerance(root_xz, target, contact_any)
    debug.update(tol_debug)
    blended = (1.0 - dynamic_keep) * root_xz + dynamic_keep * tolerated_target
    if keep_endpoints:
        blended[0] = target[0]
        blended[-1] = target[-1]
    motion[:, ROOT_X_IDX] = blended[:, 0]
    motion[:, ROOT_Z_IDX] = blended[:, 1]
    if return_debug:
        return motion.astype(np.float32), debug
    return motion.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="EDGE contact-aware foot-lock postprocess V11.1")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target_traj", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--height_threshold", type=float, default=0.035)
    parser.add_argument("--speed_threshold", type=float, default=0.08)
    parser.add_argument("--min_contact_len", type=int, default=3)
    parser.add_argument("--lock_strength", type=float, default=0.85)
    parser.add_argument("--smooth_window", type=int, default=9)
    parser.add_argument("--max_correction_m", type=float, default=0.35)
    parser.add_argument("--traj_keep", type=float, default=0.35)
    parser.add_argument("--dynamic_traj_blend", action="store_true")
    parser.add_argument("--traj_keep_contact", type=float, default=None)
    parser.add_argument("--traj_keep_air", type=float, default=None)
    parser.add_argument("--traj_blend_smooth", type=int, default=5)
    parser.add_argument("--no_keep_endpoints", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    motion = load_motion(args.motion)
    corrected, debug = foot_lock_root_correction(
        motion,
        device=args.device,
        fps=args.fps,
        height_threshold=args.height_threshold,
        speed_threshold=args.speed_threshold,
        min_contact_len=args.min_contact_len,
        lock_strength=args.lock_strength,
        smooth_window=args.smooth_window,
        max_correction_m=args.max_correction_m,
    )
    if args.target_traj:
        target_traj = np.load(args.target_traj).astype(np.float32)
        corrected, blend_debug = blend_back_to_trajectory(
            corrected,
            target_traj=target_traj,
            traj_keep=args.traj_keep,
            keep_endpoints=not args.no_keep_endpoints,
            dynamic_contact_blend=args.dynamic_traj_blend or _env_bool("EDGE_DYNAMIC_TRAJ_BLEND", False),
            contact_traj_keep=args.traj_keep_contact,
            air_traj_keep=args.traj_keep_air,
            blend_smooth_window=args.traj_blend_smooth,
            device=args.device,
            fps=args.fps,
            height_threshold=args.height_threshold,
            speed_threshold=args.speed_threshold,
            return_debug=True,
        )
        debug.update(blend_debug)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, corrected.astype(np.float32))
    print(f"✅ saved: {args.out}")
    print("debug:", debug)


if __name__ == "__main__":
    main()
