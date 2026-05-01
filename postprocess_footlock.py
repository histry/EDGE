"""
Contact-aware foot-lock postprocess for EDGE 151-D motion.

V2 fix:
- Do NOT require low foot speed to detect contact. The previous version used
  height AND low-speed; for already-sliding feet this produced no contact
  segments, so no correction was applied.
- Use height-based contact as the primary mask, matching eval_quantitative.py's
  height contact behavior when channel contacts are unreliable.
- Add debug values so you can verify whether the postprocess really changed root.

Compatible with generate_controlled.py integration:
    from postprocess_footlock import foot_lock_root_correction, blend_back_to_trajectory
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


ROOT_X_IDX = 4
ROOT_Z_IDX = 6
FOOT_JOINTS = [7, 8, 10, 11]


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
    return np.stack(
        [
            np.interp(x_new, x_old, traj[:, 0]),
            np.interp(x_new, x_old, traj[:, 1]),
        ],
        axis=1,
    ).astype(np.float32)


def motion_to_joints(motion: np.ndarray, device: str = "cpu") -> np.ndarray:
    device_t = torch.device(device)
    root = torch.from_numpy(motion[:, 4:7]).float().to(device_t).unsqueeze(0)
    rot6d = (
        torch.from_numpy(motion[:, 7:151])
        .float()
        .to(device_t)
        .reshape(1, motion.shape[0], 24, 6)
    )

    with torch.no_grad():
        q_ax = ax_from_6v(rot6d)
        joints = SMPLSkeleton(device=device_t).forward(q_ax, root)
    return joints.detach().cpu().numpy()[0].astype(np.float32)


def smooth_signal(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x
    if window % 2 == 0:
        window += 1
    window = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if window <= 1:
        return x

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        y[:, c] = np.convolve(
            np.pad(x[:, c], (pad, pad), mode="edge"),
            kernel,
            mode="valid",
        )
    return y.astype(np.float32)


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
    speed_threshold: float = 0.08,
    fps: float = 30.0,
    min_contact_ratio: float = 0.02,
) -> np.ndarray:
    """
    Contact detector used for postprocess.

    Important V2 behavior:
    - Primary contact = low foot height, matching eval_quantitative.py height contact.
    - Speed is only used as a *preferred* stricter mask. If it would remove nearly
      all contacts, we fall back to height-only. This is crucial because a sliding
      foot is exactly a low-height but high-speed foot.
    """
    feet = joints[:, FOOT_JOINTS, :]  # [T,4,3]
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    low = heights <= floor + float(height_threshold)

    if speed_threshold is None or float(speed_threshold) <= 0:
        return low

    horizontal_speed = np.zeros((len(feet), 4), dtype=np.float32)
    if len(feet) > 1:
        horizontal_speed[1:] = (
            np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
            * float(fps)
        )

    slow = horizontal_speed <= float(speed_threshold)
    strict = low & slow

    # If strict contact is too sparse, it cannot fix sliding. Fall back to height-only.
    if float(strict.mean()) < float(min_contact_ratio):
        return low

    # Per-foot fallback: if a foot loses all contact under speed gating, keep height contact.
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
    """
    Approximate foot locking by correcting root X/Z during detected contacts.

    This is not full IK. It is a conservative display postprocess:
    - detects support-foot segments by height,
    - tries to keep support foot near its segment median X/Z,
    - applies the average needed correction to root X/Z,
    - smooths and clips correction to avoid violent jumps.
    """
    motion = motion.copy().astype(np.float32)
    joints_before = motion_to_joints(motion, device=device)
    contacts = height_contact_mask(
        joints_before,
        height_threshold=height_threshold,
        speed_threshold=speed_threshold,
        fps=fps,
    )

    feet_xz = joints_before[:, FOOT_JOINTS, :][:, :, [0, 2]]
    correction_sum = np.zeros((len(motion), 2), dtype=np.float32)
    correction_count = np.zeros((len(motion), 1), dtype=np.float32)

    segment_count = 0
    for foot_i in range(4):
        for start, end in contiguous_segments(contacts[:, foot_i], min_len=min_contact_len):
            segment_count += 1
            segment_pos = feet_xz[start:end, foot_i, :]
            lock_pos = np.median(segment_pos, axis=0)
            correction = lock_pos[None, :] - segment_pos
            correction_sum[start:end] += correction.astype(np.float32)
            correction_count[start:end] += 1.0

    correction = np.zeros((len(motion), 2), dtype=np.float32)
    valid = correction_count[:, 0] > 0
    if valid.any():
        correction[valid] = correction_sum[valid] / np.maximum(correction_count[valid], 1.0)

    correction = smooth_signal(correction, smooth_window)

    max_correction_m = float(max_correction_m)
    if max_correction_m > 0:
        correction = np.clip(correction, -max_correction_m, max_correction_m)

    lock_strength = float(np.clip(lock_strength, 0.0, 1.0))
    root_before = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    motion[:, ROOT_X_IDX] += lock_strength * correction[:, 0]
    motion[:, ROOT_Z_IDX] += lock_strength * correction[:, 1]
    root_after = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()

    changed = np.abs(root_after - root_before)
    debug = {
        "contact_ratio_used": float(contacts.mean()),
        "contact_segment_count": int(segment_count),
        "mean_abs_root_change_m": float(changed.mean()),
        "max_abs_root_change_m": float(changed.max()),
        "mean_abs_raw_correction_m": float(np.abs(correction).mean()),
        "max_abs_raw_correction_m": float(np.abs(correction).max()),
    }
    return motion.astype(np.float32), debug


def blend_back_to_trajectory(
    motion: np.ndarray,
    target_traj: np.ndarray,
    traj_keep: float = 0.35,
    keep_endpoints: bool = True,
) -> np.ndarray:
    """
    Softly blend root X/Z back toward target trajectory.

    For foot-lock, do not use a value close to 1.0. A high traj_keep erases the
    foot-lock correction. Recommended range:
    - 0.00: strongest foot lock, largest trajectory drift
    - 0.25~0.50: practical compromise
    - 0.80~1.00: mostly trajectory, little foot-lock effect
    """
    motion = motion.copy().astype(np.float32)
    target = resample_trajectory(target_traj, len(motion))
    traj_keep = float(np.clip(traj_keep, 0.0, 1.0))

    root_xz = motion[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    blended = (1.0 - traj_keep) * root_xz + traj_keep * target

    if keep_endpoints:
        blended[0] = target[0]
        blended[-1] = target[-1]

    motion[:, ROOT_X_IDX] = blended[:, 0]
    motion[:, ROOT_Z_IDX] = blended[:, 1]
    return motion.astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="EDGE contact-aware foot-lock postprocess V2")
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
        corrected = blend_back_to_trajectory(
            corrected,
            target_traj=target_traj,
            traj_keep=args.traj_keep,
            keep_endpoints=not args.no_keep_endpoints,
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, corrected.astype(np.float32))
    print(f"✅ saved: {args.out}")
    print("debug:", debug)


if __name__ == "__main__":
    main()
