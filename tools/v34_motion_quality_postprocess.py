#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process V34 151D motions for contact locking and light jitter control.

This tool is intentionally conservative: it does not change event selection or
network weights.  It only applies a root-translation correction when a foot is
already marked as contacting the ground, then optionally smooths high-frequency
rotation noise with a short symmetric filter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


PARENTS = np.array([
    -1, 0, 0, 0,
     1, 2, 3,
     4, 5, 6,
     7, 8, 9,
     9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21,
], dtype=np.int64)

OFFSETS = np.array([
    [ 0.00,  0.00,  0.00],
    [-0.10, -0.10,  0.00],
    [ 0.10, -0.10,  0.00],
    [ 0.00,  0.13,  0.00],
    [ 0.00, -0.42,  0.00],
    [ 0.00, -0.42,  0.00],
    [ 0.00,  0.14,  0.00],
    [ 0.00, -0.40,  0.00],
    [ 0.00, -0.40,  0.00],
    [ 0.00,  0.14,  0.00],
    [ 0.00, -0.08,  0.12],
    [ 0.00, -0.08,  0.12],
    [ 0.00,  0.14,  0.00],
    [-0.10,  0.08,  0.00],
    [ 0.10,  0.08,  0.00],
    [ 0.00,  0.16,  0.00],
    [-0.18,  0.00,  0.00],
    [ 0.18,  0.00,  0.00],
    [-0.28,  0.00,  0.00],
    [ 0.28,  0.00,  0.00],
    [-0.25,  0.00,  0.00],
    [ 0.25,  0.00,  0.00],
    [-0.08,  0.00,  0.00],
    [ 0.08,  0.00,  0.00],
], dtype=np.float32)

FOOT_JOINTS = np.array([7, 8, 10, 11], dtype=np.int64)


def _load_motion(path: Path) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        obj = x.item()
        x = obj.get("motion", obj.get("pose", x))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x


def _rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def _fk_from_t151(motion: np.ndarray) -> np.ndarray:
    t = motion.shape[0]
    root = motion[:, [4, 5, 6]].astype(np.float32)
    local_r = _rot6d_to_matrix(motion[:, 7:151].reshape(t, 24, 6))
    joints = np.zeros((t, 24, 3), dtype=np.float32)
    global_r = np.zeros((t, 24, 3, 3), dtype=np.float32)
    joints[:, 0] = root
    global_r[:, 0] = local_r[:, 0]
    for j in range(1, 24):
        p = int(PARENTS[j])
        global_r[:, j] = np.matmul(global_r[:, p], local_r[:, j])
        off = OFFSETS[j][None, :, None]
        joints[:, j] = joints[:, p] + np.matmul(global_r[:, p], off)[..., 0]
    return joints


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32, copy=True)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    flat = x.reshape(x.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[1]):
        out[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return out.reshape(x.shape).astype(np.float32)


def _segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.astype(bool).tolist() + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                rows.append((start, i))
            start = None
    return rows


def _contact_mask(motion: np.ndarray, threshold: float) -> np.ndarray:
    c = np.asarray(motion[:, 0:4], dtype=np.float32)
    if c.shape[1] != 4:
        return np.zeros((len(motion), 4), dtype=bool)
    return c >= float(threshold)


def _mean_contact_speed(joints: np.ndarray, contact: np.ndarray) -> float:
    feet = joints[:, FOOT_JOINTS, :]
    speed = np.zeros(feet.shape[:2], dtype=np.float32)
    speed[1:] = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1)
    active = contact.astype(bool)
    if not np.any(active):
        return 0.0
    return float(np.mean(speed[active]))


def contact_lock_root(
    motion: np.ndarray,
    *,
    contact_threshold: float,
    min_contact_frames: int,
    strength: float,
    smooth_window: int,
    max_correction: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    out = motion.astype(np.float32, copy=True)
    joints = _fk_from_t151(out)
    contact = _contact_mask(out, contact_threshold)
    correction = np.zeros((len(out), 2), dtype=np.float32)
    counts = np.zeros((len(out), 1), dtype=np.float32)
    segment_count = 0

    feet = joints[:, FOOT_JOINTS, :]
    for local_foot in range(4):
        for start, end in _segments(contact[:, local_foot], min_contact_frames):
            segment = feet[start:end, local_foot, :][:, [0, 2]]
            anchor = np.median(segment, axis=0)
            correction[start:end] += anchor[None, :] - segment
            counts[start:end] += 1.0
            segment_count += 1

    active = counts[:, 0] > 0
    correction[active] /= np.maximum(counts[active], 1.0)
    if max_correction > 0:
        norm = np.linalg.norm(correction, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_correction) / np.maximum(norm, 1e-8))
        correction *= scale
    correction = _moving_average(correction, smooth_window)
    out[:, 4] += float(strength) * correction[:, 0]
    out[:, 6] += float(strength) * correction[:, 1]

    after_joints = _fk_from_t151(out)
    summary = {
        "contact_segments": int(segment_count),
        "active_contact_frames": int(np.sum(active)),
        "mean_contact_speed_before": _mean_contact_speed(joints, contact),
        "mean_contact_speed_after": _mean_contact_speed(after_joints, contact),
        "max_root_xz_correction": float(np.max(np.linalg.norm(correction, axis=1))) if len(out) else 0.0,
        "contact_threshold": float(contact_threshold),
        "strength": float(strength),
    }
    return out, summary


def smooth_motion(
    motion: np.ndarray,
    *,
    rotation_window: int,
    root_y_window: int,
    strength: float,
) -> np.ndarray:
    out = motion.astype(np.float32, copy=True)
    strength = float(np.clip(strength, 0.0, 1.0))
    if rotation_window > 1 and strength > 0:
        smoothed = _moving_average(out[:, 7:151], rotation_window)
        out[:, 7:151] = (1.0 - strength) * out[:, 7:151] + strength * smoothed
    if root_y_window > 1 and strength > 0:
        smoothed_y = _moving_average(out[:, 5:6], root_y_window)
        out[:, 5:6] = (1.0 - strength) * out[:, 5:6] + strength * smoothed_y
    return out


def process_file(args: argparse.Namespace) -> Dict[str, object]:
    src = Path(args.motion)
    out_path = Path(args.out)
    motion = _load_motion(src)
    original = motion.copy()

    contact_summary: Dict[str, object] = {"enabled": False}
    if args.contact_lock:
        motion, contact_summary = contact_lock_root(
            motion,
            contact_threshold=args.contact_threshold,
            min_contact_frames=args.min_contact_frames,
            strength=args.contact_lock_strength,
            smooth_window=args.contact_smooth_window,
            max_correction=args.max_root_correction,
        )
        contact_summary["enabled"] = True

    if args.smooth:
        motion = smooth_motion(
            motion,
            rotation_window=args.rotation_smooth_window,
            root_y_window=args.root_y_smooth_window,
            strength=args.smooth_strength,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, motion.astype(np.float32))

    summary = {
        "version": "v34_motion_quality_postprocess",
        "input": str(src),
        "output": str(out_path),
        "frames": int(len(motion)),
        "contact_lock": contact_summary,
        "smooth": {
            "enabled": bool(args.smooth),
            "rotation_window": int(args.rotation_smooth_window),
            "root_y_window": int(args.root_y_smooth_window),
            "strength": float(args.smooth_strength),
        },
        "root_xz_delta_mean": float(np.mean(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
        "root_xz_delta_max": float(np.max(np.linalg.norm(motion[:, [4, 6]] - original[:, [4, 6]], axis=1))) if len(motion) else 0.0,
    }
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--contact_lock", type=int, default=1)
    parser.add_argument("--contact_threshold", type=float, default=0.65)
    parser.add_argument("--min_contact_frames", type=int, default=8)
    parser.add_argument("--contact_lock_strength", type=float, default=0.85)
    parser.add_argument("--contact_smooth_window", type=int, default=11)
    parser.add_argument("--max_root_correction", type=float, default=0.18)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--rotation_smooth_window", type=int, default=3)
    parser.add_argument("--root_y_smooth_window", type=int, default=5)
    parser.add_argument("--smooth_strength", type=float, default=0.35)
    args = parser.parse_args()
    process_file(args)


if __name__ == "__main__":
    main()
