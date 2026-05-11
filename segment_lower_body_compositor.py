#!/usr/bin/env python3
"""Stage-1 validation: segment-level lower-body compositor.

Blend retrieved motion-unit lower-body rotations into an existing generated base
motion. This directly tests whether selected units can solve the current
"root follows S trajectory but body stays frozen" failure mode without retraining.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from footstep_phase_utils import (
    CONTACT_SLICE,
    LOWER_ROT_INDEX,
    ROOT_X_IDX,
    ROOT_Z_IDX,
    TORSO_ROT_INDEX,
    UPPER_ROT_INDEX,
    as_t151,
)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_frames(text: str) -> List[int]:
    return [int(round(float(x))) for x in parse_list(text)]


def smooth_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    # Hann with non-zero center. Good enough for segment blending.
    return np.hanning(n).astype(np.float32)


def renormalize_6d(motion: np.ndarray) -> np.ndarray:
    out = motion.copy()
    rot = out[:, 7:151].reshape(out.shape[0], 24, 6)
    a1 = rot[..., 0:3]
    a2 = rot[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    rot[..., 0:3] = b1
    rot[..., 3:6] = b2
    out[:, 7:151] = rot.reshape(out.shape[0], -1)
    return out.astype(np.float32)


def blend_segment(base: np.ndarray, unit: np.ndarray, frame: int, window: int, lower_strength: float, torso_strength: float, upper_strength: float, contact_strength: float, keep_root_xz: bool = True):
    T = base.shape[0]
    unit = as_t151(unit)
    center = len(unit) // 2
    half = max(1, int(window) // 2)
    start = max(0, int(frame) - half)
    end = min(T, int(frame) + half + 1)
    if end <= start:
        return base
    u_start = max(0, center - (int(frame) - start))
    u_end = min(len(unit), u_start + (end - start))
    seg_len = min(end - start, u_end - u_start)
    if seg_len <= 0:
        return base
    end = start + seg_len
    u_end = u_start + seg_len
    w = smooth_window(seg_len)[:, None]
    if float(w.max()) > 1e-8:
        w = w / float(w.max())

    def apply(indices, strength):
        if len(indices) == 0 or strength <= 0:
            return
        alpha = np.clip(w * float(strength), 0.0, 1.0)
        base[start:end, indices] = (1.0 - alpha) * base[start:end, indices] + alpha * unit[u_start:u_end, indices]

    apply(LOWER_ROT_INDEX, lower_strength)
    apply(TORSO_ROT_INDEX, torso_strength)
    apply(UPPER_ROT_INDEX, upper_strength)

    # Contacts are not rotations; blend them conservatively and clip.
    if contact_strength > 0:
        alpha = np.clip(w * float(contact_strength), 0.0, 1.0)
        base[start:end, CONTACT_SLICE] = (1.0 - alpha) * base[start:end, CONTACT_SLICE] + alpha * unit[u_start:u_end, CONTACT_SLICE]
        base[start:end, CONTACT_SLICE] = np.clip(base[start:end, CONTACT_SLICE], 0.0, 1.0)

    if keep_root_xz:
        # Explicitly preserve base S-curve trajectory.
        pass
    else:
        alpha = np.clip(w * min(float(lower_strength), 0.35), 0.0, 1.0)
        for idx in [ROOT_X_IDX, ROOT_Z_IDX]:
            base[start:end, idx:idx+1] = (1.0 - alpha) * base[start:end, idx:idx+1] + alpha * unit[u_start:u_end, idx:idx+1]
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Base generated motion .npy [T,151]")
    parser.add_argument("--unit_files", required=True, help="Comma-separated *_unit.npy files")
    parser.add_argument("--frames", required=True, help="Comma-separated center frames")
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", type=int, default=45)
    parser.add_argument("--lower_strength", type=float, default=0.85)
    parser.add_argument("--torso_strength", type=float, default=0.25)
    parser.add_argument("--upper_strength", type=float, default=0.0)
    parser.add_argument("--contact_strength", type=float, default=0.75)
    parser.add_argument("--allow_root_blend", action="store_true")
    parser.add_argument("--no_renorm_6d", action="store_true")
    args = parser.parse_args()

    base_raw = np.load(args.base, allow_pickle=True)
    base = as_t151(base_raw).astype(np.float32).copy()
    root_xz_before = base[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()
    unit_files = parse_list(args.unit_files)
    frames = parse_frames(args.frames)
    if len(unit_files) != len(frames):
        raise ValueError(f"unit_files count {len(unit_files)} != frames count {len(frames)}")

    audit = []
    for unit_file, frame in zip(unit_files, frames):
        unit = as_t151(np.load(unit_file, allow_pickle=True)).astype(np.float32)
        blend_segment(
            base,
            unit,
            frame=frame,
            window=args.window,
            lower_strength=args.lower_strength,
            torso_strength=args.torso_strength,
            upper_strength=args.upper_strength,
            contact_strength=args.contact_strength,
            keep_root_xz=not args.allow_root_blend,
        )
        audit.append({"unit_file": unit_file, "frame": int(frame), "unit_len": int(len(unit))})

    if not args.no_renorm_6d:
        base = renormalize_6d(base)
    if not args.allow_root_blend:
        base[:, [ROOT_X_IDX, ROOT_Z_IDX]] = root_xz_before

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, base.astype(np.float32))
    report = {
        "base": args.base,
        "out": str(out),
        "window": args.window,
        "lower_strength": args.lower_strength,
        "torso_strength": args.torso_strength,
        "upper_strength": args.upper_strength,
        "contact_strength": args.contact_strength,
        "allow_root_blend": bool(args.allow_root_blend),
        "segments": audit,
    }
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote composited motion: {out}")
    print(f"   report={out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
