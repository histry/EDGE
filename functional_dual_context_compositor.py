#!/usr/bin/env python3
"""Functional dual-context compositor.

Unlike naive "lower walks, upper dances", this compositor aligns both contexts
around the same support-event frames:
  - support units: lower/contact support and weight-shift timing
  - expressive-mobile units: torso/upper response during those support events

Root X/Z is preserved from the base motion by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

from footstep_phase_utils import (
    as_t151,
    CONTACT_SLICE,
    LOWER_ROT_INDEX,
    TORSO_ROT_INDEX,
    UPPER_ROT_INDEX,
    ROOT_X_IDX,
    ROOT_Z_IDX,
)


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def parse_frames(text: str) -> List[int]:
    return [int(round(float(x))) for x in parse_list(text)]


def smooth_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = np.hanning(n).astype(np.float32)
    if float(w.max()) > 1e-8:
        w = w / float(w.max())
    return w


def renorm_6d(motion: np.ndarray) -> np.ndarray:
    out = motion.copy()
    rot = out[:, 7:151].reshape(out.shape[0], 24, 6)
    a1, a2 = rot[..., :3], rot[..., 3:6]
    b1 = a1 / np.clip(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8, None)
    proj = (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.clip(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8, None)
    rot[..., :3] = b1
    rot[..., 3:6] = b2
    out[:, 7:151] = rot.reshape(out.shape[0], -1)
    return out.astype(np.float32)


def blend_indices(base, unit, frame, window, indices, strength):
    if strength <= 0 or len(indices) == 0:
        return
    T = len(base)
    unit = as_t151(unit)
    center = len(unit) // 2
    half = int(window) // 2
    start = max(0, int(frame) - half)
    end = min(T, int(frame) + half + 1)
    if end <= start:
        return
    u_start = max(0, center - (int(frame) - start))
    u_end = min(len(unit), u_start + (end - start))
    seg_len = min(end - start, u_end - u_start)
    if seg_len <= 0:
        return
    end = start + seg_len
    u_end = u_start + seg_len
    alpha = np.clip(smooth_window(seg_len)[:, None] * float(strength), 0.0, 1.0)
    base[start:end, indices] = (1 - alpha) * base[start:end, indices] + alpha * unit[u_start:u_end, indices]


def blend_contacts(base, unit, frame, window, strength):
    if strength <= 0:
        return
    T = len(base)
    unit = as_t151(unit)
    center = len(unit) // 2
    half = int(window) // 2
    start = max(0, int(frame) - half)
    end = min(T, int(frame) + half + 1)
    if end <= start:
        return
    u_start = max(0, center - (int(frame) - start))
    u_end = min(len(unit), u_start + (end - start))
    seg_len = min(end - start, u_end - u_start)
    if seg_len <= 0:
        return
    end = start + seg_len
    u_end = u_start + seg_len
    alpha = np.clip(smooth_window(seg_len)[:, None] * float(strength), 0.0, 1.0)
    base[start:end, CONTACT_SLICE] = (1 - alpha) * base[start:end, CONTACT_SLICE] + alpha * unit[u_start:u_end, CONTACT_SLICE]
    base[start:end, CONTACT_SLICE] = np.clip(base[start:end, CONTACT_SLICE], 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--support_units", required=True)
    ap.add_argument("--expressive_units", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--support_lower_strength", type=float, default=0.85)
    ap.add_argument("--support_contact_strength", type=float, default=0.85)
    ap.add_argument("--support_torso_strength", type=float, default=0.05)
    ap.add_argument("--expressive_torso_strength", type=float, default=0.45)
    ap.add_argument("--expressive_upper_strength", type=float, default=0.55)
    ap.add_argument("--expressive_lower_strength", type=float, default=0.05)
    ap.add_argument("--no_renorm_6d", action="store_true")
    ap.add_argument("--allow_root_blend", action="store_true")
    args = ap.parse_args()

    base = as_t151(np.load(args.base, allow_pickle=True)).astype(np.float32).copy()
    root_xz_before = base[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()

    support_units = parse_list(args.support_units)
    expressive_units = parse_list(args.expressive_units)
    frames = parse_frames(args.frames)

    n = min(len(frames), len(support_units), len(expressive_units))
    if n <= 0:
        raise ValueError("Need at least one frame/support_unit/expressive_unit")

    audit = []
    for i in range(n):
        frame = frames[i]
        sup = as_t151(np.load(support_units[i], allow_pickle=True)).astype(np.float32)
        exp = as_t151(np.load(expressive_units[i], allow_pickle=True)).astype(np.float32)

        # Support context: when/how to support.
        blend_indices(base, sup, frame, args.window, LOWER_ROT_INDEX, args.support_lower_strength)
        blend_indices(base, sup, frame, args.window, TORSO_ROT_INDEX, args.support_torso_strength)
        blend_contacts(base, sup, frame, args.window, args.support_contact_strength)

        # Expressive-mobile context: how torso/arms respond during the same support event.
        blend_indices(base, exp, frame, args.window, TORSO_ROT_INDEX, args.expressive_torso_strength)
        blend_indices(base, exp, frame, args.window, UPPER_ROT_INDEX, args.expressive_upper_strength)
        blend_indices(base, exp, frame, args.window, LOWER_ROT_INDEX, args.expressive_lower_strength)

        audit.append({
            "frame": int(frame),
            "support_unit": support_units[i],
            "expressive_unit": expressive_units[i],
        })

    if not args.allow_root_blend:
        base[:, [ROOT_X_IDX, ROOT_Z_IDX]] = root_xz_before

    if not args.no_renorm_6d:
        base = renorm_6d(base)

    if not args.allow_root_blend:
        base[:, [ROOT_X_IDX, ROOT_Z_IDX]] = root_xz_before

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, base.astype(np.float32))

    report = {
        "base": args.base,
        "out": str(out),
        "window": args.window,
        "support_lower_strength": args.support_lower_strength,
        "support_contact_strength": args.support_contact_strength,
        "support_torso_strength": args.support_torso_strength,
        "expressive_torso_strength": args.expressive_torso_strength,
        "expressive_upper_strength": args.expressive_upper_strength,
        "expressive_lower_strength": args.expressive_lower_strength,
        "allow_root_blend": bool(args.allow_root_blend),
        "events": audit,
    }
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ Functional dual-context composited motion saved: {out}")
    print(f"   report={out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
