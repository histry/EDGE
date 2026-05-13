#!/usr/bin/env python3
"""Functional dual-context compositor with turn-aware split frames.

Replacement version.

Compared with the old compositor, support and expressive contexts no longer
have to be centered at the same frame:
  support_frames    = turn event - lag, for support/weight-shift preparation
  expressive_frames = turn event + lag, for torso/upper response

This is a no-training diagnostic compositor. It preserves root X/Z by default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

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
from turn_aware_event_utils import parse_int_list


def parse_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).replace(";", ",").split(",") if x.strip()]


def smooth_window(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones((n,), dtype=np.float32)
    w = np.hanning(n).astype(np.float32)
    if float(w.max()) > 1e-8:
        w = w / float(w.max())
    return w


def renorm_6d(motion: np.ndarray) -> np.ndarray:
    out = np.asarray(motion, dtype=np.float32).copy()
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


def _span(T: int, unit_len: int, frame: int, window: int):
    center = unit_len // 2
    half = int(window) // 2
    start = max(0, int(frame) - half)
    end = min(T, int(frame) + half + 1)
    if end <= start:
        return None
    u_start = max(0, center - (int(frame) - start))
    u_end = min(unit_len, u_start + (end - start))
    seg_len = min(end - start, u_end - u_start)
    if seg_len <= 0:
        return None
    return start, start + seg_len, u_start, u_start + seg_len


def blend_indices(base: np.ndarray, unit: np.ndarray, frame: int, window: int, indices, strength: float) -> None:
    if strength <= 0 or len(indices) == 0:
        return
    unit = as_t151(unit).astype(np.float32)
    span = _span(len(base), len(unit), int(frame), int(window))
    if span is None:
        return
    start, end, u_start, u_end = span
    alpha = np.clip(smooth_window(end - start)[:, None] * float(strength), 0.0, 1.0)
    base[start:end, indices] = (1 - alpha) * base[start:end, indices] + alpha * unit[u_start:u_end, indices]


def blend_contacts(base: np.ndarray, unit: np.ndarray, frame: int, window: int, strength: float) -> None:
    if strength <= 0:
        return
    unit = as_t151(unit).astype(np.float32)
    span = _span(len(base), len(unit), int(frame), int(window))
    if span is None:
        return
    start, end, u_start, u_end = span
    alpha = np.clip(smooth_window(end - start)[:, None] * float(strength), 0.0, 1.0)
    base[start:end, CONTACT_SLICE] = (1 - alpha) * base[start:end, CONTACT_SLICE] + alpha * unit[u_start:u_end, CONTACT_SLICE]
    base[start:end, CONTACT_SLICE] = np.clip(base[start:end, CONTACT_SLICE], 0.0, 1.0)


def frames_from_args(common: str, split: str) -> List[int]:
    xs = parse_int_list(split)
    if xs:
        return xs
    return parse_int_list(common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--support_units", required=True)
    ap.add_argument("--expressive_units", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--support_frames", default="")
    ap.add_argument("--expressive_frames", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=45)
    ap.add_argument("--support_window", type=int, default=0, help="0 means use --window")
    ap.add_argument("--expressive_window", type=int, default=0, help="0 means use --window")
    ap.add_argument("--support_lower_strength", type=float, default=0.45)
    ap.add_argument("--support_contact_strength", type=float, default=0.55)
    ap.add_argument("--support_torso_strength", type=float, default=0.05)
    ap.add_argument("--expressive_torso_strength", type=float, default=0.20)
    ap.add_argument("--expressive_upper_strength", type=float, default=0.30)
    ap.add_argument("--expressive_lower_strength", type=float, default=0.05)
    ap.add_argument("--no_renorm_6d", action="store_true")
    ap.add_argument("--allow_root_blend", action="store_true")
    args = ap.parse_args()

    base = as_t151(np.load(args.base, allow_pickle=True)).astype(np.float32).copy()
    root_xz_before = base[:, [ROOT_X_IDX, ROOT_Z_IDX]].copy()

    support_units = parse_list(args.support_units)
    expressive_units = parse_list(args.expressive_units)
    support_frames = frames_from_args(args.frames, args.support_frames)
    expressive_frames = frames_from_args(args.frames, args.expressive_frames)
    if not support_frames or not expressive_frames:
        raise ValueError("Need --frames or both --support_frames/--expressive_frames")

    support_window = int(args.support_window or args.window)
    expressive_window = int(args.expressive_window or args.window)

    n = min(len(support_frames), len(expressive_frames), len(support_units), len(expressive_units))
    if n <= 0:
        raise ValueError("Need at least one support/expressive unit and frame")

    audit = []
    for i in range(n):
        sf = int(support_frames[i])
        ef = int(expressive_frames[i])
        sup = as_t151(np.load(support_units[i], allow_pickle=True)).astype(np.float32)
        exp = as_t151(np.load(expressive_units[i], allow_pickle=True)).astype(np.float32)

        # Support context: lower/contact before or around turning event.
        blend_indices(base, sup, sf, support_window, LOWER_ROT_INDEX, args.support_lower_strength)
        blend_indices(base, sup, sf, support_window, TORSO_ROT_INDEX, args.support_torso_strength)
        blend_contacts(base, sup, sf, support_window, args.support_contact_strength)

        # Expressive context: torso/upper response can be phase-shifted after support.
        blend_indices(base, exp, ef, expressive_window, TORSO_ROT_INDEX, args.expressive_torso_strength)
        blend_indices(base, exp, ef, expressive_window, UPPER_ROT_INDEX, args.expressive_upper_strength)
        blend_indices(base, exp, ef, expressive_window, LOWER_ROT_INDEX, args.expressive_lower_strength)

        audit.append(
            {
                "support_frame": sf,
                "expressive_frame": ef,
                "support_unit": support_units[i],
                "expressive_unit": expressive_units[i],
            }
        )

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
        "support_window": support_window,
        "expressive_window": expressive_window,
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
