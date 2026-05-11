#!/usr/bin/env python3
"""Stage-3 practical decoupled merge utility.

This is the safe first implementation of decoupled upper/lower generation:
- Take a lower/root/contact motion from Stage 1 or gait-phase model.
- Take an upper-body Text/Pose Context RAG motion from Stage 2.
- Merge root/lower/contact from lower source and torso/upper from upper source.

It does not replace a full diffusion inpainting pass, but gives a deterministic
validation target and is useful for ablation videos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from footstep_phase_utils import CONTACT_SLICE, LOWER_ROT_INDEX, ROOT_X_IDX, ROOT_Y_IDX, ROOT_Z_IDX, TORSO_ROT_INDEX, UPPER_ROT_INDEX, as_t151
from segment_lower_body_compositor import renormalize_6d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lower_motion", required=True, help="Motion providing contacts/root/lower [T,151]")
    parser.add_argument("--upper_motion", required=True, help="Motion providing torso/upper style [T,151]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--torso_from", choices=["lower", "upper", "blend"], default="blend")
    parser.add_argument("--torso_blend", type=float, default=0.35, help="When torso_from=blend, fraction from lower")
    parser.add_argument("--keep_root_y_from", choices=["lower", "upper"], default="lower")
    args = parser.parse_args()

    lower = as_t151(np.load(args.lower_motion, allow_pickle=True)).astype(np.float32)
    upper = as_t151(np.load(args.upper_motion, allow_pickle=True)).astype(np.float32)
    T = min(len(lower), len(upper))
    lower, upper = lower[:T], upper[:T]

    out = upper.copy()
    # Always trust lower pass for contact/root/lower rotations.
    out[:, CONTACT_SLICE] = lower[:, CONTACT_SLICE]
    out[:, [ROOT_X_IDX, ROOT_Z_IDX]] = lower[:, [ROOT_X_IDX, ROOT_Z_IDX]]
    out[:, LOWER_ROT_INDEX] = lower[:, LOWER_ROT_INDEX]
    if args.keep_root_y_from == "lower":
        out[:, ROOT_Y_IDX] = lower[:, ROOT_Y_IDX]

    if args.torso_from == "lower":
        out[:, TORSO_ROT_INDEX] = lower[:, TORSO_ROT_INDEX]
    elif args.torso_from == "blend":
        a = float(np.clip(args.torso_blend, 0.0, 1.0))
        out[:, TORSO_ROT_INDEX] = a * lower[:, TORSO_ROT_INDEX] + (1.0 - a) * upper[:, TORSO_ROT_INDEX]
    # else torso stays from upper.

    out = renormalize_6d(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out.astype(np.float32))
    report = {
        "lower_motion": args.lower_motion,
        "upper_motion": args.upper_motion,
        "out": str(out_path),
        "torso_from": args.torso_from,
        "torso_blend": args.torso_blend,
        "keep_root_y_from": args.keep_root_y_from,
        "frames": int(T),
    }
    out_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote decoupled merged motion: {out_path}")
    print(f"   report={out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
