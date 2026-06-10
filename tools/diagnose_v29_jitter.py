#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone V29 jitter diagnosis for an EDGE [T,151] motion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.v29_motion_geometry import jitter_statistics_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    motion = np.asarray(np.load(args.motion, allow_pickle=True), dtype=np.float32)
    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]
    result = {
        "version": "v29_jitter_diagnosis",
        "motion": args.motion,
        "frames": int(len(motion)),
        "fps": float(args.fps),
        **jitter_statistics_np(motion, fps=args.fps),
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
