#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="Manually verified Dunhuang-style [T,151] or [1,T,151] NPY")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    x = np.load(args.motion, allow_pickle=True).astype(np.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"Expected [T,151] or [1,T,151], got {x.shape}")
    frame = int(np.clip(args.frame, 0, len(x) - 1))
    pose = x[frame].copy()
    pose[4] = 0.0
    pose[6] = 0.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pose)
    print("saved:", out)
    print("source:", args.motion)
    print("frame:", frame)


if __name__ == "__main__":
    main()
