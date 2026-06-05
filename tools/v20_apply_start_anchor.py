#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT = slice(7, 151)

def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--start_pose", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blend_frames", type=int, default=8)
    args = ap.parse_args()

    x = np.load(args.motion).astype(np.float32)
    batched = x.ndim == 3
    if batched:
        x = x[0]
    if x.ndim != 2 or x.shape[-1] != 151:
        raise ValueError(f"motion should be [T,151] or [1,T,151], got {x.shape}")

    s = np.load(args.start_pose).astype(np.float32).reshape(-1)
    if s.shape[0] != 151:
        raise ValueError(f"start_pose should be [151], got {s.shape}")

    y = x.copy()
    T = len(y)
    bf = max(1, min(args.blend_frames, T))

    # 第 0 帧严格一致
    y[0, CONTACT] = s[CONTACT]
    y[0, ROOT_Y] = s[ROOT_Y]
    y[0, ROT] = s[ROT]

    # root X/Z 仍然保持原地
    y[:, ROOT_X] = 0.0
    y[:, ROOT_Z] = 0.0

    # 前几帧从统一 start pose 软过渡到原计划动作
    for t in range(1, bf):
        a = smoothstep(t / max(bf - 1, 1))
        y[t, ROOT_Y] = (1 - a) * s[ROOT_Y] + a * y[t, ROOT_Y]
        y[t, ROT] = (1 - a) * s[ROT] + a * y[t, ROT]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, y[None].astype(np.float32) if batched else y.astype(np.float32))
    print("saved:", out)

if __name__ == "__main__":
    main()
