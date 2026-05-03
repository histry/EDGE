import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from vis import SMPLSkeleton, skeleton_render


def render_pkl_clip(args):
    with open(args.pkl, "rb") as f:
        d = pickle.load(f)

    if not isinstance(d, dict) or "pos" not in d or "q" not in d:
        raise ValueError(f"{args.pkl} 需要包含 pos 和 q")

    pos = np.asarray(d["pos"], dtype=np.float32)
    q = np.asarray(d["q"], dtype=np.float32)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos 应该是 [T,3]，当前 {pos.shape}")
    if q.ndim != 2 or q.shape[1] != 72:
        raise ValueError(f"q 应该是 [T,72]，当前 {q.shape}")

    T = len(pos)

    if args.center >= 0:
        start = max(0, args.center - args.pre)
        end = min(T, args.center + args.post)
    else:
        start = max(0, args.start)
        end = min(T, args.end)

    if end <= start:
        raise ValueError(f"非法区间: start={start}, end={end}, T={T}")

    root = pos[start:end].copy()
    rot = q[start:end].reshape(-1, 24, 3).copy()

    if args.body_centered:
        root[:, 0] = 0.0
        root[:, 2] = 0.0

    root_t = torch.from_numpy(root).float()[None]          # [1,T,3]
    rot_t = torch.from_numpy(rot).float()[None]            # [1,T,24,3]

    skel = SMPLSkeleton()
    with torch.no_grad():
        joints = skel.forward(rot_t, root_t)[0].cpu().numpy().astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skeleton_render(
        poses=joints,
        epoch=0,
        out=str(out_path.parent),
        name=args.music,
        sound=True,
        contact=None,
        render=True,
        camera_mode=args.camera_mode,
        output_path=str(out_path),
        render_smooth_window=args.smooth_window,
    )

    print("✅ rendered:", out_path)
    print("clip frames:", start, end, "len:", end - start)
    print("root min:", root.min(axis=0))
    print("root max:", root.max(axis=0))
    print("rot mean/std:", float(rot.mean()), float(rot.std()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--center", type=int, default=-1)
    parser.add_argument("--pre", type=int, default=20)
    parser.add_argument("--post", type=int, default=130)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=150)
    parser.add_argument("--camera_mode", default="fixed", choices=["fixed", "follow"])
    parser.add_argument("--body_centered", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()
    render_pkl_clip(args)


if __name__ == "__main__":
    main()
