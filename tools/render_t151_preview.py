#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


PARENTS = np.array([
    -1, 0, 0, 0,
     1, 2, 3,
     4, 5, 6,
     7, 8, 9,
     9, 9, 12,
    13, 14, 16, 17, 18, 19, 20, 21
], dtype=np.int64)

# SMPL-like approximate rest offsets. Only for preview rendering.
OFFSETS = np.array([
    [ 0.00,  0.00,  0.00],   # 0 pelvis
    [-0.10, -0.10,  0.00],   # 1 L hip
    [ 0.10, -0.10,  0.00],   # 2 R hip
    [ 0.00,  0.13,  0.00],   # 3 spine1
    [ 0.00, -0.42,  0.00],   # 4 L knee
    [ 0.00, -0.42,  0.00],   # 5 R knee
    [ 0.00,  0.14,  0.00],   # 6 spine2
    [ 0.00, -0.40,  0.00],   # 7 L ankle
    [ 0.00, -0.40,  0.00],   # 8 R ankle
    [ 0.00,  0.14,  0.00],   # 9 spine3
    [ 0.00, -0.08,  0.12],   # 10 L foot
    [ 0.00, -0.08,  0.12],   # 11 R foot
    [ 0.00,  0.14,  0.00],   # 12 neck
    [-0.10,  0.08,  0.00],   # 13 L collar
    [ 0.10,  0.08,  0.00],   # 14 R collar
    [ 0.00,  0.16,  0.00],   # 15 head
    [-0.18,  0.00,  0.00],   # 16 L shoulder
    [ 0.18,  0.00,  0.00],   # 17 R shoulder
    [-0.28,  0.00,  0.00],   # 18 L elbow
    [ 0.28,  0.00,  0.00],   # 19 R elbow
    [-0.25,  0.00,  0.00],   # 20 L wrist
    [ 0.25,  0.00,  0.00],   # 21 R wrist
    [-0.08,  0.00,  0.00],   # 22 L hand
    [ 0.08,  0.00,  0.00],   # 23 R hand
], dtype=np.float32)


EDGES = [(i, int(PARENTS[i])) for i in range(1, len(PARENTS))]


def load_motion(path: str) -> np.ndarray:
    x = np.load(path, allow_pickle=True)
    if x.ndim == 0 and isinstance(x.item(), dict):
        d = x.item()
        x = d.get("motion", d.get("pose", x))
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2 or x.shape[1] != 151:
        raise ValueError(f"{path}: expected [T,151], got {x.shape}")
    return x


def rot6d_to_matrix(x: np.ndarray) -> np.ndarray:
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1).astype(np.float32)


def fk_from_t151(motion: np.ndarray) -> np.ndarray:
    T = motion.shape[0]
    root = motion[:, [4, 5, 6]].astype(np.float32)
    rot6d = motion[:, 7:151].reshape(T, 24, 6)
    local_R = rot6d_to_matrix(rot6d)

    joints = np.zeros((T, 24, 3), dtype=np.float32)
    global_R = np.zeros((T, 24, 3, 3), dtype=np.float32)

    joints[:, 0] = root
    global_R[:, 0] = local_R[:, 0]

    for j in range(1, 24):
        p = int(PARENTS[j])
        global_R[:, j] = np.matmul(global_R[:, p], local_R[:, j])
        off = OFFSETS[j][None, :, None]
        joints[:, j] = joints[:, p] + np.matmul(global_R[:, p], off)[..., 0]

    return joints


def equal_axis_limits(joints: np.ndarray, root: np.ndarray):
    xyz = joints.reshape(-1, 3)
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)

    center = (lo + hi) / 2.0
    radius = max(float((hi - lo).max()) * 0.55, 1.0)

    # Keep X/Z broad enough for S trajectory.
    xlo, xhi = center[0] - radius, center[0] + radius
    ylo, yhi = max(0.0, center[1] - radius), center[1] + radius
    zlo, zhi = center[2] - radius, center[2] + radius

    root_lo = root.min(axis=0)
    root_hi = root.max(axis=0)
    xlo = min(xlo, float(root_lo[0]) - 0.5)
    xhi = max(xhi, float(root_hi[0]) + 0.5)
    zlo = min(zlo, float(root_lo[2]) - 0.5)
    zhi = max(zhi, float(root_hi[2]) + 0.5)

    return (xlo, xhi), (ylo, yhi), (zlo, zhi)


def render(input_path: str, out_path: str, title: str, fps: int = 20, audio: str = ""):
    motion = load_motion(input_path)
    joints = fk_from_t151(motion)
    root = joints[:, 0]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_out = out
    if audio:
        tmp_out = out.with_name(out.stem + "_silent.mp4")

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="3d")

    (xlim, ylim, zlim) = equal_axis_limits(joints, root)
    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in EDGES]
    root_line, = ax.plot([], [], [], linewidth=1.5, linestyle="--")
    current_root, = ax.plot([], [], [], marker="o", markersize=4)

    def init():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=18, azim=-70)
        try:
            ax.set_box_aspect((xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]))
        except Exception:
            pass
        return lines + [root_line, current_root]

    def update(t):
        J = joints[t]
        for line, (j, p) in zip(lines, EDGES):
            pts = J[[p, j]]
            line.set_data(pts[:, 0], pts[:, 2])
            line.set_3d_properties(pts[:, 1])

        path = root[:t+1]
        root_line.set_data(path[:, 0], path[:, 2])
        root_line.set_3d_properties(path[:, 1] * 0.0)

        current_root.set_data([root[t, 0]], [root[t, 2]])
        current_root.set_3d_properties([0.0])

        ax.set_title(f"{title} | frame {t+1}/{len(joints)}")
        return lines + [root_line, current_root]

    ani = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(joints),
        interval=1000 / fps,
        blit=False,
    )

    writer = FFMpegWriter(fps=fps, bitrate=2200)
    ani.save(str(tmp_out), writer=writer)
    plt.close(fig)

    if audio:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(tmp_out),
            "-i", audio,
            "-shortest",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        try:
            tmp_out.unlink()
        except Exception:
            pass

    print(f"✅ rendered: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--audio", default="")
    args = parser.parse_args()

    title = args.title or Path(args.input).stem
    render(args.input, args.out, title=title, fps=args.fps, audio=args.audio)


if __name__ == "__main__":
    main()
