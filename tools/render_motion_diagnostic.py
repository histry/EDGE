#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROT_START = 7
ROT_DIM = 6
LOWER = [1,2,4,5,7,8,10,11]
TORSO = [3,6,9]
UPPER = [12,13,14,15,16,17,18,19,20,21,22,23]

def idx(joints):
    out=[]
    for j in joints:
        out += list(range(ROT_START + ROT_DIM*j, ROT_START + ROT_DIM*j + ROT_DIM))
    return out

LOWER_IDX = idx(LOWER)
TORSO_IDX = idx(TORSO)
UPPER_IDX = idx(UPPER)

def load_motion(path):
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

def frame_energy(x, indices):
    d = np.zeros((len(x),), dtype=np.float32)
    d[1:] = np.sqrt(np.mean((x[1:, indices] - x[:-1, indices]) ** 2, axis=1))
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    m = load_motion(args.input)
    T = len(m)
    root = m[:, [4,6]]
    lower_e = frame_energy(m, LOWER_IDX)
    torso_e = frame_energy(m, TORSO_IDX)
    upper_e = frame_energy(m, UPPER_IDX)
    contact = (m[:, :4] > 0.5).astype(np.float32)
    contact_sum = contact.sum(axis=1)

    # Abstract local pose proxy: use first 2 PCA-like dimensions from body feature deltas.
    feat = m[:, 7:151].copy()
    feat = feat - feat.mean(axis=0, keepdims=True)
    try:
        u, s, vh = np.linalg.svd(feat, full_matrices=False)
        local2 = u[:, :2] * s[:2]
    except Exception:
        local2 = feat[:, :2]
    local2 = local2 / (np.std(local2, axis=0, keepdims=True) + 1e-6)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 7))
    ax0 = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 1, 2)

    title = args.title or Path(args.input).stem

    def update(t):
        ax0.clear()
        ax1.clear()
        ax2.clear()

        ax0.plot(root[:,0], root[:,1], "--", linewidth=1)
        ax0.scatter(root[t,0], root[t,1], s=30)
        ax0.set_title("Global root X/Z trajectory")
        ax0.set_aspect("equal", adjustable="box")
        ax0.grid(True)

        ax1.plot(local2[:,0], local2[:,1], linewidth=1)
        ax1.scatter(local2[t,0], local2[t,1], s=30)
        ax1.set_title("Root-centered local pose proxy")
        ax1.grid(True)

        ax2.plot(lower_e, label="lower")
        ax2.plot(torso_e, label="torso")
        ax2.plot(upper_e, label="upper")
        ax2.plot(contact_sum / 4.0 * max(upper_e.max(), lower_e.max(), 1e-6), label="contact proxy")
        ax2.axvline(t, linestyle="--")
        ax2.legend(loc="upper right")
        ax2.set_title("Per-frame feature motion energy / contact")
        ax2.grid(True)

        fig.suptitle(f"{title} | frame {t+1}/{T}")

    ani = FuncAnimation(fig, update, frames=T, interval=1000/args.fps)
    ani.save(str(out), writer=FFMpegWriter(fps=args.fps, bitrate=2500))
    plt.close(fig)
    print("✅ wrote", out)

if __name__ == "__main__":
    main()
