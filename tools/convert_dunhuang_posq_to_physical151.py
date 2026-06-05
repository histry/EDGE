#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton


def convert_file(src: Path, dst: Path, foot_vel_threshold: float) -> None:
    with src.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict) or "pos" not in data or "q" not in data:
        raise ValueError(f"{src}: expected dict with pos/q")

    pos = np.asarray(data["pos"], dtype=np.float32)
    q = np.asarray(data["q"], dtype=np.float32)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"{src}: invalid pos shape {pos.shape}")
    if q.ndim != 2 or q.shape[1] != 72:
        raise ValueError(f"{src}: invalid q shape {q.shape}")
    if len(pos) != len(q):
        raise ValueError(f"{src}: pos/q length mismatch")

    root_pos = torch.from_numpy(pos).float().unsqueeze(0)
    local_q = torch.from_numpy(q.reshape(len(q), 24, 3)).float().unsqueeze(0)

    smpl = SMPLSkeleton()

    with torch.no_grad():
        positions = smpl.forward(local_q, root_pos)

        # EDGE 原始脚部关节索引
        feet = positions[:, :, (7, 8, 10, 11)]

        feet_velocity = torch.zeros(
            feet.shape[:3],
            dtype=feet.dtype,
            device=feet.device,
        )

        if feet.shape[1] > 1:
            feet_velocity[:, :-1] = (
                feet[:, 1:] - feet[:, :-1]
            ).norm(dim=-1)
            feet_velocity[:, -1] = feet_velocity[:, -2]

        contacts = (feet_velocity < foot_vel_threshold).to(local_q.dtype)
        rot6d = ax_to_6v(local_q)

        motion = torch.cat(
            [
                contacts,
                root_pos,
                rot6d.reshape(1, len(q), 144),
            ],
            dim=-1,
        )[0]

    motion_np = motion.cpu().numpy().astype(np.float32)

    if motion_np.shape != (len(q), 151):
        raise RuntimeError(f"{src}: converted shape {motion_np.shape}")

    out = dict(data)
    out["motion"] = motion_np
    out["motion_151"] = motion_np
    out["source_file"] = str(src)
    out["contact_method"] = "smpl_fk_foot_velocity"
    out["foot_vel_threshold"] = float(foot_vel_threshold)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--foot_vel_threshold", type=float, default=0.01)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(input_dir.rglob("*.pkl"))
    if not files:
        raise RuntimeError(f"No pkl files in {input_dir}")

    ok = 0
    failed = 0

    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel

        try:
            convert_file(src, dst, args.foot_vel_threshold)
            ok += 1
            print(f"[OK] {src} -> {dst}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {src}: {exc}", flush=True)

    print("=" * 60)
    print("converted:", ok)
    print("failed:", failed)
    print("output:", output_dir)


if __name__ == "__main__":
    main()
