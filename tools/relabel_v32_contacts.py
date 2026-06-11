#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct missing EDGE [0:4] foot-contact channels from FK kinematics.

The generated labels are kinematic pseudo-labels, not human ground truth.
Only contact channels are changed; root and rotation channels are preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import os
import tempfile
import numpy as np
import torch

from tools.v29_motion_geometry import CONTACT, MOTION_DIM, motion_to_joint_positions_torch

FOOT_JOINTS = (7, 8, 10, 11)  # left ankle, right ankle, left toe, right toe


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def clean_runs(binary: np.ndarray, min_run: int = 2, max_gap: int = 2) -> np.ndarray:
    x = np.asarray(binary, dtype=bool).copy()
    if len(x) < 2:
        return x

    def runs(a):
        start = 0
        value = bool(a[0])
        for i in range(1, len(a) + 1):
            if i == len(a) or bool(a[i]) != value:
                yield start, i, value
                if i < len(a):
                    start, value = i, bool(a[i])

    for start, end, value in list(runs(x)):
        if (not value and end - start <= max_gap and start > 0 and end < len(x)
                and x[start - 1] and x[end]):
            x[start:end] = True
    for start, end, value in list(runs(x)):
        if value and end - start < min_run:
            x[start:end] = False
    return x


def infer_contacts(
    feet: np.ndarray,
    fps: float = 30.0,
    enter: float = 0.97,
    leave: float = 0.90,
    min_run: int = 3,
    max_gap: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer hard labels and soft confidence from [T,4,3] foot positions."""
    feet = np.asarray(feet, np.float32)
    t = len(feet)
    if t < 2:
        return np.zeros((t, 4), np.float32), np.zeros((t, 4), np.float32)

    # Smooth only for velocity estimation.
    padded = np.pad(feet, ((1, 1), (0, 0), (0, 0)), mode="edge")
    smooth = 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]
    velocity = np.zeros_like(smooth)
    velocity[0] = (smooth[1] - smooth[0]) * fps
    velocity[-1] = (smooth[-1] - smooth[-2]) * fps
    if t > 2:
        velocity[1:-1] = (smooth[2:] - smooth[:-2]) * (0.5 * fps)
    speed = np.linalg.norm(velocity[..., (0, 2)], axis=-1)

    height = feet[..., 1]
    ground = np.quantile(height, 0.08, axis=0)
    q35_height = np.quantile(height, 0.35, axis=0)
    height_margin = np.clip(0.55 * (q35_height - ground) + 0.018, 0.025, 0.080)
    height_threshold = ground + height_margin
    height_scale = np.maximum(0.25 * height_margin, 0.006)

    q35_speed = np.quantile(speed, 0.35, axis=0)
    speed_threshold = np.clip(1.50 * q35_speed + 0.018, 0.055, 0.240)
    speed_scale = np.maximum(0.30 * speed_threshold, 0.015)

    p_height = sigmoid((height_threshold[None] - height) / height_scale[None])
    p_speed = sigmoid((speed_threshold[None] - speed) / speed_scale[None])
    probability = (p_height ** 0.70 * p_speed ** 0.30).astype(np.float32)

    hard = np.zeros_like(probability, np.float32)
    for channel in range(4):
        state = False
        values = np.zeros((t,), dtype=bool)
        for i, p in enumerate(probability[:, channel]):
            state = bool(p >= (leave if state else enter))
            values[i] = state
        hard[:, channel] = clean_runs(
            values,
            min_run=min_run,
            max_gap=max_gap,
        ).astype(np.float32)

    # Avoid an entirely empty anatomical side in very short/root-local clips.
    for channels in ((0, 2), (1, 3)):
        if hard[:, channels].sum() == 0:
            side = probability[:, channels]
            flat = side.reshape(-1)
            count = min(len(flat), max(2, int(round(0.06 * t))))
            selected = np.argpartition(flat, -count)[-count:]
            for value in selected:
                frame = int(value // 2)
                local = int(value % 2)
                hard[frame, channels[local]] = 1.0
            for channel in channels:
                hard[:, channel] = clean_runs(
                    hard[:, channel] > 0.5,
                    min_run=min_run,
                    max_gap=max_gap,
                ).astype(np.float32)

    return hard, probability


def contact_rate(target: np.ndarray, mask: np.ndarray) -> float:
    valid = mask[..., None]
    return float((target[..., :4].clip(0, 1) * valid).sum() /
                 max(float(valid.sum()) * 4.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_npz", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--enter", type=float, default=0.97)
    parser.add_argument("--leave", type=float, default=0.90)
    parser.add_argument("--min_run", type=int, default=3)
    parser.add_argument("--max_gap", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input_npz)
    destination = Path(args.out_npz)
    z = np.load(source, allow_pickle=True)
    arrays = {name: np.asarray(z[name]) for name in z.files}
    for name in ("target", "mask", "start", "end", "length"):
        if name not in arrays:
            raise RuntimeError(f"Missing array: {name}")

    target = np.asarray(arrays["target"], np.float32).copy()
    mask = np.asarray(arrays["mask"], np.float32)
    start = np.asarray(arrays["start"], np.float32).copy()
    end = np.asarray(arrays["end"], np.float32).copy()
    length = np.asarray(arrays["length"], np.int32)
    original_rate = contact_rate(target, mask)
    if original_rate >= 0.005 and not bool(args.force):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=destination.stem + ".", suffix=".npz",
            dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary_path, **arrays)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        result = {
            "input": str(source),
            "output": str(destination),
            "num_samples": int(len(target)),
            "original_contact_rate": original_rate,
            "reconstructed_contact_rate": original_rate,
            "label_source": "preserved_existing_contact",
            "skipped_reconstruction": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.out_json:
            Path(args.out_json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return

    n, maximum, dim = target.shape
    if dim != MOTION_DIM:
        raise ValueError(f"Expected motion dim {MOTION_DIM}, got {dim}")
    confidence_mean = np.zeros((n, 4), np.float32)
    sample_rate = np.zeros((n, 4), np.float32)
    device = torch.device(args.device)

    for first in range(0, n, args.batch_size):
        last = min(n, first + args.batch_size)
        count = last - first
        context = np.zeros((count, maximum + 2, MOTION_DIM), np.float32)
        context[:, 0] = start[first:last]
        for local, sample in enumerate(range(first, last)):
            k = int(length[sample])
            context[local, 1:k + 1] = target[sample, :k]
            context[local, k + 1:] = end[sample]

        with torch.no_grad():
            tensor = torch.from_numpy(context).to(device)
            feet = motion_to_joint_positions_torch(tensor)[..., FOOT_JOINTS, :].cpu().numpy()

        for local, sample in enumerate(range(first, last)):
            k = int(length[sample])
            hard, probability = infer_contacts(
                feet[local, :k + 2],
                fps=args.fps,
                enter=args.enter,
                leave=args.leave,
                min_run=args.min_run,
                max_gap=args.max_gap,
            )
            start[sample, :4] = hard[0]
            target[sample, :k, :4] = hard[1:k + 1]
            target[sample, k:, :4] = 0.0
            end[sample, :4] = hard[k + 1]
            confidence_mean[sample] = probability[1:k + 1].mean(axis=0)
            sample_rate[sample] = hard[1:k + 1].mean(axis=0)
        print(f"[CONTACT RELABEL] {last}/{n}", flush=True)

    arrays["target"] = target
    arrays["start"] = start
    arrays["end"] = end
    arrays["contact_label_source"] = np.asarray(
        ["kinematic_pseudo_contact"] * n, dtype=object
    )
    arrays["contact_confidence_mean"] = confidence_mean
    arrays["contact_rate_per_sample"] = sample_rate

    meta = {}
    if "meta" in arrays:
        try:
            meta = json.loads(str(arrays["meta"].item()))
        except Exception:
            meta = {}
    final_rate = contact_rate(target, mask)
    meta["contact_reconstruction"] = {
        "enabled": True,
        "label_status": "kinematic_pseudo_contact_not_human_ground_truth",
        "method": "FK foot height + horizontal speed + hysteresis + temporal cleanup",
        "fps": args.fps,
        "enter_threshold": args.enter,
        "leave_threshold": args.leave,
        "min_run_frames": args.min_run,
        "max_gap_frames": args.max_gap,
        "original_contact_rate": original_rate,
        "reconstructed_contact_rate": final_rate,
    }
    arrays["meta"] = np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + ".", suffix=".npz", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)

    result = {
        "input": str(source),
        "output": str(destination),
        "num_samples": int(n),
        "original_contact_rate": original_rate,
        "reconstructed_contact_rate": final_rate,
        "mean_contact_rate_per_channel": sample_rate.mean(axis=0).astype(float).tolist(),
        "mean_confidence_per_channel": confidence_mean.mean(axis=0).astype(float).tolist(),
        "label_source": "kinematic_pseudo_contact",
        "enter_threshold": args.enter,
        "leave_threshold": args.leave,
        "min_run_frames": args.min_run,
        "max_gap_frames": args.max_gap,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
