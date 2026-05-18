#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LEFT_FOOT_JOINT = 10
RIGHT_FOOT_JOINT = 11


def _segments(mask: np.ndarray, min_len: int):
    segs = []
    start = None
    for i, v in enumerate(mask.astype(bool)):
        if v and start is None:
            start = i
        if (not v or i == len(mask) - 1) and start is not None:
            end = i if not v else i + 1
            if end - start >= min_len:
                segs.append((start, end))
            start = None
    return segs


def _fk_positions(motion_151: np.ndarray, device: str) -> np.ndarray:
    from dataset.quaternion import ax_from_6v
    from vis import SMPLSkeleton

    x = torch.tensor(motion_151[None], dtype=torch.float32, device=device)
    pos = x[:, :, 4:7]
    q6 = x[:, :, 7:].reshape(1, x.shape[1], 24, 6)
    qax = ax_from_6v(q6)
    smpl = SMPLSkeleton(device=torch.device(device))
    with torch.no_grad():
        joints = smpl.forward(qax, pos)
    return joints.detach().cpu().numpy()[0]


def _contact_masks(motion: np.ndarray, threshold: float):
    contacts = motion[:, :4]
    # Conservative mapping:
    #   channels 0/2 -> left side
    #   channels 1/3 -> right side
    # This works for common toe/heel contact layouts.
    left = (contacts[:, 0] > threshold) | (contacts[:, 2] > threshold)
    right = (contacts[:, 1] > threshold) | (contacts[:, 3] > threshold)
    return left, right


def footlock_one(motion: np.ndarray, threshold: float, min_segment: int, strength: float, device: str):
    if motion.ndim != 2 or motion.shape[-1] != 151:
        raise ValueError(f"expected [T,151], got {motion.shape}")

    out = motion.copy()
    joints = _fk_positions(out, device=device)

    left_contact, right_contact = _contact_masks(out, threshold=threshold)
    correction = np.zeros((out.shape[0], 2), dtype=np.float32)
    counts = np.zeros((out.shape[0], 1), dtype=np.float32)

    for side_name, mask, joint_id in [
        ("left", left_contact, LEFT_FOOT_JOINT),
        ("right", right_contact, RIGHT_FOOT_JOINT),
    ]:
        for s, e in _segments(mask, min_segment):
            foot_xz = joints[s:e, joint_id, :][:, [0, 2]]
            # Use median as a robust planted target.
            target = np.median(foot_xz, axis=0)
            delta = target[None, :] - foot_xz
            correction[s:e] += delta.astype(np.float32)
            counts[s:e] += 1.0
            print(f"lock {side_name} segment [{s},{e}) target_xz={target}")

    valid = counts[:, 0] > 0
    if valid.any():
        correction[valid] = correction[valid] / counts[valid]
        # root x/z indices are 4 and 6.
        out[valid, 4] += strength * correction[valid, 0]
        out[valid, 6] += strength * correction[valid, 1]

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--contact_threshold", type=float, default=0.5)
    ap.add_argument("--min_segment", type=int, default=3)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    arr = np.load(args.motion, allow_pickle=True).astype(np.float32)
    original_shape = arr.shape

    if arr.ndim == 2:
        arr_b = arr[None]
    elif arr.ndim == 3:
        arr_b = arr
    else:
        raise ValueError(f"motion must be [T,151] or [B,T,151], got {arr.shape}")

    outs = []
    for b in range(arr_b.shape[0]):
        print(f"processing batch {b}")
        outs.append(
            footlock_one(
                arr_b[b],
                threshold=args.contact_threshold,
                min_segment=args.min_segment,
                strength=args.strength,
                device=args.device,
            )
        )

    out = np.stack(outs, axis=0).astype(np.float32)
    if len(original_shape) == 2:
        out = out[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, out)
    print("saved:", out_path, out.shape)


if __name__ == "__main__":
    main()
