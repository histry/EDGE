"""Trajectory postprocess utilities for EDGE 151-D motion.

Drop-in replacement for trajectory_postprocess.py.

New in this version:
- mode="contact_soft": contact-aware soft root anchoring.
  During foot contact, root X/Z is pulled toward the target trajectory with a
  lower strength, so the trajectory is followed without dragging planted feet.
- optional CLI for standalone postprocessing of existing .npy motions.

The old modes are preserved: none, soft, hard, optimize.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


TRAJECTORY_POST_MODES = ("none", "soft", "hard", "optimize", "contact_soft")


def _as_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if not isinstance(device, torch.device) else device


def _target_to_numpy(target_traj, length):
    if target_traj is None:
        return None
    if torch.is_tensor(target_traj):
        target_traj = target_traj.detach().cpu().float().numpy()
    target_traj = np.asarray(target_traj, dtype=np.float32)
    if target_traj.ndim == 3:
        target_traj = target_traj[0]
    if target_traj.ndim != 2 or target_traj.shape[1] < 2:
        raise ValueError(f"target_traj must have shape [T,2] or [1,T,2], got {target_traj.shape}")
    target_traj = target_traj[:, :2]
    if len(target_traj) != int(length):
        old_x = np.linspace(0.0, 1.0, len(target_traj), dtype=np.float32)
        new_x = np.linspace(0.0, 1.0, int(length), dtype=np.float32)
        target_traj = np.stack([
            np.interp(new_x, old_x, target_traj[:, 0]),
            np.interp(new_x, old_x, target_traj[:, 1]),
        ], axis=-1).astype(np.float32)
    return target_traj[:length, :2].astype(np.float32)


def _moving_average_2d(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(np.pad(x[:, c], (pad, pad), mode="edge"), kernel, mode="valid")
    return out.astype(np.float32)


def remove_initial_root_offset(output_np):
    output = np.asarray(output_np, dtype=np.float32).copy()
    output[:, 4] -= output[0, 4]
    output[:, 6] -= output[0, 6]
    return output


def _motion_to_joints(output_np, device=None):
    device = _as_device(device)
    output_np = np.asarray(output_np, dtype=np.float32)
    seq_len = output_np.shape[0]
    q_6d = torch.tensor(output_np[:, 7:], device=device, dtype=torch.float32).reshape(1, seq_len, 24, 6)
    root = torch.tensor(output_np[:, 4:7], device=device, dtype=torch.float32).reshape(1, seq_len, 3)
    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        joints = SMPLSkeleton(device=device).forward(q_ax, root)
    return joints.detach().cpu().numpy()[0]


def _height_contact_mask(output_np, device=None, height_threshold=0.035):
    joints = _motion_to_joints(output_np, device=device)
    feet = joints[:, [7, 8, 10, 11], :]
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    return (heights <= floor + float(height_threshold)).astype(bool)


def infer_contact_activity(
    output_np,
    device=None,
    contact_threshold=0.8,
    height_threshold=0.035,
    source="auto",
):
    """Return per-frame bool: whether any foot is likely planted.

    source="auto" uses contact channels when they look valid, and falls back to
    height contacts otherwise. This matches the spirit of eval_quantitative.py.
    """
    output_np = np.asarray(output_np, dtype=np.float32)
    channel_contacts = output_np[:, 0:4] > float(contact_threshold)
    ratio = float(channel_contacts.mean())

    if source == "channels":
        contacts = channel_contacts
        actual = "channels"
    elif source == "height":
        contacts = _height_contact_mask(output_np, device=device, height_threshold=height_threshold)
        actual = "height"
    else:
        if 0.01 <= ratio <= 0.95:
            contacts = channel_contacts
            actual = "channels"
        else:
            contacts = _height_contact_mask(output_np, device=device, height_threshold=height_threshold)
            actual = "height"

    return contacts.any(axis=1).astype(bool), contacts.astype(bool), actual


def contact_aware_foot_lock(output_np, device=None, contact_threshold=0.8):
    output = np.asarray(output_np, dtype=np.float32).copy()
    seq_len = output.shape[0]
    if seq_len < 2:
        return output

    device = _as_device(device)
    contact_probs = output[:, 0:4]
    q_6d = torch.tensor(output[:, 7:], device=device, dtype=torch.float32).reshape(1, seq_len, 24, 6)
    root = torch.tensor(output[:, 4:7], device=device, dtype=torch.float32).reshape(1, seq_len, 3)

    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        feet_positions = SMPLSkeleton(device=device).forward(q_ax, root).detach().cpu().numpy()[0][:, [7, 8, 10, 11], :]

    for frame_idx in range(1, seq_len):
        is_contact = contact_probs[frame_idx] > contact_threshold
        if not np.any(is_contact):
            continue
        displacement = feet_positions[frame_idx, is_contact] - feet_positions[frame_idx - 1, is_contact]
        mean_displacement_x = float(np.mean(displacement[:, 0]))
        mean_displacement_z = float(np.mean(displacement[:, 2]))
        output[frame_idx:, 4] -= mean_displacement_x
        output[frame_idx:, 6] -= mean_displacement_z
        feet_positions[frame_idx:, :, 0] -= mean_displacement_x
        feet_positions[frame_idx:, :, 2] -= mean_displacement_z

    return output.astype(np.float32)


def _precompute_foot_offsets_xz(output_np, device):
    seq_len = output_np.shape[0]
    q_6d = torch.tensor(output_np[:, 7:], device=device, dtype=torch.float32).reshape(1, seq_len, 24, 6)
    root_zero = torch.zeros((1, seq_len, 3), device=device, dtype=torch.float32)
    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        joints = SMPLSkeleton(device=device).forward(q_ax, root_zero)[0, :, [7, 8, 10, 11], :]
    return joints[..., [0, 2]].detach()


def optimize_root_for_trajectory(
    output_np,
    target_traj,
    device=None,
    steps=80,
    lr=0.05,
    contact_threshold=0.8,
    traj_weight=8.0,
    velocity_weight=1.0,
    smooth_weight=0.15,
    foot_weight=2.0,
):
    output = np.asarray(output_np, dtype=np.float32).copy()
    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return contact_aware_foot_lock(remove_initial_root_offset(output), device=device, contact_threshold=contact_threshold)

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    device = _as_device(device)

    target = torch.tensor(target_np[:seq_len], device=device, dtype=torch.float32)
    root = torch.tensor(output[:, [4, 6]], device=device, dtype=torch.float32)
    root = torch.nn.Parameter(root)

    foot_offsets_xz = _precompute_foot_offsets_xz(output, device)
    contacts = torch.tensor(output[:, 0:4] > contact_threshold, device=device)
    contact_pairs = contacts[1:] & contacts[:-1]

    optimizer = torch.optim.Adam([root], lr=lr)
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        traj_loss = (root - target).pow(2).mean()
        root_delta = root[1:] - root[:-1]
        target_delta = target[1:] - target[:-1]
        velocity_loss = (root_delta - target_delta).pow(2).mean()
        if seq_len > 2:
            root_acc = root[2:] - 2.0 * root[1:-1] + root[:-2]
            smooth_loss = root_acc.pow(2).mean()
        else:
            smooth_loss = root_delta.pow(2).mean()

        foot_loss = torch.tensor(0.0, device=device)
        if bool(contact_pairs.any().item()):
            foot_delta = foot_offsets_xz[1:] - foot_offsets_xz[:-1]
            foot_world_delta = foot_delta + root_delta[:, None, :]
            foot_error = foot_world_delta.pow(2).sum(dim=-1)
            foot_loss = foot_error[contact_pairs].mean()

        anchor_loss = (root[0] - target[0]).pow(2).mean() + (root[-1] - target[-1]).pow(2).mean()
        loss = traj_weight * traj_loss + velocity_weight * velocity_loss + smooth_weight * smooth_loss + foot_weight * foot_loss + 20.0 * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            root[0].copy_(target[0])
            root[-1].copy_(target[-1])

    optimized_root = root.detach().cpu().numpy()
    output[:, 4] = optimized_root[:, 0]
    output[:, 6] = optimized_root[:, 1]
    return output.astype(np.float32)


def contact_aware_soft_anchor(
    output_np,
    target_traj,
    device=None,
    base_strength=0.80,
    contact_strength_scale=0.30,
    contact_threshold=0.8,
    height_threshold=0.035,
    smooth_window=11,
    endpoint_strength=1.0,
    contact_source="auto",
    foot_lock_after=False,
):
    """Softly move root X/Z toward target while respecting foot contacts.

    The original hard/soft anchor treats all frames equally. That gives low ADE
    but drags planted feet. This function reduces anchor strength on frames where
    a foot is likely in contact. It preserves local root-foot rhythm while still
    following the global S trajectory.
    """
    output = np.asarray(output_np, dtype=np.float32).copy()
    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return output.astype(np.float32)

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    target_np = target_np[:seq_len]

    contact_any, contacts, actual_source = infer_contact_activity(
        output,
        device=device,
        contact_threshold=contact_threshold,
        height_threshold=height_threshold,
        source=contact_source,
    )

    root = output[:, [4, 6]].copy()
    delta = target_np - root
    delta = _moving_average_2d(delta, smooth_window)

    base_strength = float(np.clip(base_strength, 0.0, 1.0))
    contact_strength_scale = float(np.clip(contact_strength_scale, 0.0, 1.0))
    strength = np.full((seq_len,), base_strength, dtype=np.float32)
    strength[contact_any] *= contact_strength_scale

    # Keep endpoints attached, but ramp instead of creating one-frame snaps.
    endpoint_strength = float(np.clip(endpoint_strength, 0.0, 1.0))
    ramp_len = min(8, max(1, seq_len // 10))
    if endpoint_strength > 0:
        ramp = np.linspace(endpoint_strength, base_strength, ramp_len, dtype=np.float32)
        strength[:ramp_len] = np.maximum(strength[:ramp_len], ramp)
        strength[-ramp_len:] = np.maximum(strength[-ramp_len:], ramp[::-1])

    new_root = root + strength[:, None] * delta
    output[:, 4] = new_root[:, 0]
    output[:, 6] = new_root[:, 1]
    output = output.astype(np.float32)

    if foot_lock_after:
        output = contact_aware_foot_lock(output, device=device, contact_threshold=contact_threshold)

    # Attach lightweight diagnostics for callers that inspect the function.
    contact_aware_soft_anchor.last_debug = {
        "mode": "contact_soft",
        "contact_source_actual": actual_source,
        "contact_ratio_any": float(contact_any.mean()),
        "base_strength": float(base_strength),
        "contact_strength_scale": float(contact_strength_scale),
        "smooth_window": int(smooth_window),
        "foot_lock_after": bool(foot_lock_after),
    }
    return output


def apply_trajectory_postprocess(
    output_np,
    target_traj=None,
    mode="optimize",
    device=None,
    contact_threshold=0.8,
    soft_strength=0.65,
    contact_strength_scale=0.30,
    height_threshold=0.035,
    smooth_window=11,
    foot_lock_after=False,
):
    output = np.asarray(output_np, dtype=np.float32).copy()
    mode = (mode or "optimize").lower()
    if mode not in TRAJECTORY_POST_MODES:
        raise ValueError(f"Unknown trajectory post mode: {mode}; expected one of {TRAJECTORY_POST_MODES}")

    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return contact_aware_foot_lock(remove_initial_root_offset(output), device=device, contact_threshold=contact_threshold)

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    target_np = target_np[:seq_len]

    if mode == "none":
        return output.astype(np.float32)
    if mode == "hard":
        output[:, 4] = target_np[:, 0]
        output[:, 6] = target_np[:, 1]
        return output.astype(np.float32)
    if mode == "soft":
        soft_strength = float(np.clip(soft_strength, 0.0, 1.0))
        output[:, 4] = output[:, 4] * (1.0 - soft_strength) + target_np[:, 0] * soft_strength
        output[:, 6] = output[:, 6] * (1.0 - soft_strength) + target_np[:, 1] * soft_strength
        return contact_aware_foot_lock(output, device=device, contact_threshold=contact_threshold)
    if mode == "contact_soft":
        return contact_aware_soft_anchor(
            output,
            target_np,
            device=device,
            base_strength=soft_strength,
            contact_strength_scale=contact_strength_scale,
            contact_threshold=contact_threshold,
            height_threshold=height_threshold,
            smooth_window=smooth_window,
            foot_lock_after=foot_lock_after,
        )
    return optimize_root_for_trajectory(output, target_np, device=device, contact_threshold=contact_threshold)


def _load_motion(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if "motion" in data:
            arr = data["motion"]
        else:
            raise ValueError(f"{path} is dict npy but has no 'motion' key")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected [T,151], got {arr.shape} from {path}")
    return arr


def parse_args():
    p = argparse.ArgumentParser(description="Trajectory postprocess for EDGE 151-D motion.")
    p.add_argument("--motion", required=True)
    p.add_argument("--target_traj", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", default="contact_soft", choices=TRAJECTORY_POST_MODES)
    p.add_argument("--device", default="cpu")
    p.add_argument("--contact_threshold", type=float, default=0.8)
    p.add_argument("--height_threshold", type=float, default=0.035)
    p.add_argument("--soft_strength", type=float, default=0.80)
    p.add_argument("--contact_strength_scale", type=float, default=0.30)
    p.add_argument("--smooth_window", type=int, default=11)
    p.add_argument("--foot_lock_after", action="store_true")
    p.add_argument("--debug_json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    motion = _load_motion(args.motion)
    target = np.load(args.target_traj, allow_pickle=True).astype(np.float32)
    out = apply_trajectory_postprocess(
        motion,
        target,
        mode=args.mode,
        device=args.device,
        contact_threshold=args.contact_threshold,
        soft_strength=args.soft_strength,
        contact_strength_scale=args.contact_strength_scale,
        height_threshold=args.height_threshold,
        smooth_window=args.smooth_window,
        foot_lock_after=args.foot_lock_after,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out.astype(np.float32))
    print(f"saved: {args.out}")

    debug = getattr(contact_aware_soft_anchor, "last_debug", {"mode": args.mode})
    if args.debug_json:
        Path(args.debug_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.debug_json, "w", encoding="utf-8") as f:
            json.dump(debug, f, ensure_ascii=False, indent=2)
        print(f"debug: {args.debug_json}")


if __name__ == "__main__":
    main()
