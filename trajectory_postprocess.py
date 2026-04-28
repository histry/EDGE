import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


TRAJECTORY_POST_MODES = ("none", "soft", "hard", "optimize")


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
    return target_traj[:length, :2]


def remove_initial_root_offset(output_np):
    output = np.asarray(output_np, dtype=np.float32).copy()
    output[:, 4] -= output[0, 4]
    output[:, 6] -= output[0, 6]
    return output


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

    return output


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
        loss = (
            traj_weight * traj_loss
            + velocity_weight * velocity_loss
            + smooth_weight * smooth_loss
            + foot_weight * foot_loss
            + 20.0 * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            root[0].copy_(target[0])
            root[-1].copy_(target[-1])

    optimized_root = root.detach().cpu().numpy()
    output[:, 4] = optimized_root[:, 0]
    output[:, 6] = optimized_root[:, 1]
    return output


def apply_trajectory_postprocess(
    output_np,
    target_traj=None,
    mode="optimize",
    device=None,
    contact_threshold=0.8,
    soft_strength=0.65,
):
    output = np.asarray(output_np, dtype=np.float32).copy()
    mode = (mode or "optimize").lower()
    if mode not in TRAJECTORY_POST_MODES:
        raise ValueError(f"Unknown trajectory post mode: {mode}")

    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return contact_aware_foot_lock(
            remove_initial_root_offset(output),
            device=device,
            contact_threshold=contact_threshold,
        )

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    target_np = target_np[:seq_len]

    if mode == "none":
        return output
    if mode == "hard":
        output[:, 4] = target_np[:, 0]
        output[:, 6] = target_np[:, 1]
        return output
    if mode == "soft":
        output[:, 4] = output[:, 4] * (1.0 - soft_strength) + target_np[:, 0] * soft_strength
        output[:, 6] = output[:, 6] * (1.0 - soft_strength) + target_np[:, 1] * soft_strength
        return contact_aware_foot_lock(output, device=device, contact_threshold=contact_threshold)

    return optimize_root_for_trajectory(
        output,
        target_np,
        device=device,
        contact_threshold=contact_threshold,
    )
