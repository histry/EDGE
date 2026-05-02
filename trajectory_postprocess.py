import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


TRAJECTORY_POST_MODES = ("none", "soft", "hard", "optimize", "balanced")


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
    if target_traj.ndim != 2 or target_traj.shape[-1] < 2:
        raise ValueError(f"target_traj must be [T,2] or [1,T,2], got {target_traj.shape}")
    return target_traj[:length, :2].astype(np.float32)


def remove_initial_root_offset(output_np):
    output = np.asarray(output_np, dtype=np.float32).copy()
    if len(output) == 0:
        return output
    output[:, 4] -= output[0, 4]
    output[:, 6] -= output[0, 6]
    return output


def _smooth_2d(x, window=5):
    x = np.asarray(x, dtype=np.float32)
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    out = np.zeros_like(x, dtype=np.float32)
    for d in range(x.shape[1]):
        out[:, d] = np.convolve(np.pad(x[:, d], (pad, pad), mode="edge"), kernel, mode="valid")
    return out.astype(np.float32)


def contact_aware_foot_lock(
    output_np,
    device=None,
    contact_threshold=0.8,
    strength=1.0,
    max_root_delta=0.08,
    smooth_window=5,
):
    """Reduce foot sliding by root X/Z compensation during contact frames.

    This is intentionally conservative: root corrections are clipped and
    smoothed to avoid introducing new jitter.
    """
    output = np.asarray(output_np, dtype=np.float32).copy()
    seq_len = output.shape[0]
    if seq_len < 2:
        return output

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return output

    device = _as_device(device)
    contact_probs = output[:, 0:4]
    q_6d = torch.tensor(output[:, 7:], device=device, dtype=torch.float32).reshape(1, seq_len, 24, 6)
    root = torch.tensor(output[:, 4:7], device=device, dtype=torch.float32).reshape(1, seq_len, 3)

    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        feet_positions = SMPLSkeleton(device=device).forward(q_ax, root).detach().cpu().numpy()[0][:, [7, 8, 10, 11], :]

    root_delta = np.zeros((seq_len, 2), dtype=np.float32)
    count = np.zeros((seq_len, 1), dtype=np.float32)

    for frame_idx in range(1, seq_len):
        is_contact = contact_probs[frame_idx] > contact_threshold
        if not np.any(is_contact):
            continue

        displacement = feet_positions[frame_idx, is_contact] - feet_positions[frame_idx - 1, is_contact]
        # Compensate horizontal foot drift by moving root in the opposite direction.
        delta = -np.mean(displacement[:, [0, 2]], axis=0).astype(np.float32)
        root_delta[frame_idx] += delta
        count[frame_idx, 0] += 1.0

    valid = count[:, 0] > 0
    if not np.any(valid):
        return output

    root_delta[valid] = root_delta[valid] / np.maximum(count[valid], 1e-6)

    if max_root_delta is not None and float(max_root_delta) > 0:
        norm = np.linalg.norm(root_delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_root_delta) / np.maximum(norm, 1e-8))
        root_delta = root_delta * scale

    root_delta = _smooth_2d(root_delta, smooth_window)
    output[:, 4] += strength * root_delta[:, 0]
    output[:, 6] += strength * root_delta[:, 1]
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
    lr=0.03,
    contact_threshold=0.8,
    traj_weight=4.0,
    velocity_weight=1.0,
    smooth_weight=0.35,
    foot_weight=6.0,
    anchor_weight=6.0,
    endpoint_hard_anchor=False,
    max_root_delta=0.08,
    max_root_step=0.10,
    post_foot_lock_strength=0.35,
    verbose=False,
):
    """Balance trajectory following and foot stability.

    Previous defaults over-emphasized trajectory and endpoint anchor
    (traj=8, anchor=20, foot=2), which can drag the body along a path and cause
    foot sliding.  The new defaults keep endpoint/trajectory control useful but
    give contact stability comparable weight.

    Args:
        endpoint_hard_anchor: If True, exactly pins first/last root to target.
            Keep False for natural motion evaluation; enable only for strict demos.
        max_root_delta: Per-frame optimization step clamp in root X/Z space.
        max_root_step: Clamp optimized frame-to-frame root step.
    """
    output = np.asarray(output_np, dtype=np.float32).copy()
    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return contact_aware_foot_lock(
            remove_initial_root_offset(output),
            device=device,
            contact_threshold=contact_threshold,
            strength=post_foot_lock_strength,
            max_root_delta=max_root_delta,
        )

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    device = _as_device(device)

    target = torch.tensor(target_np[:seq_len], device=device, dtype=torch.float32)
    initial_root = torch.tensor(output[:, [4, 6]], device=device, dtype=torch.float32)
    root = torch.nn.Parameter(initial_root.clone())

    foot_offsets_xz = _precompute_foot_offsets_xz(output, device)
    contacts = torch.tensor(output[:, 0:4] > contact_threshold, device=device)
    contact_pairs = contacts[1:] & contacts[:-1]

    optimizer = torch.optim.Adam([root], lr=float(lr))
    max_root_delta = float(max_root_delta) if max_root_delta is not None else 0.0
    max_root_step = float(max_root_step) if max_root_step is not None else 0.0

    last_loss = None
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
            float(traj_weight) * traj_loss
            + float(velocity_weight) * velocity_loss
            + float(smooth_weight) * smooth_loss
            + float(foot_weight) * foot_loss
            + float(anchor_weight) * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([root], max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            if max_root_delta > 0:
                delta_from_initial = root - initial_root
                norm = delta_from_initial.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                factor = (max_root_delta / norm).clamp(max=1.0)
                root.copy_(initial_root + delta_from_initial * factor)

            if max_root_step > 0 and seq_len > 1:
                # Keep motion plausible by limiting extreme frame-to-frame root jumps.
                for i in range(1, seq_len):
                    step = root[i] - root[i - 1]
                    norm = step.norm().clamp_min(1e-8)
                    if float(norm.item()) > max_root_step:
                        root[i].copy_(root[i - 1] + step * (max_root_step / norm))

            if endpoint_hard_anchor:
                root[0].copy_(target[0])
                root[-1].copy_(target[-1])

        last_loss = float(loss.detach().cpu())

    optimized_root = root.detach().cpu().numpy()
    output[:, 4] = optimized_root[:, 0]
    output[:, 6] = optimized_root[:, 1]

    if verbose:
        print(
            "trajectory optimize balanced: "
            f"last_loss={last_loss}, traj_weight={traj_weight}, foot_weight={foot_weight}, "
            f"anchor_weight={anchor_weight}, endpoint_hard_anchor={endpoint_hard_anchor}"
        )

    if float(post_foot_lock_strength) > 0:
        output = contact_aware_foot_lock(
            output,
            device=device,
            contact_threshold=contact_threshold,
            strength=post_foot_lock_strength,
            max_root_delta=max_root_delta,
        )

    return output.astype(np.float32)


def apply_trajectory_postprocess(
    output_np,
    target_traj=None,
    mode="balanced",
    device=None,
    contact_threshold=0.8,
    soft_strength=0.45,
    traj_weight=4.0,
    velocity_weight=1.0,
    smooth_weight=0.35,
    foot_weight=6.0,
    anchor_weight=6.0,
    endpoint_hard_anchor=False,
    max_root_delta=0.08,
    max_root_step=0.10,
    post_foot_lock_strength=0.35,
):
    """Apply trajectory post-processing.

    Recommended:
        mode="none" for raw model evaluation.
        mode="soft" for gentle system demos.
        mode="balanced" for trajectory demos that still protect foot contacts.
        mode="hard" only for debugging target trajectory serialization.
    """
    output = np.asarray(output_np, dtype=np.float32).copy()
    mode = (mode or "balanced").lower()
    if mode not in TRAJECTORY_POST_MODES:
        raise ValueError(f"Unknown trajectory post mode: {mode}")

    target_np = _target_to_numpy(target_traj, output.shape[0])
    if target_np is None:
        return contact_aware_foot_lock(
            remove_initial_root_offset(output),
            device=device,
            contact_threshold=contact_threshold,
            strength=post_foot_lock_strength,
            max_root_delta=max_root_delta,
        )

    seq_len = min(output.shape[0], target_np.shape[0])
    output = output[:seq_len].copy()
    target_np = target_np[:seq_len]

    if mode == "none":
        return output.astype(np.float32)

    if mode == "hard":
        # Hard anchoring destroys foot contact consistency. Use only for
        # debugging or strict path visualization, not for naturalness metrics.
        output[:, 4] = target_np[:, 0]
        output[:, 6] = target_np[:, 1]
        return output.astype(np.float32)

    if mode == "soft":
        strength = float(np.clip(soft_strength, 0.0, 1.0))
        output[:, 4] = output[:, 4] * (1.0 - strength) + target_np[:, 0] * strength
        output[:, 6] = output[:, 6] * (1.0 - strength) + target_np[:, 1] * strength
        return contact_aware_foot_lock(
            output,
            device=device,
            contact_threshold=contact_threshold,
            strength=post_foot_lock_strength,
            max_root_delta=max_root_delta,
        )

    # "optimize" is kept for backward compatibility, but now uses the balanced
    # optimizer unless callers explicitly pass old-style weights.
    return optimize_root_for_trajectory(
        output,
        target_np,
        device=device,
        contact_threshold=contact_threshold,
        traj_weight=traj_weight,
        velocity_weight=velocity_weight,
        smooth_weight=smooth_weight,
        foot_weight=foot_weight,
        anchor_weight=anchor_weight,
        endpoint_hard_anchor=endpoint_hard_anchor,
        max_root_delta=max_root_delta,
        max_root_step=max_root_step,
        post_foot_lock_strength=post_foot_lock_strength,
    )
