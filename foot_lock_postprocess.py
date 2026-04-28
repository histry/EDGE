import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


FOOT_JOINTS = [7, 8, 10, 11]


def load_motion(path):
    motion = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if motion.ndim == 0 and isinstance(motion.item(), dict):
        motion = np.asarray(motion.item()["motion"], dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 151:
        raise ValueError(f"Expected [T,151] motion, got {motion.shape} from {path}")
    return motion


def parse_frames(text, seq_len):
    if not text:
        return []
    frames = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if 0.0 < value < 1.0:
            value *= seq_len - 1
        frames.append(max(0, min(seq_len - 1, int(round(value)))))
    return sorted(set(frames))


def motion_to_feet(motion, device):
    device = torch.device(device)
    root = torch.tensor(motion[:, 4:7], dtype=torch.float32, device=device).unsqueeze(0)
    q_6d = torch.tensor(motion[:, 7:], dtype=torch.float32, device=device).reshape(1, motion.shape[0], 24, 6)
    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        joints = SMPLSkeleton(device=device).forward(q_ax, root)
    return joints[0, :, FOOT_JOINTS, :].detach().cpu().numpy()


def contiguous_segments(mask, min_len):
    segments = []
    start = None
    for idx, active in enumerate(mask):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask)))
    return segments


def remove_short_contacts(mask, min_len):
    cleaned = np.zeros_like(mask, dtype=bool)
    for foot_idx in range(mask.shape[1]):
        for start, end in contiguous_segments(mask[:, foot_idx], min_len):
            cleaned[start:end, foot_idx] = True
    return cleaned


def smooth_curve(values, window):
    window = int(window)
    if window <= 1 or len(values) < 3:
        return values
    if window % 2 == 0:
        window += 1
    window = min(window, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window <= 1:
        return values

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    flat = values.reshape(values.shape[0], -1)
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(flat)
    for col in range(flat.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out.reshape(values.shape).astype(values.dtype, copy=False)


def protected_envelope(seq_len, protected_frames, radius):
    envelope = np.ones((seq_len, 1), dtype=np.float32)
    radius = max(0, int(radius))
    for frame in protected_frames:
        left = max(0, frame - radius)
        right = min(seq_len, frame + radius + 1)
        for idx in range(left, right):
            if radius == 0:
                weight = 0.0
            else:
                dist = abs(idx - frame) / float(radius)
                weight = 1.0 - 0.5 * (1.0 + np.cos(np.pi * min(1.0, dist)))
            envelope[idx, 0] = min(envelope[idx, 0], weight)
    return envelope


def build_lock_targets(
    motion,
    feet,
    fps,
    contact_threshold,
    height_threshold,
    max_lock_speed,
    min_contact_frames,
):
    seq_len = motion.shape[0]
    root_xz = motion[:, [4, 6]]
    feet_xz = feet[:, :, [0, 2]]
    local_offsets = feet_xz - root_xz[:, None, :]

    heights = feet[:, :, 1]
    floor = float(np.percentile(heights, 2))
    channel_contacts = motion[:, 0:4] > contact_threshold
    height_contacts = heights <= floor + height_threshold

    foot_speed = np.zeros((seq_len, 4), dtype=np.float32)
    if seq_len > 1:
        pair_speed = np.linalg.norm(feet_xz[1:] - feet_xz[:-1], axis=-1) * float(fps)
        foot_speed[1:] = np.maximum(foot_speed[1:], pair_speed)
        foot_speed[:-1] = np.maximum(foot_speed[:-1], pair_speed)
    speed_contacts = foot_speed <= max_lock_speed

    lock_contacts = channel_contacts & height_contacts & speed_contacts
    lock_contacts = remove_short_contacts(lock_contacts, min_contact_frames)

    lock_targets = np.zeros((seq_len, 4, 2), dtype=np.float32)
    lock_weights = np.zeros((seq_len, 4), dtype=np.float32)
    segment_count = 0

    for foot_idx in range(4):
        for start, end in contiguous_segments(lock_contacts[:, foot_idx], min_contact_frames):
            segment_positions = feet_xz[start:end, foot_idx]
            anchor = np.median(segment_positions, axis=0).astype(np.float32)
            confidence = np.clip(motion[start:end, foot_idx], 0.0, 1.0).astype(np.float32)
            # Downweight long sliding segments instead of forcing the whole body to chase them.
            segment_speed = foot_speed[start:end, foot_idx]
            speed_weight = np.clip(1.0 - segment_speed / max(max_lock_speed, 1e-6), 0.15, 1.0)
            weights = confidence * speed_weight
            lock_targets[start:end, foot_idx] = anchor[None, :] - local_offsets[start:end, foot_idx]
            lock_weights[start:end, foot_idx] = weights
            segment_count += 1

    stats = {
        "floor_y_m": floor,
        "channel_contact_ratio": float(channel_contacts.mean()),
        "height_contact_ratio": float(height_contacts.mean()),
        "lock_contact_ratio": float(lock_contacts.mean()),
        "lock_segment_count": int(segment_count),
        "lock_frame_count": int(lock_contacts.sum()),
    }
    return lock_targets, lock_weights, lock_contacts, stats


def optimize_root_xz(
    motion,
    lock_targets,
    lock_weights,
    protected_frames,
    steps,
    lr,
    lock_weight,
    trajectory_weight,
    velocity_weight,
    smooth_weight,
    protected_weight,
    max_root_shift,
    device,
):
    device = torch.device(device)
    root_orig = torch.tensor(motion[:, [4, 6]], dtype=torch.float32, device=device)
    root = torch.nn.Parameter(root_orig.clone())
    targets = torch.tensor(lock_targets, dtype=torch.float32, device=device)
    weights = torch.tensor(lock_weights, dtype=torch.float32, device=device)
    protected = torch.tensor(protected_frames, dtype=torch.long, device=device) if protected_frames else None

    optimizer = torch.optim.Adam([root], lr=float(lr))
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)

        loss = trajectory_weight * (root - root_orig).pow(2).mean()
        if weights.sum() > 1e-6:
            root_for_feet = root[:, None, :]
            lock_error = (root_for_feet - targets).pow(2).sum(dim=-1)
            loss = loss + lock_weight * (lock_error * weights).sum() / weights.sum().clamp(min=1e-6)

        if root.shape[0] > 1:
            root_delta = root[1:] - root[:-1]
            orig_delta = root_orig[1:] - root_orig[:-1]
            loss = loss + velocity_weight * (root_delta - orig_delta).pow(2).mean()

        if root.shape[0] > 2:
            root_acc = root[2:] - 2.0 * root[1:-1] + root[:-2]
            loss = loss + smooth_weight * root_acc.pow(2).mean()

        shift_norm = torch.linalg.norm(root - root_orig, dim=-1)
        loss = loss + 20.0 * torch.relu(shift_norm - max_root_shift).pow(2).mean()

        if protected is not None and protected.numel() > 0:
            loss = loss + protected_weight * (root[protected] - root_orig[protected]).pow(2).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([root], 1.0)
        optimizer.step()

        with torch.no_grad():
            if protected is not None and protected.numel() > 0:
                root[protected] = root_orig[protected]
            shift = root - root_orig
            norm = torch.linalg.norm(shift, dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(max_root_shift / norm, max=1.0)
            root.copy_(root_orig + shift * scale)

    return root.detach().cpu().numpy().astype(np.float32)


def apply_foot_lock(motion, args):
    output = motion.copy()
    seq_len = output.shape[0]
    protected_frames = parse_frames(args.protected_frames, seq_len)
    if not protected_frames and args.protect_endpoints:
        protected_frames = [0, seq_len - 1]

    feet = motion_to_feet(output, args.device)
    lock_targets, lock_weights, lock_contacts, stats = build_lock_targets(
        output,
        feet,
        fps=args.fps,
        contact_threshold=args.contact_threshold,
        height_threshold=args.height_threshold,
        max_lock_speed=args.max_lock_speed,
        min_contact_frames=args.min_contact_frames,
    )

    if lock_weights.sum() <= 1e-6:
        stats["applied"] = False
        return output, stats

    root_locked = optimize_root_xz(
        output,
        lock_targets,
        lock_weights,
        protected_frames=protected_frames,
        steps=args.steps,
        lr=args.lr,
        lock_weight=args.lock_weight,
        trajectory_weight=args.trajectory_weight,
        velocity_weight=args.velocity_weight,
        smooth_weight=args.smooth_weight,
        protected_weight=args.protected_weight,
        max_root_shift=args.max_root_shift,
        device=args.device,
    )

    correction = root_locked - output[:, [4, 6]]
    correction = smooth_curve(correction, args.correction_smooth_window)
    correction *= protected_envelope(seq_len, protected_frames, args.protected_radius)

    shift_norm = np.linalg.norm(correction, axis=-1, keepdims=True)
    scale = np.minimum(1.0, args.max_root_shift / np.maximum(shift_norm, 1e-6))
    correction = correction * scale * float(args.strength)

    output[:, [4, 6]] += correction
    if args.update_contact_channels:
        output[:, 0:4] = lock_contacts.astype(np.float32)

    stats.update(
        {
            "applied": True,
            "protected_frames": protected_frames,
            "strength": float(args.strength),
            "max_root_shift_m": float(np.linalg.norm(correction, axis=-1).max()),
            "mean_root_shift_m": float(np.linalg.norm(correction, axis=-1).mean()),
            "updated_contact_channels": bool(args.update_contact_channels),
        }
    )
    return output.astype(np.float32), stats


def main():
    parser = argparse.ArgumentParser(description="Conservative contact-aware foot lock for EDGE 151-D motions.")
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats_json", default="")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--protected_frames", default="", help="Comma separated frames to keep unchanged, e.g. 0,180,360,599")
    parser.add_argument("--protect_endpoints", action="store_true", default=True)
    parser.add_argument("--protected_radius", type=int, default=8)
    parser.add_argument("--contact_threshold", type=float, default=0.8)
    parser.add_argument("--height_threshold", type=float, default=0.06)
    parser.add_argument("--max_lock_speed", type=float, default=0.8)
    parser.add_argument("--min_contact_frames", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--lock_weight", type=float, default=6.0)
    parser.add_argument("--trajectory_weight", type=float, default=18.0)
    parser.add_argument("--velocity_weight", type=float, default=2.0)
    parser.add_argument("--smooth_weight", type=float, default=0.25)
    parser.add_argument("--protected_weight", type=float, default=200.0)
    parser.add_argument("--max_root_shift", type=float, default=0.12)
    parser.add_argument("--correction_smooth_window", type=int, default=9)
    parser.add_argument("--update_contact_channels", action="store_true", default=True)
    args = parser.parse_args()

    motion = load_motion(args.motion)
    output, stats = apply_foot_lock(motion, args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, output)

    if args.stats_json:
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved foot-locked motion: {out_path}")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")


if __name__ == "__main__":
    main()
