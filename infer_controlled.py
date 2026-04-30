import argparse
import json
import os
from pathlib import Path

import librosa
import numpy as np
import scipy.interpolate as spi
import torch
import torch.nn.functional as F

from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.jukebox_features import extract as juke_extract
from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract
from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton, skeleton_render


ROOT_X_IDX = 4
ROOT_Z_IDX = 6


def parse_control_points(text):
    """
    输入格式:
        "0,0;1,2;-1,4;0,5"

    返回:
        np.ndarray [N, 2]
    """
    points = []
    if not text:
        return None

    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, z_text = item.split(",")
        points.append([float(x_text.strip()), float(z_text.strip())])

    if len(points) == 0:
        return None
    return np.asarray(points, dtype=np.float32)


def build_trajectory_from_points(points, seq_len):
    """
    根据控制点生成逐帧 X/Z 轨迹。
    这里用三次样条；控制点不足时自动降阶。
    """
    if points is None:
        return None

    if len(points) == 1:
        return np.repeat(points[:1], seq_len, axis=0).astype(np.float32)

    pts = points.T
    k_val = min(3, len(points) - 1)
    tck, _ = spi.splprep(pts, s=0, k=k_val)
    u_new = np.linspace(0.0, 1.0, seq_len)
    x_new, z_new = spi.splev(u_new, tck)

    traj = np.stack([x_new, z_new], axis=1).astype(np.float32)
    traj = traj - traj[0:1]
    return traj


def load_target_trajectory(args, seq_len):
    """
    支持两种输入:
    1. --trajectory "0,0;1,2;-1,4"
    2. --target_traj path.npy, shape [T,2] 或 [1,T,2]
    """
    if args.target_traj:
        traj = np.asarray(np.load(args.target_traj), dtype=np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        traj = traj[:, :2]
        if len(traj) != seq_len:
            old_x = np.linspace(0.0, 1.0, len(traj))
            new_x = np.linspace(0.0, 1.0, seq_len)
            traj = np.stack(
                [
                    np.interp(new_x, old_x, traj[:, 0]),
                    np.interp(new_x, old_x, traj[:, 1]),
                ],
                axis=1,
            ).astype(np.float32)
        if not args.keep_trajectory_absolute:
            traj = traj - traj[0:1]
        return traj

    points = parse_control_points(args.trajectory)
    return build_trajectory_from_points(points, seq_len)


def normalize_trajectory(traj_physical, normalizer):
    """
    轨迹条件必须和 motion 里的 root X/Z 处在同一归一化空间。
    你的 DunhuangDataset 也是用 normalizer.mean[4/6], std[4/6] 做这个操作。
    """
    traj = traj_physical.copy().astype(np.float32)

    if normalizer is None or not hasattr(normalizer, "mean"):
        return traj

    mean_x = float(normalizer.mean[ROOT_X_IDX])
    mean_z = float(normalizer.mean[ROOT_Z_IDX])
    std_x = float(normalizer.std[ROOT_X_IDX])
    std_z = float(normalizer.std[ROOT_Z_IDX])

    traj[:, 0] = (traj[:, 0] - mean_x) / (std_x + 1e-8)
    traj[:, 1] = (traj[:, 1] - mean_z) / (std_z + 1e-8)
    return traj.astype(np.float32)


def load_keyframe(path, normalizer, keyframe_space):
    """
    返回 normalized 151D pose，因为 diffusion constraint value 应该和 x_start 同空间。

    keyframe_space:
    - normalized: 文件已经是 normalized 151D
    - physical: 文件是物理空间 151D，需要用 checkpoint normalizer 归一化
    """
    pose = np.asarray(np.load(path), dtype=np.float32).reshape(-1)

    if pose.shape[0] != 151:
        raise ValueError(f"Expected 151-D keyframe from {path}, got shape {pose.shape}")

    if keyframe_space == "normalized":
        return pose.astype(np.float32)

    if keyframe_space == "physical":
        if normalizer is None:
            raise ValueError("--keyframe_space physical requires checkpoint normalizer")
        pose_t = torch.from_numpy(pose).float().view(1, 1, 151)
        pose_norm = normalizer.normalize(pose_t).view(151).cpu().numpy()
        return pose_norm.astype(np.float32)

    raise ValueError(f"Unknown keyframe_space: {keyframe_space}")

def keyframe_window(frame, seq_len, width):
    """
    Return frame indices around a keyframe.
    width=1 means only the exact frame.
    width=3 means frame-1, frame, frame+1 when possible.
    """
    width = max(1, int(width))
    half = width // 2

    if frame <= 0:
        start = 0
        end = min(seq_len, width)
    elif frame >= seq_len - 1:
        start = max(0, seq_len - width)
        end = seq_len
    else:
        start = max(0, frame - half)
        end = min(seq_len, start + width)
        start = max(0, end - width)

    return range(start, end)

def build_keyframe_constraint(args, seq_len, normalizer, traj_norm=None):
    """
    构造 diffusion inpainting 约束:
        mask:  [1, T, 1]
        value: [1, T, 151]

    mask=1 的帧会把 value 作为已知关键帧条件。
    """
    mask = torch.zeros((1, seq_len, 1), dtype=torch.float32)
    value = torch.zeros((1, seq_len, 151), dtype=torch.float32)

    keyframes = []

    if args.start_pose:
        keyframes.append((0, args.start_pose))

    if args.mid_poses:
        mid_paths = [x.strip() for x in args.mid_poses.replace(";", ",").split(",") if x.strip()]
        if args.mid_pose_frames:
            mid_frames = [
                int(round(float(x.strip())))
                for x in args.mid_pose_frames.replace(";", ",").split(",")
                if x.strip()
            ]
        else:
            mid_frames = [
                int(round((i + 1) * (seq_len - 1) / (len(mid_paths) + 1)))
                for i in range(len(mid_paths))
            ]

        if len(mid_paths) != len(mid_frames):
            raise ValueError("Number of --mid_poses must match --mid_pose_frames")

        for frame, path in zip(mid_frames, mid_paths):
            frame = max(1, min(seq_len - 2, frame))
            keyframes.append((frame, path))

    if args.end_pose:
        keyframes.append((seq_len - 1, args.end_pose))

    if not keyframes:
        return None

    for frame, path in keyframes:
        pose = load_keyframe(path, normalizer, args.keyframe_space)

        for target_frame in keyframe_window(frame, seq_len, args.keyframe_width):
            pose_for_frame = pose.copy()

            # Avoid conflict between pose keyframes and trajectory control.
            # Root X/Z is controlled by trajectory unless explicitly preserved.
            if traj_norm is not None and not args.preserve_keyframe_root_xz:
                pose_for_frame[ROOT_X_IDX] = traj_norm[target_frame, 0]
                pose_for_frame[ROOT_Z_IDX] = traj_norm[target_frame, 1]

            value[0, target_frame] = torch.from_numpy(pose_for_frame)
            mask[0, target_frame, 0] = 1.0
    return {"mask": mask, "value": value}

def align_audio_features(raw_feat, seq_len, expected_dim):
    raw_feat = np.asarray(raw_feat, dtype=np.float32)

    if raw_feat.ndim != 2:
        raise ValueError(f"Expected audio feature shape [T,C], got {raw_feat.shape}")

    if raw_feat.shape[-1] != expected_dim:
        raise ValueError(
            f"Audio dim mismatch: got {raw_feat.shape[-1]}, expected {expected_dim}. "
            "Please check --feature_type and --audio_dim."
        )

    raw_tensor = torch.from_numpy(raw_feat).float().unsqueeze(0).transpose(1, 2)
    aligned = F.interpolate(
        raw_tensor,
        size=seq_len,
        mode="linear",
        align_corners=False,
    )
    return aligned.transpose(1, 2).squeeze(0).numpy().astype(np.float32)

def extract_audio_features(args, seq_len):
    feature_map = {
        "hybrid": hybrid_extract,
        "baseline": baseline_extract,
        "jukebox": juke_extract,
    }

    if args.feature_type not in feature_map:
        raise ValueError(f"Unknown feature_type: {args.feature_type}")

    raw_feat, _ = feature_map[args.feature_type](args.music)
    return align_audio_features(raw_feat, seq_len=seq_len, expected_dim=args.audio_dim)


def motion_to_joints(motion_physical, device):
    """
    physical 151D motion -> FK joints [T,24,3]
    """
    motion_t = torch.from_numpy(motion_physical).float().to(device)
    root = motion_t[:, 4:7].unsqueeze(0)
    q_6d = motion_t[:, 7:].reshape(1, motion_t.shape[0], 24, 6)
    q_ax = ax_from_6v(q_6d)

    with torch.no_grad():
        joints = SMPLSkeleton(device=device).forward(q_ax, root)

    return joints.detach().cpu().numpy()[0]


def apply_trajectory_postprocess(motion_physical, traj_physical, strength=1.0):
    """
    可选后处理：把 root X/Z 往目标轨迹拉。
    strength=1.0 表示完全锚定到目标轨迹。
    """
    if traj_physical is None:
        return motion_physical

    out = motion_physical.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    out[:, ROOT_X_IDX] = (1.0 - strength) * out[:, ROOT_X_IDX] + strength * traj_physical[:, 0]
    out[:, ROOT_Z_IDX] = (1.0 - strength) * out[:, ROOT_Z_IDX] + strength * traj_physical[:, 1]
    return out.astype(np.float32)

def find_contact_segments(contact_mask, min_len=3):
    """
    contact_mask: [T] bool
    return list of (start, end), end is exclusive
    """
    segments = []
    start = None

    for i, active in enumerate(contact_mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None

    if start is not None and len(contact_mask) - start >= min_len:
        segments.append((start, len(contact_mask)))

    return segments


def height_contact_mask(feet, height_threshold=0.035):
    """
    feet: [T, 4, 3]
    return: [T, 4] bool

    当 contact channel 不可靠时，用脚高度判断接触。
    """
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    return heights <= floor + float(height_threshold)


def smooth_2d_sequence(x, window=5):
    """
    x: [T, 2]
    简单 moving average，避免 root_delta 抖动。
    """
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x

    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.zeros_like(x, dtype=np.float32)

    for dim in range(x.shape[1]):
        out[:, dim] = np.convolve(
            np.pad(x[:, dim], (pad, pad), mode="edge"),
            kernel,
            mode="valid",
        )

    return out.astype(np.float32)


def choose_contact_mask(
    motion,
    feet,
    contact_threshold=0.8,
    contact_source="auto",
    height_contact_threshold=0.035,
):
    """
    return:
        contacts: [T, 4] bool
        actual_source: str
    """
    channel_contacts = motion[:, 0:4] > float(contact_threshold)
    height_contacts = height_contact_mask(
        feet,
        height_threshold=height_contact_threshold,
    )

    if contact_source == "channels":
        return channel_contacts, "channels"

    if contact_source == "height":
        return height_contacts, "height"

    # auto: contact channel 太稀疏或太饱和时，认为不可靠，回退到 height-based contact
    ratio = float(channel_contacts.mean())
    use_channels = 0.01 <= ratio <= 0.95

    if use_channels:
        return channel_contacts, "channels"

    return height_contacts, "height"


def apply_foot_lock_postprocess(
    motion_physical,
    device,
    strength=0.7,
    contact_threshold=0.8,
    min_segment=3,
    contact_source="auto",
    height_contact_threshold=0.035,
    smooth_window=5,
    max_root_delta=0.08,
):
    """
    Reduce horizontal foot sliding by adjusting root X/Z during contact segments.

    改进点：
    1. 支持 contact_source=auto/channels/height。
    2. contact channel 饱和或失效时，自动回退到脚高度接触。
    3. 对 root_delta 做平滑，避免 foot lock 自身造成抖动。
    4. 限制每帧最大 root 修正量，避免破坏轨迹太多。
    """
    motion = motion_physical.copy().astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 0:
        return motion, "disabled"

    joints = motion_to_joints(motion, device=device)  # [T, 24, 3]
    feet = joints[:, [7, 8, 10, 11], :]               # [T, 4, 3]

    contacts, actual_source = choose_contact_mask(
        motion=motion,
        feet=feet,
        contact_threshold=contact_threshold,
        contact_source=contact_source,
        height_contact_threshold=height_contact_threshold,
    )

    foot_joint_ids = [7, 8, 10, 11]
    T = motion.shape[0]

    delta_sum = np.zeros((T, 2), dtype=np.float32)
    delta_count = np.zeros((T, 1), dtype=np.float32)

    for foot_channel, joint_id in enumerate(foot_joint_ids):
        segments = find_contact_segments(
            contacts[:, foot_channel],
            min_len=min_segment,
        )

        for start, end in segments:
            anchor_xz = joints[start, joint_id, [0, 2]].copy()

            for t in range(start, end):
                current_xz = joints[t, joint_id, [0, 2]]
                delta = anchor_xz - current_xz
                delta_sum[t] += delta.astype(np.float32)
                delta_count[t, 0] += 1.0

    valid = delta_count[:, 0] > 0
    if not np.any(valid):
        return motion, actual_source

    root_delta = np.zeros((T, 2), dtype=np.float32)
    root_delta[valid] = delta_sum[valid] / np.maximum(delta_count[valid], 1e-6)

    # 限制单帧 root 修正幅度，避免把轨迹拉坏。
    max_root_delta = float(max_root_delta)
    if max_root_delta > 0:
        delta_norm = np.linalg.norm(root_delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_root_delta / np.maximum(delta_norm, 1e-8))
        root_delta = root_delta * scale

    root_delta = smooth_2d_sequence(root_delta, window=smooth_window)

    motion[:, ROOT_X_IDX] += strength * root_delta[:, 0]
    motion[:, ROOT_Z_IDX] += strength * root_delta[:, 1]

    return motion.astype(np.float32), actual_source

def main():
    parser = argparse.ArgumentParser("Controlled EDGE inference")

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument(
        "--use_raw_weights",
        action="store_true",
        help="Use raw model_state_dict instead of ema_state_dict. Default inference uses EMA weights.",
    )
    parser.add_argument("--feature_type", default="hybrid", choices=["hybrid", "baseline", "jukebox"])
    parser.add_argument("--audio_dim", type=int, default=803)

    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--model_seq_len", type=int, default=150)

    parser.add_argument("--start_pose", default="")
    parser.add_argument("--end_pose", default="")
    parser.add_argument("--mid_poses", default="")
    parser.add_argument("--mid_pose_frames", default="")
    parser.add_argument("--keyframe_space", default="normalized", choices=["normalized", "physical"])
    parser.add_argument("--preserve_keyframe_root_xz", action="store_true")
    parser.add_argument(
        "--keyframe_width",
        type=int,
        default=3,
        help="Number of frames around each keyframe to constrain during inference.",
    )

    parser.add_argument("--trajectory", default="")
    parser.add_argument("--target_traj", default="")
    parser.add_argument("--keep_trajectory_absolute", action="store_true")

    parser.add_argument("--beat_guidance_weight", type=float, default=0.0)
    parser.add_argument("--hard_keyframe_project", action="store_true")
    parser.add_argument("--no_tto", action="store_true")

    parser.add_argument("--foot_lock_postprocess", action="store_true")
    parser.add_argument("--foot_lock_strength", type=float, default=0.7)
    parser.add_argument("--foot_lock_contact_threshold", type=float, default=0.8)
    parser.add_argument("--foot_lock_min_segment", type=int, default=3)
    parser.add_argument(
        "--foot_lock_contact_source",
        choices=["auto", "channels", "height"],
        default="auto",
        help="Contact source for foot lock. auto falls back to height contact when contact channels are saturated or unreliable.",
    )
    parser.add_argument(
        "--foot_lock_height_threshold",
        type=float,
        default=0.035,
        help="Height threshold in meters for height-based foot contact.",
    )
    parser.add_argument(
        "--foot_lock_smooth_window",
        type=int,
        default=5,
        help="Temporal smoothing window for root delta in foot lock.",
    )
    parser.add_argument(
        "--foot_lock_max_root_delta",
        type=float,
        default=0.08,
        help="Maximum per-frame root XZ correction in meters. Set <=0 to disable clipping.",
    )
    parser.add_argument(
        "--restore_trajectory_after_foot_lock",
        action="store_true",
        help="After foot lock, softly blend root X/Z back to the target trajectory.",
    )
    parser.add_argument(
        "--restore_trajectory_strength",
        type=float,
        default=0.25,
        help="Soft trajectory restoration strength after foot lock. Use small value to avoid reintroducing sliding.",
    )
    parser.add_argument("--postprocess_trajectory", action="store_true")
    parser.add_argument("--postprocess_strength", type=float, default=1.0)

    parser.add_argument("--out_dir", default="output/controlled")
    parser.add_argument("--out_name", default="controlled")
    parser.add_argument("--save_normalized", action="store_true")
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading EDGE checkpoint...")
    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        audio_dim=args.audio_dim,
        seq_len=args.model_seq_len,
        EMA=not args.use_raw_weights,
        beat_guidance_weight=args.beat_guidance_weight,
        hard_keyframe_project=args.hard_keyframe_project,
    )
    model.eval()

    normalizer = model.normalizer
    if normalizer is None:
        print("Warning: checkpoint has no normalizer; generated motion will be treated as physical space.")

    print("Preparing audio condition...")
    audio_np = extract_audio_features(args, args.frames)
    audio = torch.from_numpy(audio_np).float().unsqueeze(0).to(model.accelerator.device)

    print("Preparing trajectory condition...")
    traj_physical = load_target_trajectory(args, args.frames)
    traj_norm = None

    cond = {"audio": audio}

    if traj_physical is not None:
        traj_norm = normalize_trajectory(traj_physical, normalizer)
        traj_tensor = torch.from_numpy(traj_norm).float().unsqueeze(0).to(model.accelerator.device)
        cond["trajectory"] = traj_tensor

    print("Preparing keyframe constraints...")

    constraint = build_keyframe_constraint(
        args=args,
        seq_len=args.frames,
        normalizer=normalizer,
        traj_norm=traj_norm,
    )

    if constraint is not None:
        constraint = {
            "mask": constraint["mask"].to(model.accelerator.device),
            "value": constraint["value"].to(model.accelerator.device),
        }

    print("Sampling controlled motion...")
    shape = (1, args.frames, 151)

    with torch.no_grad():
        sample_norm = model.diffusion.inpaint_loop(
            shape=shape,
            cond=cond,
            constraint=constraint,
            use_tto=not args.no_tto,
        )

    sample_norm = sample_norm[0].detach().float().cpu()

    if args.save_normalized:
        np.save(os.path.join(args.out_dir, f"{args.out_name}_normalized.npy"), sample_norm.numpy())

    if normalizer is not None:
        sample_physical = normalizer.unnormalize(sample_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
    else:
        sample_physical = sample_norm.numpy()

    raw_motion_path = os.path.join(args.out_dir, f"{args.out_name}_raw_model.npy")
    np.save(raw_motion_path, sample_physical.astype(np.float32))

    final_motion = sample_physical.copy()
    postprocess_steps = []

    if args.postprocess_trajectory and traj_physical is not None:
        final_motion = apply_trajectory_postprocess(
            motion_physical=final_motion,
            traj_physical=traj_physical,
            strength=args.postprocess_strength,
        )
        postprocess_steps.append(
            {
                "name": "trajectory_postprocess",
                "strength": float(args.postprocess_strength),
                "note": (
                    "Root X/Z is blended toward the target trajectory. "
                    "When strength=1.0, final trajectory error mainly reflects system-level anchoring, "
                    "not pure model prediction."
                ),
            }
        )

    if args.foot_lock_postprocess:
        final_motion, actual_contact_source = apply_foot_lock_postprocess(
            motion_physical=final_motion,
            device=model.accelerator.device,
            strength=args.foot_lock_strength,
            contact_threshold=args.foot_lock_contact_threshold,
            min_segment=args.foot_lock_min_segment,
            contact_source=args.foot_lock_contact_source,
            height_contact_threshold=args.foot_lock_height_threshold,
            smooth_window=args.foot_lock_smooth_window,
            max_root_delta=args.foot_lock_max_root_delta,
        )

        postprocess_steps.append(
            {
                "name": "foot_lock_postprocess",
                "strength": float(args.foot_lock_strength),
                "contact_threshold": float(args.foot_lock_contact_threshold),
                "contact_source_requested": args.foot_lock_contact_source,
                "contact_source_actual": actual_contact_source,
                "height_contact_threshold": float(args.foot_lock_height_threshold),
                "smooth_window": int(args.foot_lock_smooth_window),
                "max_root_delta": float(args.foot_lock_max_root_delta),
                "min_segment": int(args.foot_lock_min_segment),
                "note": (
                    "Root X/Z is adjusted during detected foot-contact segments to reduce sliding. "
                    "This may trade off trajectory accuracy for visual physical plausibility. "
                    "Use raw_model_motion for model-only evaluation and final_system_motion for system display."
                ),
            }
        )

        if args.restore_trajectory_after_foot_lock and traj_physical is not None:
            final_motion = apply_trajectory_postprocess(
                motion_physical=final_motion,
                traj_physical=traj_physical,
                strength=args.restore_trajectory_strength,
            )
            postprocess_steps.append(
                {
                    "name": "restore_trajectory_after_foot_lock",
                    "strength": float(args.restore_trajectory_strength),
                    "note": (
                        "Softly blends root X/Z back toward target trajectory after foot lock. "
                        "Use a small strength; too large a value may reintroduce foot sliding."
                    ),
                }
            )

    final_motion_path = os.path.join(args.out_dir, f"{args.out_name}_final_system.npy")
    np.save(final_motion_path, final_motion.astype(np.float32))

    metadata = {
        "checkpoint": args.checkpoint,
        "weight_source": "model_state_dict" if args.use_raw_weights else "ema_state_dict",
        "music": args.music,
        "frames": args.frames,
        "feature_type": args.feature_type,
        "audio_dim": args.audio_dim,
        "start_pose": args.start_pose,
        "end_pose": args.end_pose,
        "mid_poses": args.mid_poses,
        "mid_pose_frames": args.mid_pose_frames,
        "trajectory": args.trajectory,
        "target_traj": args.target_traj,
        "hard_keyframe_project": args.hard_keyframe_project,
        "use_tto": not args.no_tto,
        "postprocess_trajectory": args.postprocess_trajectory,
        "raw_model_motion": raw_motion_path,
        "final_system_motion": final_motion_path,
        "postprocess_steps": postprocess_steps,
        "interpretation": {
            "raw_model_motion": "Use this file to evaluate the model's own keyframe/trajectory following ability.",
            "final_system_motion": "Use this file for final controlled choreography display.",
        },
    }

    with open(os.path.join(args.out_dir, f"{args.out_name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    if not args.no_render:
        print("Rendering skeleton video...")
        joints = motion_to_joints(final_motion.astype(np.float32), model.accelerator.device)
        contact = final_motion[:, 0:4]

        skeleton_render(
            joints,
            epoch=args.out_name,
            out=args.out_dir,
            name=args.music,
            sound=True,
            stitch=False,
            contact=contact,
            render=True,
            camera_mode="follow",
            output_path=os.path.join(args.out_dir, f"{args.out_name}.mp4"),
        )

    print(f"Saved raw motion:   {raw_motion_path}")
    print(f"Saved final motion: {final_motion_path}")


if __name__ == "__main__":
    main()