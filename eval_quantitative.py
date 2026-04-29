import argparse
import csv
import json
import math
import os
from pathlib import Path

import librosa
import numpy as np
import scipy.signal
import scipy.interpolate as spi
from scipy.signal import find_peaks
import torch

from dataset.preprocess import Normalizer
from dataset.quaternion import ax_from_6v, quat_from_6v
from vis import SMPLSkeleton

if not hasattr(scipy.signal, "hann") and hasattr(scipy.signal, "windows"):
    scipy.signal.hann = scipy.signal.windows.hann


def load_motion(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if "motion" in data:
            arr = data["motion"]
        else:
            raise ValueError(f"{path} is a dict .npy without a 'motion' key")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected motion shape [T,151], got {arr.shape} from {path}")
    return arr


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_normalizer(checkpoint_path):
    if not checkpoint_path:
        return None
    checkpoint = torch_load(checkpoint_path)
    norm_data = checkpoint.get("normalizer") if isinstance(checkpoint, dict) else None
    if norm_data is None:
        raise ValueError(f"No normalizer found in checkpoint: {checkpoint_path}")

    if isinstance(norm_data, dict) and "mean" in norm_data and "std" in norm_data:
        normalizer = Normalizer(torch.zeros((1, 1, 151)))
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer

    if hasattr(norm_data, "mean") and hasattr(norm_data, "std"):
        return norm_data

    raise ValueError(f"Unsupported normalizer format: {type(norm_data)}")


def parse_list(text):
    if not text:
        return []
    return [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]


def parse_frame_list(text, seq_len):
    frames = []
    for item in parse_list(text):
        value = float(item)
        if 0.0 < value < 1.0:
            value = value * (seq_len - 1)
        frames.append(int(round(value)))
    return [max(0, min(seq_len - 1, frame)) for frame in frames]


def parse_mid_frames(args, num_mid, seq_len):
    if num_mid == 0:
        return []
    if args.mid_pose_frames:
        frames = parse_frame_list(args.mid_pose_frames, seq_len)
    elif args.mid_pose_ratios:
        frames = parse_frame_list(args.mid_pose_ratios, seq_len)
    else:
        frames = [
            int(round((idx + 1) * (seq_len - 1) / (num_mid + 1)))
            for idx in range(num_mid)
        ]
    if len(frames) != num_mid:
        raise ValueError(f"Got {num_mid} mid poses but {len(frames)} mid frames")
    return [max(1, min(seq_len - 2, frame)) for frame in frames]


def load_pose(path, normalizer=None, pose_space="auto"):
    pose = np.asarray(np.load(path), dtype=np.float32)
    pose = pose.reshape(-1)
    if pose.shape[0] != 151:
        raise ValueError(f"Expected 151-D keyframe pose, got {pose.shape} from {path}")

    if pose_space == "auto":
        pose_space = "normalized" if normalizer is not None else "physical"
    if pose_space == "normalized":
        if normalizer is None:
            raise ValueError("--keyframe_space normalized requires --checkpoint")
        pose = normalizer.unnormalize(pose[None, None, :]).reshape(151).astype(np.float32)
    elif pose_space != "physical":
        raise ValueError(f"Unknown keyframe space: {pose_space}")
    return pose


def motion_to_joints(motion, device="cpu"):
    device = torch.device(device)
    root = torch.tensor(motion[:, 4:7], device=device, dtype=torch.float32).unsqueeze(0)
    q_6d = torch.tensor(motion[:, 7:], device=device, dtype=torch.float32).reshape(1, motion.shape[0], 24, 6)
    with torch.no_grad():
        q_ax = ax_from_6v(q_6d)
        joints = SMPLSkeleton(device=device).forward(q_ax, root)
    return joints.detach().cpu().numpy()[0]


def pose_to_joints(pose, device="cpu"):
    return motion_to_joints(pose.reshape(1, 151), device=device)[0]


def rotation_angle_error_deg(gen_pose, target_pose):
    gen_q = quat_from_6v(torch.tensor(gen_pose[7:].reshape(24, 6), dtype=torch.float32))
    target_q = quat_from_6v(torch.tensor(target_pose[7:].reshape(24, 6), dtype=torch.float32))
    dot = torch.sum(gen_q * target_q, dim=-1).abs().clamp(max=1.0)
    angles = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angles).mean().item())


def eval_keyframes(motion, args, normalizer=None):
    targets = []
    if args.start_pose:
        targets.append(("start", 0, args.start_pose))
    mid_paths = parse_list(args.mid_poses)
    for idx, (frame, path) in enumerate(zip(parse_mid_frames(args, len(mid_paths), len(motion)), mid_paths), start=1):
        targets.append((f"mid{idx}", frame, path))
    if args.end_pose:
        targets.append(("end", len(motion) - 1, args.end_pose))

    if not targets:
        return {}, []

    rows = []
    feature_idx = np.r_[5, 7:151]
    for name, frame, path in targets:
        target = load_pose(path, normalizer=normalizer, pose_space=args.keyframe_space)
        gen = motion[frame].copy()
        tgt = target.copy()
        if args.keyframe_ignore_root_xz:
            gen[[4, 6]] = 0.0
            tgt[[4, 6]] = 0.0

        gen_joints = pose_to_joints(gen, device=args.device)
        tgt_joints = pose_to_joints(tgt, device=args.device)
        joint_err = np.linalg.norm(gen_joints - tgt_joints, axis=-1)
        feature_rmse = float(np.sqrt(np.mean((gen[feature_idx] - tgt[feature_idx]) ** 2)))

        rows.append(
            {
                "name": name,
                "frame": int(frame),
                "path": path,
                "mpjpe_m": float(joint_err.mean()),
                "mpjpe_cm": float(joint_err.mean() * 100.0),
                "max_joint_err_m": float(joint_err.max()),
                "rot_err_deg": rotation_angle_error_deg(gen, tgt),
                "feature_rmse": feature_rmse,
                "root_y_abs_err_m": float(abs(gen[5] - tgt[5])),
            }
        )

    summary = {
        "keyframe_count": len(rows),
        "keyframe_mpjpe_m_mean": float(np.mean([r["mpjpe_m"] for r in rows])),
        "keyframe_mpjpe_m_max": float(np.max([r["mpjpe_m"] for r in rows])),
        "keyframe_rot_err_deg_mean": float(np.mean([r["rot_err_deg"] for r in rows])),
        "keyframe_rot_err_deg_max": float(np.max([r["rot_err_deg"] for r in rows])),
        "keyframe_feature_rmse_mean": float(np.mean([r["feature_rmse"] for r in rows])),
    }
    return summary, rows


def onset_strength_for_audio(audio_path, seq_len, fps):
    if not audio_path:
        return None
    hop_length = 512
    sr = int(fps * hop_length)
    y, _ = librosa.load(audio_path, sr=sr)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).astype(np.float32)
    if len(onset) == 0:
        return None
    if len(onset) != seq_len:
        x_old = np.linspace(0.0, 1.0, len(onset))
        x_new = np.linspace(0.0, 1.0, seq_len)
        onset = np.interp(x_new, x_old, onset).astype(np.float32)
    return onset


def trajectory_from_control_points(control_points_str, seq_len, audio_path=None, fps=30, use_audio_timing=True):
    points = []
    if control_points_str:
        for item in control_points_str.split(";"):
            item = item.strip()
            if not item:
                continue
            x_text, z_text = item.split(",")
            points.append([float(x_text.strip()), float(z_text.strip())])

    if not points:
        return None
    if len(points) == 1:
        return np.tile(np.asarray(points[0], dtype=np.float32), (seq_len, 1))

    pts = np.asarray(points, dtype=np.float32).T
    k_val = min(3, len(points) - 1)
    tck, _ = spi.splprep(pts, s=0, k=k_val)

    if use_audio_timing:
        onset = onset_strength_for_audio(audio_path, seq_len, fps)
    else:
        onset = None

    if onset is not None:
        bias = 0.2 * float(np.max(onset)) if float(np.max(onset)) > 0 else 1.0
        speed_curve = onset + bias
        progress = np.cumsum(speed_curve)
        denom = max(float(progress[-1] - progress[0]), 1e-8)
        u_new = (progress - progress[0]) / denom
    else:
        u_new = np.linspace(0.0, 1.0, seq_len)

    x_new, z_new = spi.splev(u_new, tck)
    return np.stack([x_new, z_new], axis=1).astype(np.float32)


def load_target_trajectory(args, seq_len):
    if args.target_traj:
        target = np.asarray(np.load(args.target_traj), dtype=np.float32)
        if target.ndim == 3:
            target = target[0]
        target = target[:seq_len, :2]
    else:
        target = trajectory_from_control_points(
            args.trajectory,
            seq_len,
            audio_path=args.audio,
            fps=args.fps,
            use_audio_timing=not args.uniform_trajectory_timing,
        )
    if target is None:
        return None
    if len(target) != seq_len:
        x_old = np.linspace(0.0, 1.0, len(target))
        x_new = np.linspace(0.0, 1.0, seq_len)
        target = np.stack(
            [np.interp(x_new, x_old, target[:, 0]), np.interp(x_new, x_old, target[:, 1])],
            axis=1,
        ).astype(np.float32)
    if not args.keep_trajectory_absolute:
        target = target - target[0:1]
    return target


def dtw_distance_per_frame(a, b):
    n, m = len(a), len(b)
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(n, m))


def eval_trajectory(motion, args):
    target = load_target_trajectory(args, len(motion))
    if target is None:
        return {}
    gen = motion[:, [4, 6]].astype(np.float32)
    if not args.keep_trajectory_absolute:
        gen = gen - gen[0:1]
    err = np.linalg.norm(gen - target, axis=1)
    return {
        "trajectory_ade_m": float(err.mean()),
        "trajectory_rmse_m": float(np.sqrt(np.mean(err ** 2))),
        "trajectory_max_error_m": float(err.max()),
        "trajectory_final_error_m": float(err[-1]),
        "trajectory_dtw_m_per_frame": dtw_distance_per_frame(gen, target),
        "trajectory_path_len_gen_m": float(np.linalg.norm(np.diff(gen, axis=0), axis=1).sum()),
        "trajectory_path_len_target_m": float(np.linalg.norm(np.diff(target, axis=0), axis=1).sum()),
    }


def prefixed(metrics, prefix):
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def eval_raw_post_trajectory(args, primary_motion):
    raw_path = getattr(args, "raw_motion", "")
    post_path = getattr(args, "post_motion", "")
    if not raw_path and not post_path:
        return {}

    raw_motion = load_motion(raw_path) if raw_path else primary_motion
    post_motion = load_motion(post_path) if post_path else primary_motion

    raw_metrics = eval_trajectory(raw_motion, args)
    post_metrics = eval_trajectory(post_motion, args)
    out = {}
    out.update(prefixed(raw_metrics, "raw_"))
    out.update(prefixed(post_metrics, "post_"))

    if raw_metrics and post_metrics:
        for key in sorted(set(raw_metrics) & set(post_metrics)):
            raw_value = raw_metrics[key]
            post_value = post_metrics[key]
            if isinstance(raw_value, float) and isinstance(post_value, float):
                out[f"post_minus_raw_{key}"] = post_value - raw_value
    return out


def contact_mask_from_height(feet, height_threshold):
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    return heights <= floor + height_threshold


def eval_foot_sliding(motion, joints, args):
    feet = joints[:, [7, 8, 10, 11], :]
    horizontal_speed = np.linalg.norm(feet[1:, :, [0, 2]] - feet[:-1, :, [0, 2]], axis=-1) * args.fps

    channel_contacts = motion[:, 0:4] > args.contact_threshold
    height_contacts = contact_mask_from_height(feet, args.height_contact_threshold)
    if args.contact_source == "channels":
        contacts = channel_contacts
        actual_source = "channels"
    elif args.contact_source == "height":
        contacts = height_contacts
        actual_source = "height"
    else:
        ratio = float(channel_contacts.mean())
        use_channels = 0.01 <= ratio <= 0.95
        contacts = channel_contacts if use_channels else height_contacts
        actual_source = "channels" if use_channels else "height"

    contact_pairs = contacts[1:] & contacts[:-1]
    denom = int(contact_pairs.sum())
    if denom == 0:
        return {
            "foot_contact_source": args.contact_source,
            "foot_contact_source_actual": actual_source,
            "foot_contact_ratio": float(contacts.mean()),
            "foot_contact_pair_count": 0,
            "foot_slide_rate": float("nan"),
        }

    contact_speeds = horizontal_speed[contact_pairs]
    slide_mask = contact_pairs & (horizontal_speed > args.slide_speed_threshold)
    slide_speeds = horizontal_speed[slide_mask]
    return {
        "foot_contact_source": args.contact_source,
        "foot_contact_source_actual": actual_source,
        "foot_contact_ratio": float(contacts.mean()),
        "foot_contact_pair_count": denom,
        "foot_slide_rate": float(slide_mask.sum() / denom),
        "foot_contact_speed_mean_mps": float(contact_speeds.mean()),
        "foot_contact_speed_p95_mps": float(np.percentile(contact_speeds, 95)),
        "foot_slide_speed_mean_mps": float(slide_speeds.mean()) if len(slide_speeds) else 0.0,
        "foot_slide_speed_max_mps": float(slide_speeds.max()) if len(slide_speeds) else 0.0,
    }


def smooth_sequence(x, window):
    window = max(1, int(window))
    if window <= 1 or len(x) < 3:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(np.pad(x, (pad, pad), mode="edge"), kernel, mode="valid")


def motion_beats_from_joints(joints, args):
    velocity = np.linalg.norm(joints[1:] - joints[:-1], axis=-1).mean(axis=-1) * args.fps
    velocity = smooth_sequence(velocity, args.motion_beat_smooth_window)
    if len(velocity) < 3:
        return np.asarray([], dtype=np.int64), velocity
    prominence = max(float(np.std(velocity) * args.motion_beat_prominence_scale), 1e-8)
    peaks, _ = find_peaks(
        -velocity,
        distance=max(1, int(args.motion_beat_min_distance)),
        prominence=prominence,
    )
    return (peaks + 1).astype(np.int64), velocity


def audio_beats_from_file(audio_path, fps):
    if not audio_path:
        return np.asarray([], dtype=np.int64)
    y, sr = librosa.load(audio_path, sr=None)
    if len(y) == 0:
        return np.asarray([], dtype=np.int64)
    _, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return np.round(np.asarray(beat_times) * fps).astype(np.int64)


def gaussian_alignment_score(query_beats, ref_beats, sigma):
    if len(query_beats) == 0 or len(ref_beats) == 0:
        return float("nan")
    distances = np.min(np.abs(query_beats[:, None] - ref_beats[None, :]), axis=1)
    return float(np.mean(np.exp(-(distances ** 2) / (2.0 * sigma ** 2))))


def eval_beatalign(joints, args):
    if not args.audio:
        return {}
    motion_beats, velocity = motion_beats_from_joints(joints, args)
    audio_beats = audio_beats_from_file(args.audio, args.fps)
    max_frame = len(joints) - 1
    audio_beats = audio_beats[(audio_beats >= 0) & (audio_beats <= max_frame)]

    m2a = gaussian_alignment_score(motion_beats, audio_beats, args.beatalign_sigma_frames)
    a2m = gaussian_alignment_score(audio_beats, motion_beats, args.beatalign_sigma_frames)
    symmetric = float(np.nanmean([m2a, a2m])) if not (math.isnan(m2a) and math.isnan(a2m)) else float("nan")
    return {
        "beatalign_motion_to_audio": m2a,
        "beatalign_audio_to_motion": a2m,
        "beatalign_symmetric": symmetric,
        "motion_beat_count": int(len(motion_beats)),
        "audio_beat_count": int(len(audio_beats)),
        "motion_velocity_mean_mps": float(np.mean(velocity)) if len(velocity) else 0.0,
    }


def finite_json(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    return value


def print_summary(metrics, keyframe_rows):
    print("\nQuantitative evaluation")
    print("-----------------------")
    for key in sorted(metrics.keys()):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}" if not math.isnan(value) else f"{key}: nan")
        else:
            print(f"{key}: {value}")
    if keyframe_rows:
        print("\nKeyframe details")
        for row in keyframe_rows:
            print(
                f"- {row['name']} frame={row['frame']} "
                f"MPJPE={row['mpjpe_cm']:.2f}cm "
                f"Rot={row['rot_err_deg']:.2f}deg "
                f"RMSE={row['feature_rmse']:.4f}"
            )


def write_outputs(args, metrics, keyframe_rows):
    payload = {"metrics": finite_json(metrics), "keyframes": finite_json(keyframe_rows)}
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key in sorted(metrics.keys()):
                writer.writerow([key, finite_json(metrics[key])])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated EDGE 151-D .npy motions.")
    parser.add_argument("--motion", required=True, help="Generated 151-D motion .npy")
    parser.add_argument("--raw_motion", default="", help="Optional pre-trajectory-postprocess .npy for raw/post trajectory comparison")
    parser.add_argument("--post_motion", default="", help="Optional postprocessed .npy for raw/post trajectory comparison")
    parser.add_argument("--checkpoint", default="", help="Checkpoint with normalizer for normalized keyframe .npy files")
    parser.add_argument("--audio", default="", help="Audio .wav for BeatAlign and audio-timed trajectory reconstruction")
    parser.add_argument("--trajectory", default="", help="Control points like '0,0;1,2;-1,4'")
    parser.add_argument("--target_traj", default="", help="Optional target trajectory .npy, shape [T,2] or [1,T,2]")
    parser.add_argument("--uniform_trajectory_timing", action="store_true", help="Do not use audio onset timing for trajectory reconstruction")
    parser.add_argument("--keep_trajectory_absolute", action="store_true", help="Do not subtract first X/Z point before trajectory evaluation")
    parser.add_argument("--start_pose", default="", help="Normalized or physical 151-D start keyframe .npy")
    parser.add_argument("--end_pose", default="", help="Normalized or physical 151-D end keyframe .npy")
    parser.add_argument("--mid_poses", default="", help="Comma/semicolon separated mid keyframe .npy files")
    parser.add_argument("--mid_pose_frames", default="", help="Comma separated mid frame ids; values in (0,1) are ratios")
    parser.add_argument("--mid_pose_ratios", default="", help="Comma separated mid ratios")
    parser.add_argument("--keyframe_space", choices=["auto", "normalized", "physical"], default="auto")
    parser.add_argument("--keyframe_ignore_root_xz", action="store_true", default=True)
    parser.add_argument("--keyframe_include_root_xz", dest="keyframe_ignore_root_xz", action="store_false")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--contact_source", choices=["auto", "channels", "height"], default="auto")
    parser.add_argument("--contact_threshold", type=float, default=0.8)
    parser.add_argument("--height_contact_threshold", type=float, default=0.035)
    parser.add_argument("--slide_speed_threshold", type=float, default=0.05, help="m/s threshold for sliding during contact")
    parser.add_argument("--beatalign_sigma_frames", type=float, default=3.0)
    parser.add_argument("--motion_beat_smooth_window", type=int, default=5)
    parser.add_argument("--motion_beat_min_distance", type=int, default=6)
    parser.add_argument("--motion_beat_prominence_scale", type=float, default=0.05)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_csv", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    motion = load_motion(args.motion)
    normalizer = load_normalizer(args.checkpoint)
    joints = motion_to_joints(motion, device=args.device)

    metrics = {
        "motion": os.path.abspath(args.motion),
        "num_frames": int(len(motion)),
        "fps": float(args.fps),
    }
    keyframe_summary, keyframe_rows = eval_keyframes(motion, args, normalizer=normalizer)
    metrics.update(keyframe_summary)
    metrics.update(eval_trajectory(motion, args))
    metrics.update(eval_raw_post_trajectory(args, motion))
    metrics.update(eval_foot_sliding(motion, joints, args))
    metrics.update(eval_beatalign(joints, args))

    print_summary(metrics, keyframe_rows)
    write_outputs(args, metrics, keyframe_rows)


if __name__ == "__main__":
    main()
