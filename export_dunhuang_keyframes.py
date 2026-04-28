import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

import numpy.core

sys.modules["numpy._core"] = numpy.core
sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
sys.modules["numpy._core.umath"] = numpy.core.umath

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v
from vis import SMPLSkeleton


def load_normalizer(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm_data = checkpoint.get("normalizer")
    if norm_data is None:
        raise ValueError(f"No normalizer found in checkpoint: {checkpoint_path}")

    if isinstance(norm_data, dict) and "mean" in norm_data and "std" in norm_data:
        normalizer = Normalizer(torch.zeros((1, 1, 151)))
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer

    if hasattr(norm_data, "mean") and hasattr(norm_data, "std"):
        return norm_data

    raise ValueError(f"Unsupported normalizer format in checkpoint: {type(norm_data)}")


def resolve_pkl(data_path, name_or_path):
    if os.path.isfile(name_or_path):
        return name_or_path

    candidates = [
        os.path.join(data_path, name_or_path),
        os.path.join(data_path, f"{name_or_path}.pkl"),
        os.path.join(data_path, "processed", name_or_path),
        os.path.join(data_path, "processed", f"{name_or_path}.pkl"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find PKL for {name_or_path}")


def load_motion_151(pkl_path):
    data = pickle.load(open(pkl_path, "rb"))
    pos = torch.tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float32).unsqueeze(0)
    q_axis = q.reshape(q.shape[0], q.shape[1], -1, 3)

    smpl = SMPLSkeleton()
    positions = smpl.forward(q_axis, pos)
    feet = positions[:, :, (7, 8, 10, 11)]
    feetv = torch.zeros(feet.shape[:3], dtype=feet.dtype)
    feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
    contacts = (feetv < 0.01).to(q_axis)

    q_6v = ax_to_6v(q_axis)
    motion = vectorize_many([contacts, pos, q_6v]).squeeze(0).float().numpy()
    if motion.shape[1] != 151:
        raise ValueError(f"Expected 151-D motion, got {motion.shape} from {pkl_path}")
    return motion


def clamp_frame(frame, num_frames):
    if frame < 0:
        frame = num_frames + frame
    return max(0, min(frame, num_frames - 1))


def export_pose(motion, frame_idx, normalizer, out_path, zero_root_xz=True):
    frame_idx = clamp_frame(frame_idx, len(motion))
    pose = motion[frame_idx].copy()
    if zero_root_xz:
        pose[4] = 0.0
        pose[6] = 0.0

    pose_norm = normalizer.normalize(pose[None, None, :]).reshape(151).astype(np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, pose_norm)
    return frame_idx, pose, pose_norm


def main():
    parser = argparse.ArgumentParser(
        description="Export normalized Dunhuang keyframe .npy files for inference_music.py."
    )
    parser.add_argument("--data_path", default="data/dunhuang_bvh/processed")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start_file", default="")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_file", default="")
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--mid_file", default="", help="optional PKL file for middle keyframes; defaults to start_file")
    parser.add_argument("--mid_frames", default="", help="comma separated middle frame indices to export")
    parser.add_argument("--out_dir", default="test_keyframes")
    parser.add_argument("--prefix", default="dunhuang")
    parser.add_argument("--keep_root_xz", action="store_true")
    parser.add_argument("--list", action="store_true", help="list available processed PKL files")
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.data_path, "*.pkl")))
    if args.list:
        for pkl_file in pkl_files:
            data = pickle.load(open(pkl_file, "rb"))
            print(f"{os.path.basename(pkl_file)}\tframes={len(data['pos'])}")
        return

    if not args.start_file:
        raise ValueError("--start_file is required unless --list is set")

    start_pkl = resolve_pkl(args.data_path, args.start_file)
    end_pkl = resolve_pkl(args.data_path, args.end_file or args.start_file)
    mid_pkl = resolve_pkl(args.data_path, args.mid_file or args.start_file) if args.mid_frames else ""
    normalizer = load_normalizer(args.checkpoint)

    start_motion = load_motion_151(start_pkl)
    end_motion = start_motion if start_pkl == end_pkl else load_motion_151(end_pkl)
    mid_motion = None
    if args.mid_frames:
        mid_motion = start_motion if mid_pkl == start_pkl else load_motion_151(mid_pkl)

    start_out = os.path.join(args.out_dir, f"{args.prefix}_start.npy")
    end_out = os.path.join(args.out_dir, f"{args.prefix}_end.npy")

    start_idx, start_pose, _ = export_pose(
        start_motion,
        args.start_frame,
        normalizer,
        start_out,
        zero_root_xz=not args.keep_root_xz,
    )
    end_idx, end_pose, _ = export_pose(
        end_motion,
        args.end_frame,
        normalizer,
        end_out,
        zero_root_xz=not args.keep_root_xz,
    )

    mid_exports = []
    if args.mid_frames:
        for mid_i, frame_text in enumerate(args.mid_frames.replace(";", ",").split(","), start=1):
            frame_text = frame_text.strip()
            if not frame_text:
                continue
            mid_out = os.path.join(args.out_dir, f"{args.prefix}_mid{mid_i}.npy")
            mid_idx, mid_pose, _ = export_pose(
                mid_motion,
                int(float(frame_text)),
                normalizer,
                mid_out,
                zero_root_xz=not args.keep_root_xz,
            )
            mid_exports.append((mid_out, mid_idx, mid_pose))

    print("Exported keyframes:")
    print(f"  start: {start_out}")
    print(f"         source={os.path.basename(start_pkl)} frame={start_idx} root_xz=({start_pose[4]:.4f}, {start_pose[6]:.4f})")
    print(f"  end:   {end_out}")
    print(f"         source={os.path.basename(end_pkl)} frame={end_idx} root_xz=({end_pose[4]:.4f}, {end_pose[6]:.4f})")
    for mid_out, mid_idx, mid_pose in mid_exports:
        print(f"  mid:   {mid_out}")
        print(f"         source={os.path.basename(mid_pkl)} frame={mid_idx} root_xz=({mid_pose[4]:.4f}, {mid_pose[6]:.4f})")
    print("Use these files as --start_pose, --end_pose and optional --mid_poses in inference_music.py.")


if __name__ == "__main__":
    main()
