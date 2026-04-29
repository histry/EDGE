import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v


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


def load_motion_npy(path):
    arr = np.load(path, allow_pickle=True)
    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if "motion" in data:
            arr = data["motion"]
        else:
            raise ValueError(f"{path} is a dict npy but has no 'motion' key")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2 or arr.shape[1] != 151:
        raise ValueError(f"Expected motion shape [T,151], got {arr.shape}")

    return arr


def pkl_to_physical_151d(path):
    """
    从敦煌 pkl 中提取物理空间 151D motion。

    要求 pkl 包含:
        pos: [T, 3]
        q:   [T, 72]
    """
    data = pickle.load(open(path, "rb"))
    pos = torch.tensor(data["pos"], dtype=torch.float32).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float32).unsqueeze(0)

    if q.ndim != 3 or q.shape[-1] != 72:
        raise ValueError(f"Expected q shape [T,72], got {q.shape}")

    q = q.reshape(q.shape[0], q.shape[1], 24, 3)

    # 这里没有重新 FK 计算 contact，先给零 contact。
    # 如果要更严格，可以复用 DunhuangDataset 里的 SMPLSkeleton contact 逻辑。
    contacts = torch.zeros((1, q.shape[1], 4), dtype=torch.float32)
    q_6d = ax_to_6v(q)

    motion = vectorize_many([contacts, pos, q_6d]).squeeze(0).numpy().astype(np.float32)

    if motion.shape[-1] != 151:
        raise RuntimeError(f"Internal error: expected 151D, got {motion.shape}")

    return motion


def maybe_normalize_pose(pose_physical, normalizer, output_space):
    if output_space == "physical":
        return pose_physical.astype(np.float32)

    if output_space == "normalized":
        if normalizer is None:
            raise ValueError("--output_space normalized requires --checkpoint")
        pose_t = torch.from_numpy(pose_physical).float().view(1, 1, 151)
        pose_norm = normalizer.normalize(pose_t).view(151).numpy()
        return pose_norm.astype(np.float32)

    raise ValueError(f"Unknown output_space: {output_space}")


def main():
    parser = argparse.ArgumentParser("Prepare 151D keyframe pose")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--motion_npy", default="", help="Existing [T,151] motion .npy")
    source.add_argument("--motion_pkl", default="", help="Dunhuang pkl with pos/q")
    source.add_argument("--skeleton2d_json", default="", help="Reserved for future 2D-to-3D preprocessing")

    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--checkpoint", default="", help="Checkpoint containing normalizer")
    parser.add_argument("--input_space", default="physical", choices=["physical", "normalized"])
    parser.add_argument("--output_space", default="normalized", choices=["physical", "normalized"])
    parser.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.skeleton2d_json:
        raise NotImplementedError(
            "2D skeleton -> 151D SMPL keyframe is not implemented in this script yet. "
            "Current supported workflow: extract 151D keyframes from motion_npy or motion_pkl. "
            "For real 2D input, add a pose-lifting + SMPL fitting/IK module before this step."
        )

    normalizer = load_normalizer(args.checkpoint) if args.checkpoint else None

    if args.motion_npy:
        motion = load_motion_npy(args.motion_npy)

        frame = max(0, min(len(motion) - 1, args.frame))
        pose = motion[frame].astype(np.float32)

        if args.input_space == "normalized" and args.output_space == "physical":
            if normalizer is None:
                raise ValueError("normalized -> physical requires --checkpoint")
            pose_t = torch.from_numpy(pose).float().view(1, 1, 151)
            pose = normalizer.unnormalize(pose_t).view(151).numpy().astype(np.float32)
        elif args.input_space == "physical" and args.output_space == "normalized":
            pose = maybe_normalize_pose(pose, normalizer, output_space="normalized")

    else:
        motion = pkl_to_physical_151d(args.motion_pkl)
        frame = max(0, min(len(motion) - 1, args.frame))
        pose_physical = motion[frame]
        pose = maybe_normalize_pose(pose_physical, normalizer, args.output_space)

    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    np.save(args.out, pose.astype(np.float32))

    print(f"Saved keyframe: {args.out}")
    print(f"shape: {pose.shape}, output_space: {args.output_space}")


if __name__ == "__main__":
    main()