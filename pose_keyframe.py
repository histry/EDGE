import argparse
import os
import pickle
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from dataset.preprocess import Normalizer
from dataset.quaternion import ax_from_6v, ax_to_6v
from vis import SMPLSkeleton


_ROMP_MODEL = None


def load_normalizer_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    norm_data = checkpoint.get("normalizer")
    if isinstance(norm_data, dict) and "mean" in norm_data:
        dummy = torch.zeros((1, 1, 151))
        normalizer = Normalizer(dummy)
        normalizer.mean = np.asarray(norm_data["mean"], dtype=np.float32)
        normalizer.std = np.asarray(norm_data["std"], dtype=np.float32)
        return normalizer
    if norm_data is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain normalizer.")
    return norm_data


def parse_xz(text: str) -> Tuple[float, float]:
    if not text:
        return 0.0, 0.0
    parts = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected X,Z pair, got: {text}")
    return float(parts[0]), float(parts[1])


def _tensor_from_target_xz(target_trans_xz, device):
    if target_trans_xz is None:
        return None
    if isinstance(target_trans_xz, str):
        target_trans_xz = parse_xz(target_trans_xz)
    if torch.is_tensor(target_trans_xz):
        return target_trans_xz.to(device=device, dtype=torch.float32).reshape(2)
    return torch.tensor(target_trans_xz, device=device, dtype=torch.float32).reshape(2)


def _load_pkl_rot_6d(pkl_path, frame_idx, device):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if "q" not in data:
        raise ValueError(f"PKL file {pkl_path} does not contain key 'q'.")

    q = torch.as_tensor(data["q"], dtype=torch.float32, device=device)
    if q.ndim == 2:
        q_frame = q[frame_idx].reshape(1, 24, 3)
    elif q.ndim == 3:
        q_frame = q[frame_idx].reshape(1, 24, 3)
    else:
        raise ValueError(f"Unsupported q shape {tuple(q.shape)} in {pkl_path}.")
    return ax_to_6v(q_frame).reshape(1, 144)


def _load_image_rot_6d(image_path, device):
    global _ROMP_MODEL
    import cv2
    import romp

    if _ROMP_MODEL is None:
        settings = romp.main.default_settings
        _ROMP_MODEL = romp.ROMP(settings)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    outputs = _ROMP_MODEL(image)
    if outputs is None or "smpl_thetas" not in outputs:
        raise RuntimeError(f"ROMP did not return smpl_thetas for {image_path}.")

    smpl_72 = outputs["smpl_thetas"][0:1, :72]
    smpl_72 = torch.as_tensor(smpl_72, dtype=torch.float32, device=device)
    return ax_to_6v(smpl_72.reshape(1, 24, 3)).reshape(1, 144)


def build_pose_151(
    image_path: Optional[str] = None,
    pkl_path: Optional[str] = None,
    normalizer=None,
    device="cuda",
    target_trans_xz: Optional[Sequence[float]] = None,
    is_flying=False,
    frame_idx=0,
    normalize=True,
):
    device = torch.device(device)
    pkl_path = pkl_path.name if hasattr(pkl_path, "name") else pkl_path

    if pkl_path and os.path.exists(pkl_path):
        rot_144 = _load_pkl_rot_6d(pkl_path, frame_idx=frame_idx, device=device)
    elif image_path and os.path.exists(image_path):
        rot_144 = _load_image_rot_6d(image_path, device=device)
    else:
        raise ValueError("Either --pkl or --image must point to an existing file.")

    temp_pos = torch.zeros(1, 1, 3, device=device, dtype=torch.float32)
    target_xz = _tensor_from_target_xz(target_trans_xz, device)
    if target_xz is not None:
        temp_pos[0, 0, 0] = target_xz[0]
        temp_pos[0, 0, 2] = target_xz[1]

    smpl = SMPLSkeleton(device=device)
    temp_q = ax_from_6v(rot_144.reshape(1, 1, 24, 6))
    joints_3d = smpl.forward(temp_q, temp_pos)
    foot_y = joints_3d[0, 0, [7, 8, 10, 11], 1]
    min_foot_y = torch.min(foot_y)

    if is_flying:
        temp_pos[0, 0, 1] = -min_foot_y + 0.5
        contacts = torch.zeros(1, 4, device=device, dtype=torch.float32)
    else:
        temp_pos[0, 0, 1] = -min_foot_y
        contacts = (foot_y - min_foot_y < 0.02).float().reshape(1, 4)

    pose_151 = torch.cat([contacts, temp_pos.squeeze(0), rot_144], dim=1)
    if normalize:
        if normalizer is None:
            raise ValueError("normalize=True requires a checkpoint normalizer.")
        pose_151 = normalizer.normalize(pose_151)
    return pose_151.reshape(151).detach().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert a pose image or expert PKL frame to an EDGE 151-D keyframe.")
    parser.add_argument("--checkpoint", required=True, help="EDGE checkpoint containing normalizer statistics")
    parser.add_argument("--image", default="", help="Input pose image. Used when --pkl is not provided.")
    parser.add_argument("--pkl", default="", help="Expert motion PKL. Takes precedence over --image.")
    parser.add_argument("--frame", type=int, default=0, help="PKL frame index to export")
    parser.add_argument("--target_xz", default="0,0", help="Physical root X,Z to assign before normalization")
    parser.add_argument("--flying", action="store_true", help="Lift root height and force foot contacts to zero")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", required=True, help="Output normalized 151-D .npy keyframe")
    parser.add_argument("--physical_out", default="", help="Optional physical-space 151-D .npy for inspection")
    args = parser.parse_args()

    normalizer = load_normalizer_from_checkpoint(args.checkpoint)
    target_xz = parse_xz(args.target_xz)

    pose_norm = build_pose_151(
        image_path=args.image,
        pkl_path=args.pkl,
        normalizer=normalizer,
        device=args.device,
        target_trans_xz=target_xz,
        is_flying=args.flying,
        frame_idx=args.frame,
        normalize=True,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, pose_norm)

    if args.physical_out:
        pose_phys = normalizer.unnormalize(pose_norm.reshape(1, 1, 151)).reshape(151).astype(np.float32)
        os.makedirs(os.path.dirname(args.physical_out) or ".", exist_ok=True)
        np.save(args.physical_out, pose_phys)

    print(f"Saved normalized keyframe: {args.out}")


if __name__ == "__main__":
    main()
