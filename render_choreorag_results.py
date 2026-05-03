import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vis import SMPLSkeleton, skeleton_render


CONTACT_SLICE = slice(0, 4)
ROOT_POS_SLICE = slice(4, 7)
ROT6D_SLICE = slice(7, 151)


def rotation_6d_to_matrix_local(d6: torch.Tensor, layout: str = "rows") -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    if layout == "cols":
        return torch.stack((b1, b2, b3), dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle_local(R: torch.Tensor) -> torch.Tensor:
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_angle)

    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1)

    sin_angle = torch.sin(angle).unsqueeze(-1)
    axis = axis / (2.0 * sin_angle + 1e-8)
    aa = axis * angle.unsqueeze(-1)

    small = angle < 1e-4
    if small.any():
        approx = 0.5 * torch.stack([rx, ry, rz], dim=-1)
        aa = torch.where(small.unsqueeze(-1), approx, aa)

    return torch.nan_to_num(aa, nan=0.0, posinf=0.0, neginf=0.0)


def motion151_to_joints(motion151: np.ndarray, sixd_layout: str = "rows", body_centered: bool = False):
    motion = np.asarray(motion151, dtype=np.float32)

    if motion.ndim == 3 and motion.shape[0] == 1:
        motion = motion[0]

    if motion.ndim != 2 or motion.shape[1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape}")

    contacts = motion[:, CONTACT_SLICE].astype(np.float32)
    root = motion[:, ROOT_POS_SLICE].astype(np.float32).copy()

    if body_centered:
        root[:, 0] = 0.0
        root[:, 2] = 0.0

    rot6d = motion[:, ROT6D_SLICE].reshape(len(motion), 24, 6).astype(np.float32)

    root_t = torch.from_numpy(root).float()[None]
    rot6d_t = torch.from_numpy(rot6d).float()

    R = rotation_6d_to_matrix_local(rot6d_t.reshape(-1, 6), layout=sixd_layout)
    aa = matrix_to_axis_angle_local(R).reshape(1, len(motion), 24, 3)

    skel = SMPLSkeleton()
    with torch.no_grad():
        joints = skel.forward(aa, root_t)[0].cpu().numpy().astype(np.float32)

    return joints, contacts


def render_one(args):
    motion = np.load(args.motion, allow_pickle=True)
    joints, contacts = motion151_to_joints(
        motion,
        sixd_layout=args.sixd_layout,
        body_centered=args.body_centered,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skeleton_render(
        poses=joints,
        epoch=0,
        out=str(out_path.parent),
        name=args.music,
        sound=True,
        contact=contacts,
        render=True,
        camera_mode=args.camera_mode,
        output_path=str(out_path),
        render_smooth_window=args.smooth_window,
    )

    print(f"✅ rendered: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--camera_mode", default="fixed", choices=["follow", "fixed"])
    parser.add_argument("--sixd_layout", default="rows", choices=["rows", "cols"])
    parser.add_argument("--body_centered", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=7)
    args = parser.parse_args()
    render_one(args)


if __name__ == "__main__":
    main()
