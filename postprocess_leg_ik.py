import argparse
from pathlib import Path

import numpy as np
import torch

from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton


ROOT_SLICE = slice(4, 7)
ROT_SLICE = slice(7, 151)

FOOT_JOINTS = [7, 8, 10, 11]      # lankle, rankle, ltoes, rtoes
LEFT_LEG_JOINTS = [1, 4, 7]       # lhip, lknee, lankle
RIGHT_LEG_JOINTS = [2, 5, 8]      # rhip, rknee, rankle
IK_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEG_JOINTS


def load_motion(path: str) -> np.ndarray:
    motion = np.load(path, allow_pickle=True)

    if motion.ndim == 0 and isinstance(motion.item(), dict):
        data = motion.item()
        if "motion" in data:
            motion = data["motion"]
        else:
            raise ValueError(f"{path} is dict npy but has no 'motion' key")

    motion = np.asarray(motion, dtype=np.float32)

    if motion.ndim != 2 or motion.shape[1] != 151:
        raise ValueError(f"Expected [T,151], got {motion.shape} from {path}")

    return motion


def motion_to_joints_torch(motion_t: torch.Tensor, smpl: SMPLSkeleton):
    """
    motion_t: [T,151]
    return joints: [T,24,3]
    """
    root = motion_t[:, ROOT_SLICE].unsqueeze(0)
    rot6d = motion_t[:, ROT_SLICE].reshape(1, motion_t.shape[0], 24, 6)
    q_ax = ax_from_6v(rot6d)
    joints = smpl.forward(q_ax, root)
    return joints[0]


def motion_to_joints_np(motion: np.ndarray, device="cpu") -> np.ndarray:
    device = torch.device(device)
    smpl = SMPLSkeleton(device=device)
    motion_t = torch.from_numpy(motion).float().to(device)

    with torch.no_grad():
        joints = motion_to_joints_torch(motion_t, smpl)

    return joints.detach().cpu().numpy()


def height_contact_mask(
    joints: np.ndarray,
    height_threshold: float = 0.035,
) -> np.ndarray:
    """
    和 eval_quantitative.py 的 height contact 逻辑保持一致：
    只按脚高度判断接触，不再用速度过滤。
    """
    feet = joints[:, FOOT_JOINTS, :]
    heights = feet[:, :, 1]
    floor = np.percentile(heights, 2)
    contacts = heights <= floor + float(height_threshold)
    return contacts.astype(bool)


def contiguous_segments(mask_1d: np.ndarray, min_len: int = 3):
    mask_1d = np.asarray(mask_1d, dtype=bool)
    segments = []
    start = None

    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                segments.append((start, i))
            start = None

    if start is not None and len(mask_1d) - start >= min_len:
        segments.append((start, len(mask_1d)))

    return segments


def build_foot_lock_targets(
    joints: np.ndarray,
    contacts: np.ndarray,
    min_contact_len: int = 3,
):
    """
    返回：
    target_pos: [T,4,3]
    target_mask: [T,4]
    """
    feet = joints[:, FOOT_JOINTS, :]
    target = feet.copy()
    mask = np.zeros(contacts.shape, dtype=bool)

    segment_count = 0

    for foot_i in range(4):
        segments = contiguous_segments(contacts[:, foot_i], min_len=min_contact_len)

        for start, end in segments:
            segment_count += 1
            lock_pos = np.median(feet[start:end, foot_i, :], axis=0)
            target[start:end, foot_i, :] = lock_pos[None, :]
            mask[start:end, foot_i] = True

    return target.astype(np.float32), mask.astype(bool), segment_count


def optimize_leg_ik(
    motion: np.ndarray,
    device: str = "cpu",
    height_threshold: float = 0.035,
    min_contact_len: int = 3,
    steps: int = 160,
    lr: float = 0.03,
    foot_weight: float = 1.0,
    reg_weight: float = 0.03,
    smooth_weight: float = 0.02,
    y_weight: float = 0.15,
    verbose: bool = True,
):
    """
    Optimization-based leg IK.

    固定：
    - root translation
    - upper body rotations
    - trajectory

    优化：
    - left hip/knee/ankle
    - right hip/knee/ankle

    损失：
    - 接触脚 ankle/toe 世界坐标接近锁定位置
    - 腿部旋转不要偏离原动作太远
    - 腿部旋转时间上平滑
    """
    device = torch.device(device)

    motion = motion.astype(np.float32).copy()
    joints0 = motion_to_joints_np(motion, device=device)

    contacts = height_contact_mask(
        joints0,
        height_threshold=height_threshold,
    )

    target_feet, target_mask, segment_count = build_foot_lock_targets(
        joints0,
        contacts,
        min_contact_len=min_contact_len,
    )

    if segment_count == 0 or not target_mask.any():
        print("⚠️ 没有检测到有效接触段，IK 不执行。")
        return motion, {
            "ik_contact_ratio": float(contacts.mean()),
            "ik_segment_count": int(segment_count),
            "ik_executed": False,
        }

    smpl = SMPLSkeleton(device=device)

    motion_base = torch.from_numpy(motion).float().to(device)
    rot6d_base = motion_base[:, ROT_SLICE].reshape(motion.shape[0], 24, 6)

    ik_joint_indices = torch.tensor(IK_JOINTS, dtype=torch.long, device=device)
    ik_rot = rot6d_base[:, IK_JOINTS, :].clone().detach().requires_grad_(True)
    ik_rot_orig = ik_rot.detach().clone()

    target_feet_t = torch.from_numpy(target_feet).float().to(device)
    target_mask_t = torch.from_numpy(target_mask).bool().to(device)

    optimizer = torch.optim.Adam([ik_rot], lr=float(lr))

    best_loss = None
    best_ik_rot = None

    for step in range(int(steps)):
        optimizer.zero_grad()

        motion_t = motion_base.clone()
        rot6d = rot6d_base.clone()
        rot6d[:, IK_JOINTS, :] = ik_rot
        motion_t[:, ROT_SLICE] = rot6d.reshape(motion.shape[0], -1)

        joints = motion_to_joints_torch(motion_t, smpl)
        feet = joints[:, FOOT_JOINTS, :]

        if target_mask_t.any():
            foot_diff_xz = feet[:, :, [0, 2]] - target_feet_t[:, :, [0, 2]]
            foot_loss = (foot_diff_xz[target_mask_t] ** 2).mean()

            foot_diff_y = feet[:, :, 1] - target_feet_t[:, :, 1]
            y_loss = (foot_diff_y[target_mask_t] ** 2).mean()
        else:
            foot_loss = ik_rot.new_tensor(0.0)
            y_loss = ik_rot.new_tensor(0.0)

        reg_loss = ((ik_rot - ik_rot_orig) ** 2).mean()

        if ik_rot.shape[0] > 2:
            vel = ik_rot[1:] - ik_rot[:-1]
            acc = vel[1:] - vel[:-1]
            smooth_loss = acc.pow(2).mean()
        else:
            smooth_loss = ik_rot.new_tensor(0.0)

        loss = (
            float(foot_weight) * foot_loss
            + float(y_weight) * y_loss
            + float(reg_weight) * reg_loss
            + float(smooth_weight) * smooth_loss
        )

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if best_loss is None or loss_value < best_loss:
            best_loss = loss_value
            best_ik_rot = ik_rot.detach().clone()

        if verbose and (step % 40 == 0 or step == int(steps) - 1):
            print(
                f"IK step {step:04d} | "
                f"loss={loss_value:.6f} "
                f"foot={float(foot_loss.detach().cpu()):.6f} "
                f"reg={float(reg_loss.detach().cpu()):.6f} "
                f"smooth={float(smooth_loss.detach().cpu()):.6f}"
            )

    if best_ik_rot is None:
        best_ik_rot = ik_rot.detach()

    out = motion.copy()
    rot6d_out = rot6d_base.detach().cpu().numpy()
    rot6d_out[:, IK_JOINTS, :] = best_ik_rot.detach().cpu().numpy()
    out[:, ROT_SLICE] = rot6d_out.reshape(motion.shape[0], -1)

    debug = {
        "ik_contact_ratio": float(contacts.mean()),
        "ik_segment_count": int(segment_count),
        "ik_executed": True,
        "ik_steps": int(steps),
        "ik_lr": float(lr),
        "ik_best_loss": float(best_loss),
        "ik_optimized_joints": IK_JOINTS,
    }

    return out.astype(np.float32), debug


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leg-IK foot lock postprocess for EDGE 151-D motion."
    )

    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--height_threshold", type=float, default=0.035)
    parser.add_argument("--min_contact_len", type=int, default=3)

    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.03)

    parser.add_argument("--foot_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.03)
    parser.add_argument("--smooth_weight", type=float, default=0.02)
    parser.add_argument("--y_weight", type=float, default=0.15)

    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    motion = load_motion(args.motion)

    out, debug = optimize_leg_ik(
        motion,
        device=args.device,
        height_threshold=args.height_threshold,
        min_contact_len=args.min_contact_len,
        steps=args.steps,
        lr=args.lr,
        foot_weight=args.foot_weight,
        reg_weight=args.reg_weight,
        smooth_weight=args.smooth_weight,
        y_weight=args.y_weight,
        verbose=not args.quiet,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, out.astype(np.float32))

    print(f"✅ saved: {args.out}")
    print("debug:", debug)


if __name__ == "__main__":
    main()