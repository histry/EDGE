import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F

try:
    import scipy.interpolate as spi
except Exception:
    spi = None

from EDGE import EDGE
from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract

try:
    from auto_keyframe_planner import (
        append_csv,
        plan_auto_keyframes,
        save_auto_keyframes,
    )
except Exception:
    append_csv = None
    plan_auto_keyframes = None
    save_auto_keyframes = None

try:
    from postprocess_footlock import foot_lock_root_correction, blend_back_to_trajectory
except Exception:
    foot_lock_root_correction = None
    blend_back_to_trajectory = None



ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_numpy(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_151_pose(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)

    if arr.ndim == 0 and isinstance(arr.item(), dict):
        data = arr.item()
        if "motion" in data:
            arr = data["motion"]
        elif "pose" in data:
            arr = data["pose"]
        else:
            raise ValueError(f"{path} 是 dict .npy，但没有 motion/pose 键。")

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 2:
        if arr.shape[1] != 151:
            raise ValueError(f"{path} 应该是 [T,151] 或 [151]，当前是 {arr.shape}")
        arr = arr[0]

    arr = arr.reshape(-1)

    if arr.shape[0] != 151:
        raise ValueError(f"{path} 必须是 151 维 pose，当前是 {arr.shape}")

    return arr.astype(np.float32)


def parse_list(text: str):
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def parse_mid_frames(text: str, num_mid: int, num_frames: int):
    if num_mid == 0:
        return []

    if not text:
        return [
            int(round((i + 1) * (num_frames - 1) / (num_mid + 1)))
            for i in range(num_mid)
        ]

    raw = parse_list(text)
    if len(raw) != num_mid:
        raise ValueError(f"--mid_pose_frames 数量应为 {num_mid}，当前为 {len(raw)}")

    frames = []
    for item in raw:
        value = float(item)
        if 0.0 < value < 1.0:
            value = value * (num_frames - 1)
        frame = int(round(value))
        frame = max(1, min(num_frames - 2, frame))
        frames.append(frame)

    return frames


def parse_trajectory_points(text: str) -> np.ndarray:
    points = []

    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue

        parts = item.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"轨迹控制点格式错误: {item}，应为 'x,z'，例如 '0,0;1,1;0,2'"
            )

        try:
            x = float(parts[0].strip())
            z = float(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"轨迹控制点不是数字: {item}") from exc

        points.append([x, z])

    if len(points) == 0:
        raise ValueError("--trajectory 至少需要一个控制点，例如 '0,0;1,1;0,2'")

    return np.asarray(points, dtype=np.float32)


def remove_duplicate_points(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32)

    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > eps:
            kept.append(point)

    return np.asarray(kept, dtype=np.float32)


def resample_feature(feature: np.ndarray, target_frames: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)

    if feature.ndim != 2:
        raise ValueError(f"feature 应该是 [T,C]，当前是 {feature.shape}")

    if feature.shape[0] == target_frames:
        return feature.astype(np.float32)

    tensor = torch.from_numpy(feature).float().unsqueeze(0).transpose(1, 2)
    tensor = F.interpolate(
        tensor,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return tensor.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def build_trajectory_progress(
    audio_feature: np.ndarray,
    use_audio_timing: bool = True,
    onset_index: int = 768,
    min_speed_bias: float = 0.20,
) -> np.ndarray:
    num_frames = audio_feature.shape[0]

    if (not use_audio_timing) or audio_feature.shape[1] <= onset_index:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    onset = audio_feature[:, onset_index].astype(np.float32)
    onset = np.maximum(onset, 0.0)

    if float(onset.max()) <= 1e-8:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    onset = onset / (float(onset.max()) + 1e-8)

    speed = onset + float(min_speed_bias)
    progress = np.cumsum(speed)
    progress = progress - progress[0]
    progress = progress / max(float(progress[-1]), 1e-8)

    return progress.astype(np.float32)


def interpolate_trajectory_smooth(
    points: np.ndarray,
    progress: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    points = remove_duplicate_points(points)

    if len(points) == 1:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)

    if float(u[-1]) <= 1e-8:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    u = u / u[-1]

    if smooth and spi is not None and len(points) >= 3:
        k = min(3, len(points) - 1)

        try:
            tck, _ = spi.splprep(
                [points[:, 0], points[:, 1]],
                u=u,
                s=0.0,
                k=k,
            )
            x_new, z_new = spi.splev(progress, tck)
            return np.stack([x_new, z_new], axis=-1).astype(np.float32)
        except Exception as exc:
            print(f"⚠️ Spline 轨迹插值失败，回退到线性插值: {exc}")

    x_new = np.interp(progress, u, points[:, 0])
    z_new = np.interp(progress, u, points[:, 1])

    return np.stack([x_new, z_new], axis=-1).astype(np.float32)


def normalize_trajectory_for_model(
    traj_physical: np.ndarray,
    normalizer,
    root_x_idx: int = ROOT_X_IDX,
    root_z_idx: int = ROOT_Z_IDX,
) -> np.ndarray:
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        raise ValueError("checkpoint 中没有有效 normalizer，无法归一化 trajectory")

    traj_physical = np.asarray(traj_physical, dtype=np.float32)

    mean_x = float(normalizer.mean[root_x_idx])
    mean_z = float(normalizer.mean[root_z_idx])
    std_x = float(normalizer.std[root_x_idx])
    std_z = float(normalizer.std[root_z_idx])

    traj_norm = traj_physical.copy()
    traj_norm[:, 0] = (traj_physical[:, 0] - mean_x) / (std_x + 1e-8)
    traj_norm[:, 1] = (traj_physical[:, 1] - mean_z) / (std_z + 1e-8)

    return traj_norm.astype(np.float32)


def build_control_trajectory(
    trajectory_text: str,
    audio_feature: np.ndarray,
    normalizer,
    target_traj_path: str = "",
    keep_absolute: bool = False,
    uniform_timing: bool = False,
    smooth: bool = True,
):
    num_frames = audio_feature.shape[0]

    if target_traj_path:
        traj_physical = np.load(target_traj_path).astype(np.float32)

        if traj_physical.ndim == 3:
            traj_physical = traj_physical[0]

        if traj_physical.ndim != 2 or traj_physical.shape[1] < 2:
            raise ValueError(
                f"--target_traj 应该是 [T,2] 或 [1,T,2]，当前是 {traj_physical.shape}"
            )

        traj_physical = traj_physical[:, :2]

        if len(traj_physical) != num_frames:
            traj_physical = resample_feature(traj_physical, num_frames)

    else:
        points = parse_trajectory_points(trajectory_text)

        progress = build_trajectory_progress(
            audio_feature=audio_feature,
            use_audio_timing=not uniform_timing,
        )

        traj_physical = interpolate_trajectory_smooth(
            points=points,
            progress=progress,
            smooth=smooth,
        )

    if not keep_absolute:
        traj_physical = traj_physical - traj_physical[0:1]

    traj_norm = normalize_trajectory_for_model(
        traj_physical=traj_physical,
        normalizer=normalizer,
    )

    return traj_physical.astype(np.float32), traj_norm.astype(np.float32)


def normalize_pose_if_needed(pose: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    if pose_space == "normalized":
        return pose.astype(np.float32)

    if pose_space == "physical":
        if normalizer is None:
            raise ValueError("--pose_space physical 需要 checkpoint 里有 normalizer")

        pose_t = torch.from_numpy(pose[None, None, :]).float()
        pose_norm = normalizer.normalize(pose_t)
        pose_norm = to_numpy(pose_norm)[0, 0]
        return pose_norm.astype(np.float32)

    raise ValueError(f"Unknown pose_space: {pose_space}")


def make_keyframe_feature_mask(constrain_contacts: bool) -> np.ndarray:
    mask_one_frame = np.zeros((151,), dtype=np.float32)

    if constrain_contacts:
        mask_one_frame[CONTACT_SLICE] = 1.0

    # root X/Z 由 trajectory 控制，不在关键帧里锁死。
    mask_one_frame[ROOT_Y_IDX] = 1.0
    mask_one_frame[ROT_SLICE] = 1.0

    return mask_one_frame


def build_constraint(args, normalizer, num_frames: int, device) -> dict:
    value = np.zeros((num_frames, 151), dtype=np.float32)
    mask = np.zeros((num_frames, 151), dtype=np.float32)

    frame_mask = make_keyframe_feature_mask(
        constrain_contacts=args.constrain_contacts,
    )

    def add_pose(path: str, frame: int, name: str):
        pose = load_151_pose(path)
        pose = normalize_pose_if_needed(pose, normalizer, args.pose_space)

        width = max(0, int(getattr(args, "infer_keyframe_width", 0)))
        start = max(0, frame - width)
        end = min(num_frames, frame + width + 1)

        for f in range(start, end):
            value[f] = pose
            mask[f] = frame_mask

        if width > 0:
            print(
                f"✅ 已添加 {name} 关键帧窗口: "
                f"center={frame}, range=[{start}, {end - 1}], path={path}"
            )
        else:
            print(f"✅ 已添加 {name} 关键帧: frame={frame}, path={path}")

    add_pose(args.start_pose, 0, "start")
    add_pose(args.end_pose, num_frames - 1, "end")

    mid_paths = parse_list(args.mid_poses)
    mid_frames = parse_mid_frames(args.mid_pose_frames, len(mid_paths), num_frames)

    for i, (path, frame) in enumerate(zip(mid_paths, mid_frames), start=1):
        add_pose(path, frame, f"mid{i}")

    return {
        "value": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),
    }


def sample_motion(model: EDGE, cond: dict, constraint: dict, args, num_frames: int):
    shape = (1, num_frames, model.repr_dim)

    if args.sampler == "ddim":
        print("🚀 使用 DDIM 采样，速度更快；轨迹主要依赖条件分支和 hard projection。")
        return model.diffusion.ddim_sample(
            shape,
            cond,
            constraint=constraint,
        )

    print("🚀 使用 DDPM 采样；若开启 TTO，可获得更强轨迹/关键帧优化，但速度更慢。")
    return model.diffusion.p_sample_loop(
        shape,
        cond,
        constraint=constraint,
        use_tto=not args.no_tto,
    )


def apply_trajectory_anchor(
    motion_physical: np.ndarray,
    traj_physical: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Softly anchor generated root X/Z to the target trajectory.

    strength:
      0.0 = keep generated trajectory
      1.0 = strictly replace root X/Z with target trajectory
      0~1 = soft blend, useful for display videos
    """
    motion = motion_physical.copy()
    traj = np.asarray(traj_physical, dtype=np.float32)

    if traj.ndim == 3:
        traj = traj[0]

    if traj.ndim != 2 or traj.shape[1] < 2:
        raise ValueError(f"traj_physical 应该是 [T,2] 或 [1,T,2]，当前是 {traj.shape}")

    traj = traj[:, :2]

    if len(traj) != len(motion):
        traj = resample_feature(traj, len(motion))

    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 0.0:
        return motion.astype(np.float32)

    motion[:, ROOT_X_IDX] = (
        (1.0 - strength) * motion[:, ROOT_X_IDX] + strength * traj[:, 0]
    )
    motion[:, ROOT_Z_IDX] = (
        (1.0 - strength) * motion[:, ROOT_Z_IDX] + strength * traj[:, 1]
    )

    return motion.astype(np.float32)


def save_eval_assets(
    out_path: str,
    motion_raw_physical: np.ndarray,
    motion_final_physical: np.ndarray,
    traj_physical: np.ndarray,
    args,
    trajectory_control_mode: str = "unknown",
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stem = out_path.stem
    out_dir = out_path.parent

    raw_motion_path = out_dir / f"{stem}_raw.npy"
    final_motion_path = out_path
    target_traj_path = out_dir / f"{stem}_target_traj.npy"
    eval_json_path = out_dir / f"{stem}_metrics.json"
    eval_csv_path = out_dir / f"{stem}_metrics.csv"
    meta_path = out_dir / f"{stem}_meta.json"

    np.save(raw_motion_path, motion_raw_physical.astype(np.float32))
    np.save(final_motion_path, motion_final_physical.astype(np.float32))
    np.save(target_traj_path, traj_physical.astype(np.float32))

    eval_cmd = [
        "python",
        "eval_quantitative.py",
        "--motion",
        str(final_motion_path),
        "--raw_motion",
        str(raw_motion_path),
        "--post_motion",
        str(final_motion_path),
        "--checkpoint",
        args.checkpoint,
        "--audio",
        args.music,
        "--target_traj",
        str(target_traj_path),
        "--start_pose",
        args.start_pose,
        "--end_pose",
        args.end_pose,
        "--out_json",
        str(eval_json_path),
        "--out_csv",
        str(eval_csv_path),
    ]

    if getattr(args, "mid_poses", ""):
        eval_cmd.extend(["--mid_poses", args.mid_poses])

    if getattr(args, "mid_pose_frames", ""):
        eval_cmd.extend(["--mid_pose_frames", args.mid_pose_frames])

    if getattr(args, "pose_space", "normalized") == "physical":
        eval_cmd.extend(["--keyframe_space", "physical"])
    else:
        eval_cmd.extend(["--keyframe_space", "normalized"])

    meta = {
        "checkpoint": args.checkpoint,
        "music": args.music,
        "start_pose": args.start_pose,
        "end_pose": args.end_pose,
        "mid_poses": getattr(args, "mid_poses", ""),
        "mid_pose_frames": getattr(args, "mid_pose_frames", ""),
        "auto_mid_keyframes": bool(getattr(args, "auto_mid_keyframes", False)),
        "auto_mid_count": int(getattr(args, "auto_mid_count", 0)),
        "rag_db": getattr(args, "rag_db", ""),
        "auto_mid_plan": getattr(args, "auto_mid_plan_path", ""),
        "trajectory": getattr(args, "trajectory", ""),
        "target_traj": getattr(args, "target_traj", ""),
        "motion_raw": str(raw_motion_path),
        "motion_final": str(final_motion_path),
        "target_traj_saved": str(target_traj_path),
        "eval_json": str(eval_json_path),
        "eval_csv": str(eval_csv_path),
        "post_anchor_trajectory": bool(getattr(args, "post_anchor_trajectory", False)),
        "trajectory_anchor_strength": float(getattr(args, "trajectory_anchor_strength", 0.0)),
        "post_foot_lock": bool(getattr(args, "post_foot_lock", False)),
        "foot_lock_strength": float(getattr(args, "foot_lock_strength", 0.75)),
        "foot_lock_traj_keep": float(getattr(args, "foot_lock_traj_keep", 0.65)),
        "foot_lock_height_threshold": float(getattr(args, "foot_lock_height_threshold", 0.035)),
        "foot_lock_speed_threshold": float(getattr(args, "foot_lock_speed_threshold", 0.08)),
        "foot_lock_min_contact_len": int(getattr(args, "foot_lock_min_contact_len", 3)),
        "foot_lock_smooth_window": int(getattr(args, "foot_lock_smooth_window", 9)),
        "sampler": getattr(args, "sampler", "ddpm"),
        "use_tto": not getattr(args, "no_tto", False),
        "tto_steps": int(getattr(args, "tto_steps", 1)),
        "tto_interval": int(getattr(args, "tto_interval", 50)),
        "tto_lr": float(getattr(args, "tto_lr", 0.03)),
        "tto_contact_threshold": float(getattr(args, "tto_contact_threshold", 0.65)),
        "tto_trajectory_loss_weight": float(getattr(args, "tto_trajectory_loss_weight", 4.0)),
        "tto_trajectory_velocity_loss_weight": float(getattr(args, "tto_trajectory_velocity_loss_weight", 0.5)),
        "tto_root_acc_loss_weight": float(getattr(args, "tto_root_acc_loss_weight", 0.05)),
        "tto_foot_loss_weight": float(getattr(args, "tto_foot_loss_weight", 0.25)),
        "eval_command": " ".join(eval_cmd),
        "trajectory_control_mode": trajectory_control_mode,
        "report_warning": (
            "If trajectory_control_mode starts with postprocess, report raw and final "
            "metrics separately. Final trajectory error may include post-processing."
        ),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ raw motion: {raw_motion_path}")
    print(f"✅ final motion: {final_motion_path}")
    print(f"✅ target trajectory: {target_traj_path}")
    print(f"✅ meta: {meta_path}")
    print("\n📏 可直接运行以下评估命令：")
    print(" ".join(eval_cmd))

    return {
        "raw_motion": str(raw_motion_path),
        "final_motion": str(final_motion_path),
        "target_traj": str(target_traj_path),
        "meta": str(meta_path),
        "eval_json": str(eval_json_path),
        "eval_csv": str(eval_csv_path),
        "eval_cmd": eval_cmd,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Controlled EDGE generation: music + start/end keyframes + 2D trajectory."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--start_pose", required=True)
    parser.add_argument("--end_pose", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument(
        "--mid_poses",
        default="",
        help="可选，中间关键帧路径，逗号或分号分隔。",
    )
    parser.add_argument(
        "--mid_pose_frames",
        default="",
        help="可选，中间关键帧位置，例如 '0.33,0.66' 或 '50,100'。",
    )

    parser.add_argument(
        "--infer_keyframe_width",
        type=int,
        default=0,
        help=(
            "推理阶段关键帧约束窗口半径。0=只锁单帧；"
            "1=锁 frame-1~frame+1。自动中间关键帧建议 0~1，"
            "手工 4key 如果最后突变可试 2~3。"
        ),
    )
    parser.add_argument(
        "--auto_mid_keyframes",
        action="store_true",
        help="启用系统自动中间关键帧规划：根据音乐 onset、轨迹转折和 RAG 姿态库自动插入中间姿态。",
    )
    parser.add_argument(
        "--rag_db",
        default="data/dunhuang_rag_db/rag_index.npz",
        help="RAG 姿态库路径。可为 build_dunhuang_rag_db.py 生成的 rag_index.npz，也可为包含 .npy/.pkl 的目录。",
    )
    parser.add_argument(
        "--auto_mid_count",
        type=int,
        default=3,
        help="系统自动插入的中间关键帧数量。",
    )
    parser.add_argument(
        "--auto_mid_min_gap",
        type=int,
        default=18,
        help="自动关键帧与用户关键帧/其他自动关键帧的最小帧距。",
    )
    parser.add_argument(
        "--auto_mid_pose_space",
        default="normalized",
        choices=["normalized", "physical"],
        help="RAG 数据库里 151-D motion/pose 的空间。build_dunhuang_rag_db.py 默认输出 normalized。",
    )
    parser.add_argument(
        "--auto_mid_max_candidates",
        type=int,
        default=5000,
        help="RAG 检索时最多加载多少候选姿态。",
    )
    parser.add_argument(
        "--auto_mid_sample_stride",
        type=int,
        default=3,
        help="RAG 检索时对候选姿态的采样间隔，越大越快但候选更少。",
    )
    parser.add_argument(
        "--auto_mid_music_weight",
        type=float,
        default=0.6,
        help="自动选帧时音乐 onset 权重。",
    )
    parser.add_argument(
        "--auto_mid_trajectory_weight",
        type=float,
        default=0.4,
        help="自动选帧时轨迹转折权重。",
    )
    parser.add_argument(
        "--save_auto_keyframes",
        action="store_true",
        help="保存自动规划出的中间关键帧 .npy 和 plan.json，便于复现实验与评估。",
    )

    parser.add_argument(
        "--trajectory",
        default="0,0",
        help="轨迹控制点，例如 '0,0;1,1;0,2;-1,3;0,4'。",
    )
    parser.add_argument(
        "--target_traj",
        default="",
        help="可选，直接读取 [T,2] 或 [1,T,2] 的 .npy 轨迹，优先级高于 --trajectory。",
    )

    parser.add_argument("--feature_type", default="hybrid", choices=["hybrid"])
    parser.add_argument("--audio_dim", type=int, default=803)
    parser.add_argument("--seq_len", type=int, default=150)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--num_frames",
        type=int,
        default=150,
        help="默认 150 帧，即 5 秒。若要按整首音乐长度生成，使用 --use_full_music。",
    )
    parser.add_argument(
        "--use_full_music",
        action="store_true",
        help="根据音乐实际时长设置 num_frames。",
    )

    parser.add_argument(
        "--pose_space",
        default="normalized",
        choices=["normalized", "physical"],
        help="关键帧 .npy 的空间。图像转3D结果一般用 physical；训练切片中导出的可用 normalized。",
    )
    parser.add_argument(
        "--constrain_contacts",
        action="store_true",
        help="是否锁定 0:4 的脚接触通道。2D 图像转姿态时通常不要开。",
    )
    parser.add_argument(
        "--keep_trajectory_absolute",
        action="store_true",
        help="默认会把轨迹平移到从第一个点开始的相对轨迹；开启后保留绝对坐标。",
    )
    parser.add_argument(
        "--uniform_trajectory_timing",
        action="store_true",
        help="默认使用音乐 onset 调整轨迹速度；开启后匀速走轨迹。",
    )
    parser.add_argument(
        "--linear_trajectory",
        action="store_true",
        help="关闭 spline 平滑插值，改用线性折线轨迹。",
    )

    parser.add_argument(
        "--sampler",
        choices=["ddim", "ddpm"],
        default="ddpm",
        help="ddpm 较慢但支持 TTO；ddim 较快。",
    )
    parser.add_argument(
        "--no_tto",
        action="store_true",
        help="关闭 test-time optimization。",
    )
    parser.add_argument(
        "--beat_guidance_weight",
        type=float,
        default=0.0,
        help="推理阶段 beat/onset 弱引导权重。没有真实配对数据时建议小心使用。",
    )
    parser.add_argument(
        "--no_hard_keyframe_project",
        action="store_true",
        help="关闭扩散过程中的硬关键帧投影。默认开启。",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="默认使用 active 权重；开启后尝试使用 EMA 权重。",
    )

    parser.add_argument(
        "--post_anchor_trajectory",
        action="store_true",
        help="生成后强制把 root X/Z 替换成目标轨迹。轨迹误差最低，但可能增加脚滑。",
    )
    parser.add_argument(
        "--trajectory_anchor_strength",
        type=float,
        default=0.0,
        help=(
            "生成后 root X/Z 贴合目标轨迹的软锚定强度。"
            "0 表示不后处理，1 表示严格替换为目标轨迹。"
            "建议展示视频用 0.6~0.9。"
        ),
    )
    parser.add_argument(
        "--post_foot_lock",
        action="store_true",
        help="生成后执行 contact-aware root-only foot lock。注意：目前更推荐单独使用 postprocess_leg_ik.py。",
    )
    parser.add_argument(
        "--foot_lock_strength",
        type=float,
        default=0.75,
        help="foot lock root 修正强度，建议 0.5~0.85。",
    )
    parser.add_argument(
        "--foot_lock_traj_keep",
        type=float,
        default=0.65,
        help="foot lock 后保留目标轨迹的比例，越大轨迹越准，越小脚滑越少。",
    )
    parser.add_argument(
        "--foot_lock_height_threshold",
        type=float,
        default=0.035,
        help="高度接触阈值，单位米。",
    )
    parser.add_argument(
        "--foot_lock_speed_threshold",
        type=float,
        default=0.08,
        help="脚底水平速度接触阈值，单位 m/s。",
    )
    parser.add_argument(
        "--foot_lock_min_contact_len",
        type=int,
        default=3,
        help="连续多少帧以上才认为是一段有效脚接触。",
    )
    parser.add_argument(
        "--foot_lock_smooth_window",
        type=int,
        default=9,
        help="root correction 平滑窗口。",
    )

    parser.add_argument(
        "--tto_steps",
        type=int,
        default=1,
        help="每次 TTO 的梯度优化步数；增大可增强关键帧/轨迹约束，但会变慢。",
    )
    parser.add_argument(
        "--tto_interval",
        type=int,
        default=50,
        help="每隔多少 diffusion step 执行一次 TTO；越小约束越强但越慢。",
    )
    parser.add_argument(
        "--tto_lr",
        type=float,
        default=0.03,
        help="TTO 梯度步长。",
    )
    parser.add_argument(
        "--tto_contact_threshold",
        type=float,
        default=0.65,
        help="TTO 中判断脚接触通道为接触的阈值。",
    )
    parser.add_argument(
        "--tto_trajectory_loss_weight",
        type=float,
        default=4.0,
        help="TTO 中轨迹位置误差权重；只影响推理阶段优化，不影响训练 loss。",
    )
    parser.add_argument(
        "--tto_trajectory_velocity_loss_weight",
        type=float,
        default=0.5,
        help="TTO 中轨迹速度误差权重；用于约束沿轨迹的速度变化。",
    )
    parser.add_argument(
        "--tto_root_acc_loss_weight",
        type=float,
        default=0.05,
        help="TTO 中 root X/Z 加速度平滑权重；过大可能削弱急转弯轨迹。",
    )
    parser.add_argument(
        "--tto_foot_loss_weight",
        type=float,
        default=0.25,
        help="TTO 中脚滑惩罚权重；过大可能牺牲轨迹贴合。",
    )

    parser.add_argument(
        "--save_controls",
        action="store_true",
        help="额外保存 target trajectory、raw motion 和 meta json，方便评估。",
    )
    parser.add_argument(
        "--save_eval_assets",
        action="store_true",
        help="保存 raw motion、target trajectory、meta，并打印 eval_quantitative.py 命令。",
    )

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    print("🚀 初始化 EDGE controlled generator")
    model = EDGE(
        feature_type=args.feature_type,
        checkpoint_path=args.checkpoint,
        audio_dim=args.audio_dim,
        seq_len=args.seq_len,
        EMA=args.use_ema,
        beat_guidance_weight=args.beat_guidance_weight,
        hard_keyframe_project=not args.no_hard_keyframe_project,
    )
    model.eval()

    model.diffusion.tto_steps = int(args.tto_steps)
    model.diffusion.tto_interval = int(args.tto_interval)
    model.diffusion.tto_lr = float(args.tto_lr)
    model.diffusion.tto_contact_threshold = float(args.tto_contact_threshold)
    model.diffusion.tto_trajectory_loss_weight = float(args.tto_trajectory_loss_weight)
    model.diffusion.tto_trajectory_velocity_loss_weight = float(args.tto_trajectory_velocity_loss_weight)
    model.diffusion.tto_root_acc_loss_weight = float(args.tto_root_acc_loss_weight)
    model.diffusion.tto_foot_loss_weight = float(args.tto_foot_loss_weight)

    print(
        "🧪 TTO config: "
        f"steps={model.diffusion.tto_steps}, "
        f"interval={model.diffusion.tto_interval}, "
        f"lr={model.diffusion.tto_lr}, "
        f"contact_threshold={model.diffusion.tto_contact_threshold}, "
        f"traj_w={model.diffusion.tto_trajectory_loss_weight}, "
        f"traj_vel_w={model.diffusion.tto_trajectory_velocity_loss_weight}, "
        f"root_acc_w={model.diffusion.tto_root_acc_loss_weight}, "
        f"foot_w={model.diffusion.tto_foot_loss_weight}"
    )

    device = model.accelerator.device
    normalizer = model.normalizer

    if normalizer is None:
        raise ValueError("checkpoint 中没有 normalizer，无法保证轨迹/姿态空间一致。")

    print("🎵 提取 hybrid audio feature")
    extracted = hybrid_extract(args.music)

    if isinstance(extracted, tuple):
        audio_feature, audio_feature_path = extracted
    else:
        audio_feature = extracted
        audio_feature_path = ""

    audio_feature = np.asarray(audio_feature, dtype=np.float32)

    if audio_feature.ndim != 2:
        raise ValueError(f"audio_feature 应该是 [T,C]，当前是 {audio_feature.shape}")

    if audio_feature.shape[1] != args.audio_dim:
        raise ValueError(
            f"audio feature dim 与模型不一致: feature={audio_feature.shape[1]}, "
            f"model/audio_dim={args.audio_dim}"
        )

    if args.use_full_music:
        duration = librosa.get_duration(path=args.music)
        num_frames = int(round(duration * args.fps))
    else:
        num_frames = int(args.num_frames)

    if num_frames <= 1:
        raise ValueError(f"num_frames 必须大于 1，当前为 {num_frames}")

    audio_feature = resample_feature(audio_feature, num_frames)

    print(f"📊 audio_feature={audio_feature.shape}, num_frames={num_frames}")

    traj_physical, traj_norm = build_control_trajectory(
        trajectory_text=args.trajectory,
        audio_feature=audio_feature,
        normalizer=model.normalizer,
        target_traj_path=args.target_traj,
        keep_absolute=args.keep_trajectory_absolute,
        uniform_timing=args.uniform_trajectory_timing,
        smooth=not args.linear_trajectory,
    )

    # ------------------------------------------------------------------
    # 自动中间关键帧规划：用户只给 start/end 或少量 mid，系统从 RAG 库检索补全。
    # 注意：这里会把自动关键帧保存为 .npy，并追加到 args.mid_poses/args.mid_pose_frames，
    # 因而后续 build_constraint 与 eval_quantitative.py 无需额外改动。
    # ------------------------------------------------------------------
    args.auto_mid_plan_path = ""
    if bool(getattr(args, "auto_mid_keyframes", False)):
        if plan_auto_keyframes is None or save_auto_keyframes is None or append_csv is None:
            raise ImportError(
                "auto_mid_keyframes requires auto_keyframe_planner.py in project root."
            )

        print("🧠 启用自动中间关键帧规划：music/trajectory/RAG retrieval。")

        start_pose_norm = normalize_pose_if_needed(
            load_151_pose(args.start_pose),
            normalizer,
            args.pose_space,
        )
        end_pose_norm = normalize_pose_if_needed(
            load_151_pose(args.end_pose),
            normalizer,
            args.pose_space,
        )

        user_mid_paths = parse_list(args.mid_poses)
        user_mid_frames = parse_mid_frames(args.mid_pose_frames, len(user_mid_paths), num_frames)
        user_mid_poses = [
            normalize_pose_if_needed(load_151_pose(path), normalizer, args.pose_space)
            for path in user_mid_paths
        ]

        # 如果用户没有显式写 mid_pose_frames，但提供了 mid_poses，
        # 先把自动推断出的 user mid frames 固化，避免追加 auto frames 后数量不一致。
        if user_mid_paths:
            args.mid_pose_frames = ",".join(str(int(f)) for f in user_mid_frames)

        auto_plan = plan_auto_keyframes(
            start_pose=start_pose_norm,
            end_pose=end_pose_norm,
            user_mid_poses=user_mid_poses,
            user_mid_frames=user_mid_frames,
            audio_feature=audio_feature,
            traj_physical=traj_physical,
            rag_db=args.rag_db,
            normalizer=normalizer,
            num_frames=num_frames,
            max_auto_keyframes=args.auto_mid_count,
            min_gap=args.auto_mid_min_gap,
            rag_pose_space=args.auto_mid_pose_space,
            max_candidates=args.auto_mid_max_candidates,
            sample_stride=args.auto_mid_sample_stride,
            music_weight=args.auto_mid_music_weight,
            trajectory_weight=args.auto_mid_trajectory_weight,
        )

        auto_paths, auto_frames, auto_plan_path = save_auto_keyframes(
            auto_plan,
            out_motion_path=args.out,
            prefix="auto_mid",
        )
        args.auto_mid_plan_path = auto_plan_path

        args.mid_poses = append_csv(args.mid_poses, auto_paths)
        args.mid_pose_frames = append_csv(args.mid_pose_frames, auto_frames)

        print("✅ 自动中间关键帧规划完成：")
        for path, frame in zip(auto_paths, auto_frames):
            print(f"  - frame={frame}, pose={path}")
        print(f"✅ auto plan: {auto_plan_path}")

    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(
            device=device,
            dtype=torch.float32,
        ),
        "trajectory": torch.from_numpy(traj_norm[None]).to(
            device=device,
            dtype=torch.float32,
        ),
    }

    # 显式提供 onset，避免 diffusion.py 里只能从 audio[...,768] fallback。
    if audio_feature.shape[1] > 768:
        onset = audio_feature[:, 768:769].astype(np.float32)
        cond["onset"] = torch.from_numpy(onset[None]).to(
            device=device,
            dtype=torch.float32,
        )

    constraint = build_constraint(args, normalizer, num_frames, device)

    with torch.no_grad():
        sample_norm = sample_motion(model, cond, constraint, args, num_frames)

    motion_norm = sample_norm.detach().cpu()
    motion_physical_raw = model.normalizer.unnormalize(motion_norm)
    motion_physical_raw = to_numpy(motion_physical_raw)[0].astype(np.float32)
    motion_physical_final = motion_physical_raw.copy()

    trajectory_control_mode = "model_condition_tto_only"

    if args.post_anchor_trajectory:
        trajectory_control_mode = "postprocess_hard_anchor"
        print(
            "⚠️ post_anchor_trajectory 已开启：root X/Z 将被严格替换为目标轨迹。"
            "该结果可以展示系统级轨迹控制，但不能单独声称是纯模型输出。"
        )
        motion_physical_final = apply_trajectory_anchor(
            motion_physical_final,
            traj_physical,
            strength=1.0,
        )

    elif float(args.trajectory_anchor_strength) > 0.0:
        trajectory_control_mode = "postprocess_soft_anchor"
        print(
            f"⚠️ soft trajectory anchor 已开启：strength={args.trajectory_anchor_strength:.3f}。"
            "请同时报告 raw motion 和 final motion 指标。"
        )
        motion_physical_final = apply_trajectory_anchor(
            motion_physical_final,
            traj_physical,
            strength=args.trajectory_anchor_strength,
        )

    else:
        print("✅ 未启用 post trajectory anchor：final motion 等于 raw model/TTO output。")

    if args.post_foot_lock:
        if foot_lock_root_correction is None or blend_back_to_trajectory is None:
            raise ImportError(
                "post_foot_lock requires postprocess_footlock.py in project root."
            )

        print("🦶 post_foot_lock 已开启：执行 contact-aware root correction。")

        motion_physical_final, footlock_debug = foot_lock_root_correction(
            motion_physical_final,
            device=str(device),
            fps=args.fps,
            height_threshold=args.foot_lock_height_threshold,
            speed_threshold=args.foot_lock_speed_threshold,
            min_contact_len=args.foot_lock_min_contact_len,
            lock_strength=args.foot_lock_strength,
            smooth_window=args.foot_lock_smooth_window,
        )

        if args.post_anchor_trajectory or float(args.trajectory_anchor_strength) > 0.0:
            motion_physical_final = blend_back_to_trajectory(
                motion_physical_final,
                target_traj=traj_physical,
                traj_keep=args.foot_lock_traj_keep,
                keep_endpoints=True,
            )

        if trajectory_control_mode.startswith("postprocess"):
            trajectory_control_mode = trajectory_control_mode + "+foot_lock"
        else:
            trajectory_control_mode = "postprocess_foot_lock"

        print(f"🦶 foot lock debug: {footlock_debug}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.save_eval_assets:
        save_eval_assets(
            out_path=args.out,
            motion_raw_physical=motion_physical_raw,
            motion_final_physical=motion_physical_final,
            traj_physical=traj_physical,
            args=args,
            trajectory_control_mode=trajectory_control_mode,
        )
    else:
        np.save(args.out, motion_physical_final.astype(np.float32))
        print(f"✅ saved motion: {args.out}, shape={motion_physical_final.shape}")

    if args.save_controls:
        out_path = Path(args.out)
        traj_path = out_path.with_name(out_path.stem + "_target_traj.npy")
        raw_path = out_path.with_name(out_path.stem + "_raw.npy")
        meta_path = out_path.with_name(out_path.stem + "_meta.json")

        np.save(traj_path, traj_physical.astype(np.float32))
        np.save(raw_path, motion_physical_raw.astype(np.float32))

        meta = {
            "checkpoint": args.checkpoint,
            "music": args.music,
            "audio_feature_path": audio_feature_path,
            "start_pose": args.start_pose,
            "end_pose": args.end_pose,
            "mid_poses": parse_list(args.mid_poses),
            "mid_pose_frames": parse_mid_frames(
                args.mid_pose_frames,
                len(parse_list(args.mid_poses)),
                num_frames,
            ),
            "auto_mid_keyframes": args.auto_mid_keyframes,
            "auto_mid_count": args.auto_mid_count,
            "rag_db": args.rag_db,
            "auto_mid_plan": getattr(args, "auto_mid_plan_path", ""),
            "auto_mid_min_gap": args.auto_mid_min_gap,
            "auto_mid_pose_space": args.auto_mid_pose_space,
            "auto_mid_max_candidates": args.auto_mid_max_candidates,
            "auto_mid_sample_stride": args.auto_mid_sample_stride,
            "auto_mid_music_weight": args.auto_mid_music_weight,
            "auto_mid_trajectory_weight": args.auto_mid_trajectory_weight,
            "trajectory": args.trajectory,
            "target_traj": args.target_traj,
            "target_traj_saved": str(traj_path),
            "raw_motion_saved": str(raw_path),
            "out": args.out,
            "num_frames": num_frames,
            "fps": args.fps,
            "pose_space": args.pose_space,
            "sampler": args.sampler,
            "use_tto": not args.no_tto,
            "tto_steps": args.tto_steps,
            "tto_interval": args.tto_interval,
            "tto_lr": args.tto_lr,
            "tto_contact_threshold": args.tto_contact_threshold,
            "tto_trajectory_loss_weight": args.tto_trajectory_loss_weight,
            "tto_trajectory_velocity_loss_weight": args.tto_trajectory_velocity_loss_weight,
            "tto_root_acc_loss_weight": args.tto_root_acc_loss_weight,
            "tto_foot_loss_weight": args.tto_foot_loss_weight,
            "hard_keyframe_project": not args.no_hard_keyframe_project,
            "post_anchor_trajectory": args.post_anchor_trajectory,
            "trajectory_anchor_strength": args.trajectory_anchor_strength,
            "post_foot_lock": args.post_foot_lock,
            "foot_lock_strength": args.foot_lock_strength,
            "foot_lock_traj_keep": args.foot_lock_traj_keep,
            "foot_lock_height_threshold": args.foot_lock_height_threshold,
            "foot_lock_speed_threshold": args.foot_lock_speed_threshold,
            "foot_lock_min_contact_len": args.foot_lock_min_contact_len,
            "foot_lock_smooth_window": args.foot_lock_smooth_window,
            "beat_guidance_weight": args.beat_guidance_weight,
            "keep_trajectory_absolute": args.keep_trajectory_absolute,
            "uniform_trajectory_timing": args.uniform_trajectory_timing,
            "linear_trajectory": args.linear_trajectory,
            "trajectory_control_mode": trajectory_control_mode,
            "report_warning": (
                "If trajectory_control_mode starts with postprocess, report raw and final "
                "metrics separately. Final trajectory error may include post-processing."
            ),
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ target trajectory: {traj_path}")
        print(f"✅ raw motion: {raw_path}")
        print(f"✅ meta: {meta_path}")


if __name__ == "__main__":
    main()