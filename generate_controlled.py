import argparse
import json
import os
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


ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
CONTACT_SLICE = slice(0, 4)
ROT_SLICE = slice(7, 151)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


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
def remove_duplicate_points(points: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if len(points) <= 1:
        return points

    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - kept[-1]) > eps:
            kept.append(p)

    return np.asarray(kept, dtype=np.float32)


def interpolate_trajectory_smooth(
    points: np.ndarray,
    progress: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    points = remove_duplicate_points(points)

    if len(points) == 1:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    # 按控制点之间的距离分配参数，避免控制点距离不均导致局部速度异常
    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32)

    if float(u[-1]) <= 1e-8:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    u = u / u[-1]

    if smooth and spi is not None and len(points) >= 3:
        # k 最多为 3；控制点少时自动降低阶数
        k = min(3, len(points) - 1)

        try:
            tck, _ = spi.splprep(
                [points[:, 0], points[:, 1]],
                u=u,
                s=0.0,
                k=k,
            )
            x_new, z_new = spi.splev(progress, tck)
            traj = np.stack([x_new, z_new], axis=-1).astype(np.float32)
            return traj
        except Exception as exc:
            print(f"⚠️ Spline 轨迹插值失败，回退到线性插值: {exc}")

    # fallback: 线性插值
    x_new = np.interp(progress, u, points[:, 0])
    z_new = np.interp(progress, u, points[:, 1])
    return np.stack([x_new, z_new], axis=-1).astype(np.float32)
def normalize_trajectory_for_model(
    traj_physical: np.ndarray,
    normalizer,
    root_x_idx: int = 4,
    root_z_idx: int = 6,
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

        traj_physical = traj_physical[:, :2]

        if len(traj_physical) != num_frames:
            traj_t = torch.from_numpy(traj_physical).float().unsqueeze(0).transpose(1, 2)
            traj_t = F.interpolate(
                traj_t,
                size=num_frames,
                mode="linear",
                align_corners=False,
            )
            traj_physical = traj_t.transpose(1, 2).squeeze(0).numpy().astype(np.float32)

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

    # 给一个基础速度，避免无 onset 的区间完全停住
    speed = onset + float(min_speed_bias)

    progress = np.cumsum(speed)
    progress = progress - progress[0]
    progress = progress / max(float(progress[-1]), 1e-8)

    return progress.astype(np.float32)

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

        x = float(parts[0].strip())
        z = float(parts[1].strip())
        points.append([x, z])

    if len(points) == 0:
        raise ValueError("--trajectory 至少需要一个控制点")

    return np.asarray(points, dtype=np.float32)

def normalize_pose_if_needed(pose: np.ndarray, normalizer, pose_space: str) -> np.ndarray:
    if pose_space == "normalized":
        return pose.astype(np.float32)

    if pose_space == "physical":
        if normalizer is None:
            raise ValueError("--pose_space physical 需要 checkpoint 里有 normalizer")
        return normalizer.normalize(torch.from_numpy(pose[None, None, :])).numpy()[0, 0].astype(np.float32)

    raise ValueError(f"Unknown pose_space: {pose_space}")


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


def resample_feature(feature: np.ndarray, target_frames: int) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)

    if feature.ndim != 2:
        raise ValueError(f"audio feature 应该是 [T,C]，当前是 {feature.shape}")

    if feature.shape[0] == target_frames:
        return feature

    tensor = torch.from_numpy(feature).float().unsqueeze(0).transpose(1, 2)
    tensor = F.interpolate(
        tensor,
        size=target_frames,
        mode="linear",
        align_corners=False,
    )
    return tensor.transpose(1, 2).squeeze(0).numpy().astype(np.float32)


def parse_trajectory_points(text: str) -> np.ndarray:
    points = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        x_text, z_text = item.split(",")
        points.append([float(x_text.strip()), float(z_text.strip())])

    if not points:
        raise ValueError("--trajectory 至少需要一个控制点，例如 '0,0;1,1;0,2'")

    return np.asarray(points, dtype=np.float32)


def build_progress_from_audio(audio_feature: np.ndarray, use_audio_timing: bool) -> np.ndarray:
    num_frames = audio_feature.shape[0]

    if not use_audio_timing or audio_feature.shape[1] <= 768:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    onset = audio_feature[:, 768].astype(np.float32)
    onset = np.maximum(onset, 0.0)

    if float(onset.max()) <= 1e-8:
        return np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    onset = onset / (float(onset.max()) + 1e-8)
    speed = onset + 0.20
    progress = np.cumsum(speed)
    progress = (progress - progress[0]) / max(float(progress[-1] - progress[0]), 1e-8)

    return progress.astype(np.float32)


def interpolate_trajectory(points: np.ndarray, progress: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    seg_len = np.linalg.norm(points[1:] - points[:-1], axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg_len)])

    if float(u[-1]) <= 1e-8:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    u = u / u[-1]

    # 去掉重复控制点，避免插值异常
    keep = np.concatenate([[True], np.diff(u) > 1e-8])
    u = u[keep]
    points = points[keep]

    if len(points) == 1:
        return np.repeat(points[None, 0, :], len(progress), axis=0).astype(np.float32)

    x = np.interp(progress, u, points[:, 0])
    z = np.interp(progress, u, points[:, 1])

    return np.stack([x, z], axis=-1).astype(np.float32)


def load_or_build_trajectory(args, audio_feature: np.ndarray, num_frames: int) -> np.ndarray:
    if args.target_traj:
        traj = np.load(args.target_traj).astype(np.float32)
        if traj.ndim == 3:
            traj = traj[0]
        traj = traj[:, :2]
        traj = resample_feature(traj, num_frames)
    else:
        points = parse_trajectory_points(args.trajectory)
        progress = build_progress_from_audio(
            audio_feature,
            use_audio_timing=not args.uniform_trajectory_timing,
        )
        traj = interpolate_trajectory(points, progress)

    if not args.keep_trajectory_absolute:
        traj = traj - traj[0:1]

    return traj.astype(np.float32)


def normalize_trajectory(traj_physical: np.ndarray, normalizer) -> np.ndarray:
    if normalizer is None:
        raise ValueError("trajectory normalization 需要 checkpoint normalizer")

    mean_x = float(normalizer.mean[ROOT_X_IDX])
    mean_z = float(normalizer.mean[ROOT_Z_IDX])
    std_x = float(normalizer.std[ROOT_X_IDX])
    std_z = float(normalizer.std[ROOT_Z_IDX])

    traj_norm = traj_physical.copy()
    traj_norm[:, 0] = (traj_norm[:, 0] - mean_x) / (std_x + 1e-8)
    traj_norm[:, 1] = (traj_norm[:, 1] - mean_z) / (std_z + 1e-8)

    return traj_norm.astype(np.float32)


def make_keyframe_feature_mask(num_frames: int, constrain_contacts: bool) -> np.ndarray:
    mask_one_frame = np.zeros((151,), dtype=np.float32)

    # 2D/3D 骨架图通常没有可靠脚接触标签，默认不锁 contacts。
    if constrain_contacts:
        mask_one_frame[CONTACT_SLICE] = 1.0

    # root X/Z 交给 trajectory 控制，不在 keyframe 里锁死。
    mask_one_frame[ROOT_Y_IDX] = 1.0
    mask_one_frame[ROT_SLICE] = 1.0

    return mask_one_frame


def build_constraint(args, normalizer, num_frames: int, device) -> dict:
    value = np.zeros((num_frames, 151), dtype=np.float32)
    mask = np.zeros((num_frames, 151), dtype=np.float32)

    frame_mask = make_keyframe_feature_mask(
        num_frames=num_frames,
        constrain_contacts=args.constrain_contacts,
    )

    def add_pose(path: str, frame: int, name: str):
        pose = load_151_pose(path)
        pose = normalize_pose_if_needed(pose, normalizer, args.pose_space)
        value[frame] = pose
        mask[frame] = frame_mask
        print(f"✅ 已添加 {name} 关键帧: frame={frame}, path={path}")

    add_pose(args.start_pose, 0, "start")
    add_pose(args.end_pose, num_frames - 1, "end")

    mid_paths = parse_list(args.mid_poses)
    mid_frames = parse_mid_frames(args.mid_pose_frames, len(mid_paths), num_frames)

    for i, (path, frame) in enumerate(zip(mid_paths, mid_frames), start=1):
        add_pose(path, frame, f"mid{i}")

    constraint = {
        "value": torch.from_numpy(value[None]).to(device=device, dtype=torch.float32),
        "mask": torch.from_numpy(mask[None]).to(device=device, dtype=torch.float32),
    }
    return constraint


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
def save_eval_assets(
    out_path: str,
    motion_raw_physical: np.ndarray,
    motion_final_physical: np.ndarray,
    traj_physical: np.ndarray,
    args,
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
        "--motion", str(final_motion_path),
        "--raw_motion", str(raw_motion_path),
        "--post_motion", str(final_motion_path),
        "--checkpoint", args.checkpoint,
        "--audio", args.music,
        "--target_traj", str(target_traj_path),
        "--start_pose", args.start_pose,
        "--end_pose", args.end_pose,
        "--out_json", str(eval_json_path),
        "--out_csv", str(eval_csv_path),
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
        "trajectory": getattr(args, "trajectory", ""),
        "target_traj": getattr(args, "target_traj", ""),
        "motion_raw": str(raw_motion_path),
        "motion_final": str(final_motion_path),
        "target_traj_saved": str(target_traj_path),
        "eval_json": str(eval_json_path),
        "eval_csv": str(eval_csv_path),
        "eval_command": " ".join(eval_cmd),
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

def main():
    parser = argparse.ArgumentParser(
        description="Controlled EDGE generation: music + start/end keyframes + 2D trajectory."
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--music", required=True)
    parser.add_argument("--start_pose", required=True)
    parser.add_argument("--end_pose", required=True)

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
        "--trajectory",
        default="0,0",
        help="轨迹控制点，例如 '0,0;1,1;0,2;-1,3;0,4'。",
    )
    parser.add_argument(
        "--target_traj",
        default="",
        help="可选，直接读取 [T,2] 或 [1,T,2] 的 .npy 轨迹，优先级高于 --trajectory。",
    )

    parser.add_argument("--out", required=True)
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
        help="当前数据管线里的关键帧 .npy 多数是 normalized；如果来自图像转 3D 的物理空间结果，用 physical。",
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
        help="生成后直接把 root X/Z 替换成目标轨迹。轨迹最严格，但可能增加脚滑。",
    )
    parser.add_argument(
        "--save_controls",
        action="store_true",
        help="额外保存 target trajectory 和 meta json，方便评估。",
    )
    parser.add_argument(
        "--target_traj",
        default="",
        help="可选，直接读取 [T,2] 或 [1,T,2] 的目标轨迹 .npy，优先级高于 --trajectory",
    )

    parser.add_argument(
        "--keep_trajectory_absolute",
        action="store_true",
        help="默认把轨迹平移到从 0 开始；开启后保留绝对 X/Z 坐标",
    )

    parser.add_argument(
        "--uniform_trajectory_timing",
        action="store_true",
        help="默认用音乐 onset 调整轨迹速度；开启后匀速走完整条轨迹",
    )

    parser.add_argument(
        "--linear_trajectory",
        action="store_true",
        help="关闭 spline 平滑插值，改用线性折线轨迹",
    )
    parser.add_argument(
        "--save_eval_assets",
        action="store_true",
        help="保存 raw motion、target trajectory、meta，并打印 eval_quantitative.py 命令",
    )

    parser.add_argument(
        "--post_anchor_trajectory",
        action="store_true",
        help="生成后强制把 root X/Z 替换成目标轨迹。轨迹误差最低，但可能增加脚滑。",
    )
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

    device = model.accelerator.device
    normalizer = model.normalizer

    if normalizer is None:
        raise ValueError("checkpoint 中没有 normalizer，无法保证轨迹/姿态空间一致。")

    print("🎵 提取 hybrid audio feature")
    audio_feature, audio_feature_path = hybrid_extract(args.music)
    audio_feature = np.asarray(audio_feature, dtype=np.float32)

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
        target_traj_path=getattr(args, "target_traj", ""),
        keep_absolute=getattr(args, "keep_trajectory_absolute", False),
        uniform_timing=getattr(args, "uniform_trajectory_timing", False),
        smooth=not getattr(args, "linear_trajectory", False),
    )

    cond = {
        "audio": torch.from_numpy(audio_feature[None]).to(
            device=model.accelerator.device,
            dtype=torch.float32,
        ),
        "trajectory": torch.from_numpy(traj_norm[None]).to(
            device=model.accelerator.device,
            dtype=torch.float32,
        ),
    }

    constraint = build_constraint(args, normalizer, num_frames, device)

    with torch.no_grad():
        sample_norm = sample_motion(model, cond, constraint, args, num_frames)

    motion_norm = sample_norm.detach().cpu()
    motion_physical_raw = model.normalizer.unnormalize(motion_norm).numpy()[0].astype(np.float32)
    motion_physical_final = motion_physical_raw.copy()

    if getattr(args, "post_anchor_trajectory", False):
        print("📌 post_anchor_trajectory 已开启：将 root X/Z 替换为目标轨迹。")
        motion_physical_final[:, 4] = traj_physical[:, 0]
        motion_physical_final[:, 6] = traj_physical[:, 1]

    if getattr(args, "save_eval_assets", False):
        save_eval_assets(
            out_path=args.out,
            motion_raw_physical=motion_physical_raw,
            motion_final_physical=motion_physical_final,
            traj_physical=traj_physical,
            args=args,
        )
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, motion_physical_final.astype(np.float32))
        print(f"✅ saved motion: {args.out}, shape={motion_physical_final.shape}")

    if args.save_controls:
        out_path = Path(args.out)
        traj_path = out_path.with_name(out_path.stem + "_target_traj.npy")
        raw_path = out_path.with_name(out_path.stem + "_raw.npy")
        meta_path = out_path.with_name(out_path.stem + "_meta.json")

        np.save(traj_path, traj_physical.astype(np.float32))
        np.save(raw_path, raw_motion_physical.astype(np.float32))

        meta = {
            "checkpoint": args.checkpoint,
            "music": args.music,
            "audio_feature_path": audio_feature_path,
            "start_pose": args.start_pose,
            "end_pose": args.end_pose,
            "mid_poses": parse_list(args.mid_poses),
            "mid_pose_frames": parse_mid_frames(args.mid_pose_frames, len(parse_list(args.mid_poses)), num_frames),
            "trajectory": args.trajectory,
            "target_traj": args.target_traj,
            "out": args.out,
            "num_frames": num_frames,
            "fps": args.fps,
            "pose_space": args.pose_space,
            "sampler": args.sampler,
            "use_tto": not args.no_tto,
            "hard_keyframe_project": not args.no_hard_keyframe_project,
            "post_anchor_trajectory": args.post_anchor_trajectory,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"✅ 已保存 target trajectory: {traj_path}")
        print(f"✅ 已保存 raw motion: {raw_path}")
        print(f"✅ 已保存 meta: {meta_path}")


if __name__ == "__main__":
    main()