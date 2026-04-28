import os
import argparse
import glob
from contextlib import nullcontext
import torch
import torch.nn.functional as F
import numpy as np
import scipy.interpolate as spi
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from data.audio_extraction.wav2vec_librosa_features import extract 
from scipy.signal import find_peaks
import random
from vis import SMPLSkeleton # ✨ 正确导入物理引擎
from dataset.preprocess import Normalizer
from dataset.quaternion import quat_from_6v, quat_slerp, quat_to_6v
from trajectory_postprocess import TRAJECTORY_POST_MODES, apply_trajectory_postprocess

class DunhuangRAGSystem:
    def __init__(self, db_dir="data/dunhuang_rag_db"):
        self.db_dir = db_dir
        self.db_records = self._load_db()

    def _load_db(self):
        records = []
        if os.path.exists(self.db_dir):
            for file in glob.glob(f"{self.db_dir}/*.npy"):
                try:
                    data = np.load(file, allow_pickle=True).item()
                    records.append({
                        "audio_feat": torch.tensor(data['audio_feat']).float(),
                        "motion": torch.tensor(data['motion']).float()
                    })
                except Exception as e:
                    print(f"⚠️ 无法加载先验库文件 {file}: {e}")
        return records

    def _temporal_pool(self, feat, num_bins=4):
        # 自动兼容 2D输入(seq_len, dim) 或 3D输入(batch, seq_len, dim)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            
        # (batch, seq_len, dim) -> (batch, dim, seq_len)
        feat_t = feat.transpose(1, 2)
        
        # 时序降维: (batch, dim, num_bins)
        pooled = F.adaptive_avg_pool1d(feat_t, num_bins)
        
        # 安全展平特征维度，保留 Batch 维度: (batch, dim * num_bins)
        return pooled.flatten(start_dim=1)

    def retrieve_prior(self, test_audio_feat, top_k=1):
        if not self.db_records:
            return None
        best_score = -1.0
        best_motion = None
        
        # 【修复】：使用时序分块池化替代全局均值，保留音乐起承转合的节奏信息
        test_global_feat = self._temporal_pool(test_audio_feat).mean(dim=0)
        for record in self.db_records:
            db_global_feat = self._temporal_pool(record['audio_feat'].to(test_global_feat.device)).mean(dim=0)
            if db_global_feat.numel() != test_global_feat.numel():
                print(
                    f"⚠️ 跳过 RAG 记录：音频特征维度不一致 "
                    f"({db_global_feat.numel()} vs {test_global_feat.numel()})"
                )
                continue
            sim = F.cosine_similarity(test_global_feat, db_global_feat, dim=0).item()
            if sim > best_score:
                best_score = sim
                best_motion = record['motion']
        return best_motion

# 修改后：将其替换为具有衰减权重的锚点掩码函数
def apply_dynamic_clip_mask(mask, value, target_clip_norm, peak_idx, is_start=True, window_size=3):
    device = mask.device
    dtype = mask.dtype
    target_clip = target_clip_norm.to(device=device, dtype=dtype)

    if target_clip.dim() == 3:
        target_clip = target_clip.squeeze(0)

    seq_len = mask.shape[1]
    window_size = max(1, int(window_size))
    if window_size == 1:
        weights = torch.ones((1,), device=device, dtype=dtype)
    else:
        t = torch.linspace(0.0, 1.0, window_size, device=device, dtype=dtype)
        smooth_t = 6 * (t ** 5) - 15 * (t ** 4) + 10 * (t ** 3)
        weights = 0.12 + 0.88 * (1.0 - smooth_t)

    for i in range(window_size):
        frame_idx = peak_idx + i if is_start else peak_idx - i
        clip_idx = min(i, target_clip.shape[0] - 1)
        if 0 <= frame_idx < seq_len:
            frame_weight = weights[i].view(1, 1)
            prev_mask = mask[:, frame_idx, :]
            use_new = frame_weight >= prev_mask
            mask[:, frame_idx, :] = torch.maximum(prev_mask, frame_weight)
            value[:, frame_idx, :] = torch.where(
                use_new,
                target_clip[clip_idx].unsqueeze(0),
                value[:, frame_idx, :],
            )

    return mask, value


def motion_clip_is_normalized(clip):
    clip = clip.detach().float()
    if clip.numel() == 0:
        return False

    # Physical 6D rotations are bounded around [-1, 1]. Normalized motion priors
    # often exceed that range, especially in the rotation channels.
    rot_abs_p99 = torch.quantile(clip[..., 7:].abs().reshape(-1), 0.99)
    global_std = clip.reshape(-1).std()
    return bool((rot_abs_p99 > 1.35 or global_std > 0.75).item())


def normalize_prior_clip(anchor_clip, normalizer):
    if motion_clip_is_normalized(anchor_clip):
        return anchor_clip.unsqueeze(0)
    return normalizer.normalize(anchor_clip.unsqueeze(0))


def apply_pose_clip_mask(mask, value, target_clip_norm, center_idx, feature_mask=None):
    device = mask.device
    dtype = mask.dtype
    target_clip = target_clip_norm.to(device=device, dtype=dtype)

    if target_clip.dim() == 3:
        target_clip = target_clip.squeeze(0)
    if target_clip.dim() == 1:
        target_clip = target_clip.unsqueeze(0)

    seq_len = mask.shape[1]
    clip_len = target_clip.shape[0]
    start = int(center_idx) - clip_len // 2

    if feature_mask is None:
        feature_mask = torch.zeros((mask.shape[-1],), device=device, dtype=dtype)
        feature_mask[:4] = 1.0
        feature_mask[7:] = 1.0
    else:
        feature_mask = feature_mask.to(device=device, dtype=dtype)

    if feature_mask.dim() == 1:
        feature_mask = feature_mask.view(1, -1).expand(clip_len, -1)
    elif feature_mask.dim() == 2:
        if feature_mask.shape[0] != clip_len or feature_mask.shape[1] != mask.shape[-1]:
            raise ValueError(
                f"feature_mask shape must be ({clip_len}, {mask.shape[-1]}), got {tuple(feature_mask.shape)}"
            )
    else:
        raise ValueError(f"feature_mask must be 1D or 2D, got {feature_mask.dim()}D")

    for clip_idx in range(clip_len):
        frame_idx = start + clip_idx
        if 0 <= frame_idx < seq_len:
            frame_mask = feature_mask[clip_idx].view(1, -1)
            prev_mask = mask[:, frame_idx, :]
            use_new = frame_mask >= prev_mask
            mask[:, frame_idx, :] = torch.maximum(prev_mask, frame_mask)
            value[:, frame_idx, :] = torch.where(
                (frame_mask > 0) & use_new,
                target_clip[clip_idx].unsqueeze(0),
                value[:, frame_idx, :],
            )

    return mask, value


def split_path_list(raw_value):
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.replace(";", ",").split(",") if item.strip()]


def parse_mid_pose_frames(args, num_poses, seq_len):
    if num_poses <= 0:
        return []

    raw_frames = getattr(args, "mid_pose_frames", "")
    raw_ratios = getattr(args, "mid_pose_ratios", "")

    if raw_frames:
        frames = []
        for item in raw_frames.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            frame = int(float(item))
            if frame < 0:
                frame = seq_len + frame
            frames.append(frame)
    elif raw_ratios:
        frames = []
        for item in raw_ratios.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            ratio = float(item)
            if ratio > 1.0:
                ratio = ratio / 100.0
            frames.append(int(round(ratio * (seq_len - 1))))
    else:
        frames = [
            int(round((idx + 1) * (seq_len - 1) / (num_poses + 1)))
            for idx in range(num_poses)
        ]

    if len(frames) != num_poses:
        raise ValueError(
            f"--mid_poses 提供了 {num_poses} 个姿态，但中间帧位置有 {len(frames)} 个。"
        )

    return [max(1, min(seq_len - 2, frame)) for frame in frames]


def build_local_pose_feature_mask(
    num_features,
    strength,
    device,
    dtype,
    include_root_xz=False,
    include_contacts=False,
):
    feature_mask = torch.zeros((num_features,), device=device, dtype=dtype)
    strength = float(np.clip(strength, 0.0, 1.0))
    if include_contacts:
        feature_mask[:4] = strength
    feature_mask[5] = strength
    feature_mask[7:] = strength
    if include_root_xz:
        feature_mask[4] = strength
        feature_mask[6] = strength
    return feature_mask


def temporal_feather_weights(width, device, dtype):
    width = max(1, int(width))
    if width == 1:
        return torch.ones((1,), device=device, dtype=dtype)
    weights = torch.hann_window(width + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    max_weight = torch.clamp(weights.max(), min=1e-6)
    return weights / max_weight


def smoothstep01(x):
    return 6 * (x ** 5) - 15 * (x ** 4) + 10 * (x ** 3)


def apply_pose_path_guidance(
    mask,
    value,
    anchor_frames,
    anchor_poses,
    feature_mask,
    hold_frames=0,
    end_lead_frames=0,
):
    if len(anchor_frames) < 2:
        return mask, value

    seq_len = mask.shape[1]
    order = sorted(range(len(anchor_frames)), key=lambda idx: anchor_frames[idx])
    anchor_frames = [max(0, min(seq_len - 1, int(anchor_frames[idx]))) for idx in order]
    anchor_poses = [anchor_poses[idx].to(device=mask.device, dtype=mask.dtype) for idx in order]
    feature_mask = feature_mask.to(device=mask.device, dtype=mask.dtype).view(1, -1)

    hold_frames = max(0, int(hold_frames))
    end_lead_frames = max(0, int(end_lead_frames))

    for idx in range(len(anchor_frames) - 1):
        left_frame = anchor_frames[idx]
        right_frame = anchor_frames[idx + 1]
        if right_frame <= left_frame:
            continue

        left_pose = anchor_poses[idx]
        right_pose = anchor_poses[idx + 1]
        segment_len = right_frame - left_frame + 1
        left_hold = min(hold_frames, max(0, segment_len // 4))
        right_hold = hold_frames
        if idx == len(anchor_frames) - 2:
            right_hold = max(right_hold, end_lead_frames)
        right_hold = min(right_hold, max(0, segment_len - left_hold - 2))

        offsets = torch.arange(segment_len, device=mask.device, dtype=mask.dtype)
        denom = max(1, segment_len - 1 - left_hold - right_hold)
        alphas = torch.clamp((offsets - left_hold) / float(denom), 0.0, 1.0)
        alphas = smoothstep01(alphas)
        segment = left_pose.unsqueeze(0) * (1.0 - alphas[:, None]) + right_pose.unsqueeze(0) * alphas[:, None]

        for offset in range(segment_len):
            frame_idx = left_frame + offset
            prev_mask = mask[:, frame_idx, :]
            use_new = feature_mask >= prev_mask
            mask[:, frame_idx, :] = torch.maximum(prev_mask, feature_mask)
            value[:, frame_idx, :] = torch.where(
                (feature_mask > 0) & use_new,
                segment[offset].view(1, -1),
                value[:, frame_idx, :],
            )

    return mask, value


def generate_trajectory_tensor(control_points_str, seq_len, device, dtype, audio_feat_np=None):
    control_points = []
    if control_points_str:
        try:
            for p in control_points_str.split(';'):
                x, z = p.split(',')
                control_points.append([float(x.strip()), float(z.strip())])
        except Exception: pass

    if not control_points:
        traj_np = np.zeros((seq_len, 2))
    elif len(control_points) == 1:
        traj_np = np.tile(np.array(control_points[0]), (seq_len, 1))
    else:
        pts = np.array(control_points).T
        k_val = min(3, len(control_points) - 1) 
        tck, _ = spi.splprep(pts, s=0, k=k_val)
        
        if audio_feat_np is not None and audio_feat_np.shape[-1] >= 769:
            onset_strength = audio_feat_np[:, 768]
            speed_curve = onset_strength + (0.2 * np.max(onset_strength) if np.max(onset_strength) > 0 else 1.0)
            cum_progress = np.cumsum(speed_curve)
            u_new = (cum_progress - cum_progress[0]) / (cum_progress[-1] - cum_progress[0])
        else:
            u_new = np.linspace(0, 1, seq_len)
            
        x_new, z_new = spi.splev(u_new, tck)
        traj_np = np.stack([x_new, z_new], axis=1)

    return torch.tensor(traj_np, device=device, dtype=dtype).unsqueeze(0)


def _rotation_spike_scores(motion_np):
    rot_6d = torch.from_numpy(motion_np[:, 7:].reshape(motion_np.shape[0], 24, 6)).float()
    with torch.no_grad():
        from pytorch3d.transforms import rotation_6d_to_matrix

        rot_mats = rotation_6d_to_matrix(rot_6d)
        rel_rot = torch.matmul(rot_mats[:-1].transpose(-1, -2), rot_mats[1:])
        trace = rel_rot.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angles = torch.acos(torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6))
        angular_velocity = torch.linalg.norm(angles, dim=-1).cpu().numpy()

    angular_acceleration = np.zeros_like(angular_velocity)
    if len(angular_velocity) > 1:
        angular_delta = np.abs(np.diff(angular_velocity))
        angular_acceleration[:-1] = np.maximum(angular_acceleration[:-1], angular_delta)
        angular_acceleration[1:] = np.maximum(angular_acceleration[1:], angular_delta)
    return angular_velocity, angular_acceleration


def suppress_motion_spikes(motion_np, threshold=0.10, radius=16, min_separation=8, protected_frames=None):
    if motion_np.shape[0] < 5 or motion_np.shape[1] < 151:
        return motion_np, [], 0.0

    filtered = motion_np.copy()
    local = motion_np[:, np.r_[5, 7:motion_np.shape[1]]]
    local_delta = np.diff(local, axis=0)
    velocity = np.linalg.norm(local_delta, axis=1)

    acceleration = np.zeros_like(velocity)
    if len(velocity) > 1:
        accel_delta = np.linalg.norm(np.diff(local_delta, axis=0), axis=1)
        acceleration[:-1] = np.maximum(acceleration[:-1], accel_delta)
        acceleration[1:] = np.maximum(acceleration[1:], accel_delta)

    angular_velocity, angular_acceleration = _rotation_spike_scores(motion_np)

    velocity_threshold = max(float(threshold), float(np.percentile(velocity, 95) * 1.15))
    acceleration_threshold = max(float(threshold) * 0.55, float(np.percentile(acceleration, 95) * 0.80))
    angular_threshold = max(float(threshold) * 0.55, float(np.percentile(angular_velocity, 95) * 1.10))
    angular_accel_threshold = max(float(threshold) * 0.40, float(np.percentile(angular_acceleration, 95) * 0.95))

    feature_spikes = (velocity > velocity_threshold) & (acceleration > acceleration_threshold)
    angular_spikes = (angular_velocity > angular_threshold) & (angular_acceleration > angular_accel_threshold)
    candidates = np.where(feature_spikes | angular_spikes)[0]
    if len(candidates) == 0:
        return filtered, [], max(velocity_threshold, angular_threshold)

    centers = sorted(int(idx) + 1 for idx in candidates)
    cluster_gap = max(int(min_separation), int(radius) * 2 + 2)
    clusters = []
    current = [centers[0]]
    for center in centers[1:]:
        if center - current[-1] <= cluster_gap:
            current.append(center)
        else:
            clusters.append(current)
            current = [center]
    clusters.append(current)

    protected = []
    if protected_frames is not None:
        protected = sorted(
            max(0, min(filtered.shape[0] - 1, int(frame)))
            for frame in protected_frames
        )

    selected = []
    for cluster in clusters:
        center = int(round(sum(cluster) / len(cluster)))
        left = max(0, min(cluster) - radius - 1)
        right = min(filtered.shape[0] - 1, max(cluster) + radius + 1)
        if right - left < 2:
            continue
        if any(left <= frame <= right for frame in protected):
            continue
        selected.append(center)

        alpha = np.linspace(0.0, 1.0, right - left + 1, dtype=np.float32)
        smooth_alpha = 6 * alpha**5 - 15 * alpha**4 + 10 * alpha**3

        # Keep X/Z trajectory untouched; only ease vertical root and joint rotations.
        filtered[left : right + 1, 5] = (
            filtered[left, 5] * (1.0 - smooth_alpha)
            + filtered[right, 5] * smooth_alpha
        )

        with torch.no_grad():
            blend_t = torch.from_numpy(smooth_alpha).float().view(-1, 1)
            left_q = quat_from_6v(torch.from_numpy(filtered[left, 7:].reshape(1, 24, 6)).float())
            right_q = quat_from_6v(torch.from_numpy(filtered[right, 7:].reshape(1, 24, 6)).float())
            left_q = left_q.expand(len(smooth_alpha), -1, -1).clone()
            right_q = right_q.expand(len(smooth_alpha), -1, -1).clone()
            blended_q = quat_slerp(left_q, right_q, blend_t)
            blended_6d = quat_to_6v(blended_q).reshape(len(smooth_alpha), -1).cpu().numpy()
        filtered[left : right + 1, 7:] = blended_6d

    return filtered, selected, max(velocity_threshold, angular_threshold)


def smooth_motion_jitter(motion_np, window=5, strength=0.0, protected_frames=None):
    strength = float(np.clip(strength, 0.0, 1.0))
    window = max(1, int(window))
    if strength <= 0.0 or window <= 1 or motion_np.shape[0] < 3 or motion_np.shape[1] < 151:
        return motion_np

    if window % 2 == 0:
        window += 1

    smoothed = motion_np.copy()
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)

    root_y = motion_np[:, 5]
    root_y_padded = np.pad(root_y, (pad, pad), mode="edge")
    root_y_smooth = np.convolve(root_y_padded, kernel, mode="valid")
    smoothed[:, 5] = root_y * (1.0 - strength) + root_y_smooth * strength

    rot = motion_np[:, 7:].reshape(motion_np.shape[0], -1)
    rot_padded = np.pad(rot, ((pad, pad), (0, 0)), mode="edge")
    rot_smooth = np.empty_like(rot, dtype=np.float32)
    for channel in range(rot.shape[1]):
        rot_smooth[:, channel] = np.convolve(rot_padded[:, channel], kernel, mode="valid")

    with torch.no_grad():
        original_q = quat_from_6v(torch.from_numpy(motion_np[:, 7:].reshape(-1, 24, 6)).float())
        smooth_q = quat_from_6v(torch.from_numpy(rot_smooth.reshape(-1, 24, 6)).float())
        blend = torch.full_like(original_q[..., 0], strength)
        blended_q = quat_slerp(original_q, smooth_q, blend)
        smoothed[:, 7:] = quat_to_6v(blended_q).reshape(motion_np.shape[0], -1).cpu().numpy()

    # Preserve the planned floor trajectory exactly.
    smoothed[:, 4] = motion_np[:, 4]
    smoothed[:, 6] = motion_np[:, 6]
    if protected_frames is not None:
        for frame in protected_frames:
            frame = max(0, min(smoothed.shape[0] - 1, int(frame)))
            smoothed[frame, 5] = motion_np[frame, 5]
            smoothed[frame, 7:] = motion_np[frame, 7:]
    return smoothed


def smooth_root_trajectory_jitter(motion_np, window=7, strength=0.0):
    strength = float(np.clip(strength, 0.0, 1.0))
    window = max(1, int(window))
    if strength <= 0.0 or window <= 1 or motion_np.shape[0] < 3 or motion_np.shape[1] < 7:
        return motion_np

    if window % 2 == 0:
        window += 1

    smoothed = motion_np.copy()
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)

    for column in (4, 6):
        root_axis = motion_np[:, column]
        padded = np.pad(root_axis, (pad, pad), mode="edge")
        root_smooth = np.convolve(padded, kernel, mode="valid")
        smoothed[:, column] = root_axis * (1.0 - strength) + root_smooth * strength

        # Keep the absolute start/end anchors exactly on the planned path.
        drift = np.linspace(
            smoothed[0, column] - motion_np[0, column],
            smoothed[-1, column] - motion_np[-1, column],
            motion_np.shape[0],
            dtype=np.float32,
        )
        smoothed[:, column] -= drift

    return smoothed


# 🌟 新增：分块切片器（防止长音频撑爆显存）
def chunk_tensor(tensor, horizon=150, stride=75, target_frames=150, is_mask=False):
    tensor = tensor.squeeze(0) 
    chunks = []
    
    def safe_pad(seq, target_len, is_mask):
        pad_len = target_len - seq.shape[0]
        if is_mask:
            pad_tensor = torch.zeros_like(seq[-1:]).repeat(pad_len, 1)
            return torch.cat([seq, pad_tensor], dim=0)
        else:
            seq_np = seq.cpu().numpy()
            if seq_np.shape[0] == 1:
                seq_np = np.repeat(seq_np, target_len, axis=0)
            else:
                curr_pad = pad_len
                while curr_pad > 0:
                    step_pad = min(curr_pad, seq_np.shape[0] - 1)
                    seq_np = np.pad(seq_np, ((0, step_pad), (0, 0)), mode='reflect')
                    curr_pad -= step_pad
            return torch.from_numpy(seq_np).to(seq.device).to(seq.dtype)

    if target_frames <= horizon:
        chunk = safe_pad(tensor, horizon, is_mask)
        chunks.append(chunk)
    else:
        for i in range(0, target_frames, stride):
            chunk = tensor[i : i + horizon]
            if chunk.shape[0] < horizon:
                chunk = safe_pad(chunk, horizon, is_mask)
            chunks.append(chunk)
            if i + horizon >= target_frames:
                break
    return torch.stack(chunks, dim=0)

def run_music_driven_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    print(f"🎵 正在提取测试音乐特征: {args.audio}")
    audio_feat_np, _ = extract(args.audio) 
    audio_seq_len = audio_feat_np.shape[0]
    
    if audio_seq_len % 2 != 0:
        audio_seq_len -= 1
        audio_feat_np = audio_feat_np[:audio_seq_len]
        
    cond_audio = torch.tensor(audio_feat_np, device=device, dtype=dtype).unsqueeze(0)
    has_traj = bool(args.trajectory and args.trajectory.strip())
    physical_target_traj = generate_trajectory_tensor(
        args.trajectory, audio_seq_len, device, dtype, audio_feat_np
    )
    
    # 以首帧为物理原点，同时保留一份物理量纲轨迹用于最终锚定。
    start_xz_abs = physical_target_traj[:, 0, :].clone()
    physical_target_traj = physical_target_traj - start_xz_abs.unsqueeze(1)
    normalized_cond_traj = physical_target_traj.clone()
    
    # ====== 全局执行量纲对齐 ======
    
    # 🌟 核心修复 1：把 Normalizer 的加载提到全局归一化之前！
    # ✅ Fixed Code
    normalizer = None
    if os.path.exists(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
        state_dict = checkpoint.get('ema_state_dict', checkpoint.get('model_state_dict'))
        
        norm_data = checkpoint.get("normalizer")
        if isinstance(norm_data, dict) and "mean" in norm_data:
            dummy = torch.zeros((1, 1, 151))
            normalizer = Normalizer(dummy)
            normalizer.mean = np.array(norm_data["mean"])
            normalizer.std = np.array(norm_data["std"])
        else:
            normalizer = norm_data
    else:
        raise FileNotFoundError("❌ 找不到权重文件")
        
    # ====== 全局执行量纲对齐 ======
    if normalizer is not None:
        mean_x = torch.tensor(normalizer.mean[4], device=device, dtype=dtype)
        mean_z = torch.tensor(normalizer.mean[6], device=device, dtype=dtype)
        std_x = torch.tensor(normalizer.std[4], device=device, dtype=dtype)
        std_z = torch.tensor(normalizer.std[6], device=device, dtype=dtype)
        
        normalized_cond_traj[:, :, 0] = (normalized_cond_traj[:, :, 0] - mean_x) / (std_x + 1e-6)
        normalized_cond_traj[:, :, 1] = (normalized_cond_traj[:, :, 1] - mean_z) / (std_z + 1e-6)
    
    # 🌟 核心修复 2：彻底删除切块内部的反归一化逻辑，进行基于物理空间的相对平移
    # (切块操作将延后到所有 Mask 特征组装完毕后统一进行)
    
    horizon = 150
    stride = horizon // 2

    # 强制定义模型时使用 150 长度
    model = DanceDecoder(
        nfeats=151, seq_len=horizon, cond_feature_dim=803, latent_dim=512, 
        ff_size=1024, num_layers=8, num_heads=8, dropout=0.1
    ).to(device)
    
    if 'null_cond_embed' in state_dict:
        saved_shape = state_dict['null_cond_embed'].shape 
        target_shape = model.null_cond_embed.shape        
        if saved_shape != target_shape:
            saved_embed = state_dict['null_cond_embed'].permute(0, 2, 1) 
            resized_embed = F.interpolate(saved_embed, size=target_shape[1], mode='linear', align_corners=False)
            state_dict['null_cond_embed'] = resized_embed.permute(0, 2, 1)

    model.load_state_dict(state_dict)
    print("✅ 成功加载权重及 Normalizer")
        
    model.eval()

    rag_system = DunhuangRAGSystem(db_dir=args.rag_db_dir)
    use_rag = bool(args.enable_rag) and not bool(args.disable_rag)
    retrieved_motion = rag_system.retrieve_prior(cond_audio[0]) if use_rag else None

    mask = torch.zeros(1, audio_seq_len, 151).to(device).to(dtype)
    value = torch.zeros(1, audio_seq_len, 151).to(device).to(dtype)
    
    zero_prior_norm = normalizer.normalize(torch.zeros(1, 1, 151)).squeeze(0).squeeze(0) 
    def load_pose_norm(pose_arg, default_pose):
        if pose_arg and os.path.exists(pose_arg):
            try:
                loaded_pose = torch.tensor(np.load(pose_arg)).to(device).to(dtype)
                return loaded_pose
            except Exception as e:
                print(f"⚠️ 加载姿态文件 {pose_arg} 失败: {e}，回退到零先验。")
        return default_pose

    start_pose_norm = load_pose_norm(args.start_pose if hasattr(args, 'start_pose') else None, zero_prior_norm)
    end_pose_norm = load_pose_norm(args.end_pose if hasattr(args, 'end_pose') else None, zero_prior_norm)
    protected_filter_frames = [0, audio_seq_len - 1]

    if has_traj:
        start_pose_norm[4] = normalized_cond_traj[0, 0, 0].to(start_pose_norm.dtype)
        start_pose_norm[6] = normalized_cond_traj[0, 0, 1].to(start_pose_norm.dtype)
        end_pose_norm[4] = normalized_cond_traj[0, -1, 0].to(end_pose_norm.dtype)
        end_pose_norm[6] = normalized_cond_traj[0, -1, 1].to(end_pose_norm.dtype)

    endpoint_width = max(1, int(getattr(args, "endpoint_pose_width", 3)))
    mask, value = apply_dynamic_clip_mask(
        mask, value, start_pose_norm.unsqueeze(0), 0, is_start=True, window_size=endpoint_width
    )
    mask, value = apply_dynamic_clip_mask(
        mask, value, end_pose_norm.unsqueeze(0), audio_seq_len - 1, is_start=False, window_size=endpoint_width
    )

    pose_path_hold_frames = max(0, int(getattr(args, "pose_path_hold_frames", 0)))
    pose_path_hold_release_frames = max(0, int(getattr(args, "pose_path_hold_release_frames", 0)))
    pose_path_end_lead_frames = max(0, int(getattr(args, "pose_path_end_lead_frames", 0)))
    if pose_path_hold_frames > 0:
        hold_total_frames = pose_path_hold_frames + pose_path_hold_release_frames
        hold_feature_mask = build_local_pose_feature_mask(
            151,
            0.95,
            device,
            dtype,
            include_root_xz=False,
            include_contacts=False,
        )
        if pose_path_hold_release_frames > 0:
            hold_weights = torch.ones((hold_total_frames,), device=device, dtype=dtype) * 0.95
            release_t = torch.linspace(0.0, 1.0, pose_path_hold_release_frames, device=device, dtype=dtype)
            hold_weights[pose_path_hold_frames:] = 0.95 * (1.0 - smoothstep01(release_t)) + 0.12 * smoothstep01(release_t)
            hold_feature_mask = hold_weights[:, None] * hold_feature_mask.view(1, -1)
        hold_clip = start_pose_norm.view(1, -1).repeat(hold_total_frames, 1)
        mask, value = apply_pose_clip_mask(
            mask,
            value,
            hold_clip.unsqueeze(0),
            hold_total_frames // 2,
            feature_mask=hold_feature_mask,
        )

    if pose_path_end_lead_frames > 0:
        lead_feature_mask = build_local_pose_feature_mask(
            151,
            1.0,
            device,
            dtype,
            include_root_xz=False,
            include_contacts=False,
        )
        lead_t = torch.linspace(0.0, 1.0, pose_path_end_lead_frames, device=device, dtype=dtype)
        lead_weights = 0.2 + 0.75 * smoothstep01(lead_t)
        lead_clip = end_pose_norm.view(1, -1).repeat(pose_path_end_lead_frames, 1)
        mask, value = apply_pose_clip_mask(
            mask,
            value,
            lead_clip.unsqueeze(0),
            audio_seq_len - 1 - pose_path_end_lead_frames // 2,
            feature_mask=lead_weights[:, None] * lead_feature_mask.view(1, -1),
        )

    mid_pose_paths = split_path_list(getattr(args, "mid_poses", ""))
    mid_entries = []
    if mid_pose_paths:
        mid_pose_frames = parse_mid_pose_frames(args, len(mid_pose_paths), audio_seq_len)
        mid_width = max(1, int(getattr(args, "mid_pose_width", 1)))
        mid_strength = float(np.clip(getattr(args, "mid_pose_strength", 1.0), 0.0, 1.0))
        mid_include_root_xz = bool(getattr(args, "mid_pose_include_root_xz", False))
        mid_include_contacts = bool(getattr(args, "mid_pose_include_contacts", False))
        mid_feature_mask = build_local_pose_feature_mask(
            151,
            mid_strength,
            device,
            dtype,
            include_root_xz=mid_include_root_xz,
            include_contacts=mid_include_contacts,
        )

        mid_entries = []
        for pose_path, center_frame in zip(mid_pose_paths, mid_pose_frames):
            protected_filter_frames.append(center_frame)
            mid_pose_norm = load_pose_norm(pose_path, zero_prior_norm).clone()
            if has_traj:
                mid_pose_norm[4] = normalized_cond_traj[0, center_frame, 0].to(mid_pose_norm.dtype)
                mid_pose_norm[6] = normalized_cond_traj[0, center_frame, 1].to(mid_pose_norm.dtype)
            mid_entries.append((pose_path, center_frame, mid_pose_norm))

    path_strength = float(np.clip(getattr(args, "pose_path_strength", 0.0), 0.0, 1.0))
    if path_strength > 0:
        path_feature_mask = build_local_pose_feature_mask(
            151,
            path_strength,
            device,
            dtype,
            include_root_xz=False,
            include_contacts=False,
        )
        path_frames = [0] + [entry[1] for entry in mid_entries] + [audio_seq_len - 1]
        path_poses = [start_pose_norm] + [entry[2] for entry in mid_entries] + [end_pose_norm]
        mask, value = apply_pose_path_guidance(
            mask,
            value,
            path_frames,
            path_poses,
            path_feature_mask,
            hold_frames=pose_path_hold_frames,
            end_lead_frames=pose_path_end_lead_frames,
        )
        print(
            f"🧭 已启用弱姿态路径引导: strength={path_strength:.2f}, "
            f"anchors={len(path_frames)}, hold={pose_path_hold_frames}, "
            f"hold_release={pose_path_hold_release_frames}, end_lead={pose_path_end_lead_frames}"
        )

    if mid_entries:
        print("🎯 已启用推理中间关键帧约束：")
        temporal_weights = temporal_feather_weights(mid_width, device, dtype)
        temporal_feature_mask = temporal_weights[:, None] * mid_feature_mask.view(1, -1)
        for pose_path, center_frame, mid_pose_norm in mid_entries:
            clip_frames = []
            clip_start = center_frame - mid_width // 2
            for offset in range(mid_width):
                frame_idx = max(0, min(audio_seq_len - 1, clip_start + offset))
                frame_pose = mid_pose_norm.clone()
                if has_traj:
                    frame_pose[4] = normalized_cond_traj[0, frame_idx, 0].to(frame_pose.dtype)
                    frame_pose[6] = normalized_cond_traj[0, frame_idx, 1].to(frame_pose.dtype)
                clip_frames.append(frame_pose)

            mid_clip = torch.stack(clip_frames, dim=0)
            mask, value = apply_pose_clip_mask(
                mask,
                value,
                mid_clip.unsqueeze(0),
                center_frame,
                feature_mask=temporal_feature_mask,
            )
            print(
                f"   - frame={center_frame}, width={mid_width}, strength={mid_strength:.2f}, "
                f"root_xz={mid_include_root_xz}, pose={pose_path}"
            )

    print("🥁 正在分析音乐节拍，提取推理阶段节拍锚点...")
    if audio_feat_np.shape[-1] >= 769:
        onset_strength = audio_feat_np[:, 768]
        peaks, _ = find_peaks(onset_strength, distance=30, height=np.mean(onset_strength)*1.5)
        valid_peaks = [p for p in peaks if 30 < p < audio_seq_len - 30]
        valid_peaks = sorted(valid_peaks, key=lambda x: onset_strength[x], reverse=True)[:3]
        
        if valid_peaks and retrieved_motion is not None:
            print(f"✨ 找到 {len(valid_peaks)} 个核心重音节拍：{valid_peaks}，注入可选 RAG 姿态锚点。")
            record_motion = retrieved_motion.to(device) 
            
            for peak in valid_peaks:
                clip_len = max(1, int(args.rag_clip_len))
                if clip_len % 2 == 0:
                    clip_len += 1
                start_idx = random.randint(0, max(0, record_motion.shape[0] - clip_len))
                anchor_clip = record_motion[start_idx : start_idx + clip_len]
                
                if anchor_clip.shape[0] < clip_len:
                    pad = anchor_clip[-1:].repeat(clip_len - anchor_clip.shape[0], 1)
                    anchor_clip = torch.cat([anchor_clip, pad], dim=0)
                    
                anchor_clip_norm = normalize_prior_clip(anchor_clip, normalizer)

                # RAG 片段只提供局部身体姿态先验，不能硬覆盖全局 root。
                # 否则随机先验片段的 root_y/root_xz 会在重音处造成身体压缩、瞬移或轨迹断裂。
                feature_mask = torch.zeros((151,), device=device, dtype=dtype)
                rag_strength = float(np.clip(args.rag_pose_strength, 0.0, 1.0))
                feature_mask[:4] = rag_strength
                feature_mask[7:] = rag_strength
                mask, value = apply_pose_clip_mask(mask, value, anchor_clip_norm, peak, feature_mask=feature_mask)
        elif valid_peaks and retrieved_motion is None:
            print("⏭️ 已检测到重音；当前未启用有效 RAG 先验，仅保留预训练音频条件与节拍锚点轨迹引导。")
    
    N_cond_audio = chunk_tensor(cond_audio, horizon, stride, audio_seq_len)
    N_cond_traj  = chunk_tensor(normalized_cond_traj, horizon, stride, audio_seq_len)
    N_mask       = chunk_tensor(mask, horizon, stride, audio_seq_len, is_mask=True) 
    N_value      = chunk_tensor(value, horizon, stride, audio_seq_len)
    
    # ✨ 修复：彻底删除上述 for 循环中的局部 offset 代码。
    # 既然在全局已经做过了 (cond_traj - mean) / std，各块之间的相对位置在切块时自然就是对齐的。
    # 直接保留物理空间的绝对位置系，交给 Transformer 的自注意力去捕捉连贯性。

    N = N_cond_audio.shape[0]

    if has_traj:
        cond_dict = {"audio": N_cond_audio, "trajectory": N_cond_traj}
    else:
        cond_dict = {"audio": N_cond_audio}
        print("🕊️ 用户未指定轨迹，已解锁根节点位移限制，允许模型自由生成。")
        
    constraint_dict = {"mask": N_mask, "value": N_value}

    active_smpl = SMPLSkeleton(device=device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon,       
        repr_dim=151,          
        smpl=active_smpl,      
        n_timestep=1000,       
        predict_epsilon=False,
        guidance_weight=2.5  
    ).to(device)
    
    diffusion.normalizer = normalizer
    
    # ✅ Fixed Code
    print(f"🚀 开始执行多模态条件驱动生成 (切割为 {N} 块)...")
    # ✅ Fixed Code
    batch_size = max(1, int(getattr(args, "chunk_batch_size", 1)))
    batch_size = min(batch_size, N)
    step = batch_size  
    overlap_frames = max(1, int(getattr(args, "chunk_overlap_blend_frames", stride)))
    overlap_frames = min(overlap_frames, stride, horizon - stride)
    all_outputs = []
    
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad():
        with autocast_ctx:
            for i in range(0, N, step):
                end_idx = min(i + batch_size, N)
                current_batch_size = end_idx - i
                
                sub_cond_dict = {
                    "audio": cond_dict["audio"][i:end_idx]
                }
                if "trajectory" in cond_dict:
                    sub_cond_dict["trajectory"] = cond_dict["trajectory"][i:end_idx]
                
                sub_constraint_mask = constraint_dict["mask"][i:end_idx].clone()
                sub_constraint_value = constraint_dict["value"][i:end_idx].clone()
                
                # 滑动窗口：利用上一个 Batch 的结果作为当前 Batch 起始的硬约束
                if i > 0 and len(all_outputs) > 0:
                    prev_last_chunk = all_outputs[-1][-1:].to(device).clone()

                    overlap_value = prev_last_chunk[0, stride : stride + overlap_frames, :]
                    overlap_w = torch.linspace(
                        1.0,
                        0.05,
                        overlap_frames,
                        device=device,
                        dtype=sub_constraint_mask.dtype,
                    ).view(overlap_frames, 1)
                    prev_mask = sub_constraint_mask[0, :overlap_frames, :]
                    use_overlap = overlap_w >= prev_mask
                    sub_constraint_mask[0, :overlap_frames, :] = torch.maximum(prev_mask, overlap_w)
                    sub_constraint_value[0, :overlap_frames, :] = torch.where(
                        use_overlap,
                        overlap_value,
                        sub_constraint_value[0, :overlap_frames, :],
                    )

                sub_output = diffusion.long_inpaint_loop(
                    shape=(current_batch_size, horizon, 151), 
                    cond=sub_cond_dict,
                    constraint={"mask": sub_constraint_mask, "value": sub_constraint_value},
                    use_tto=args.use_tto
                )
                
                all_outputs.append(sub_output.detach().cpu())

    output_motion = torch.cat(all_outputs, dim=0)

    from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d
    from dataset.quaternion import quat_slerp, ax_from_6v

    output_motion_physical = normalizer.unnormalize(output_motion)
    current_motion = output_motion_physical[0].clone()
    
    for i in range(1, N):
        next_chunk = output_motion_physical[i].clone()

        # 绝对偏移量校正
        delta_x = current_motion[i * stride, 4] - next_chunk[0, 4]
        delta_z = current_motion[i * stride, 6] - next_chunk[0, 6]
        next_chunk[:, 4] += delta_x
        next_chunk[:, 6] += delta_z

        blend_len = min(overlap_frames, stride, next_chunk.shape[0])
        fade_w = torch.linspace(0, 1, blend_len, device=next_chunk.device).unsqueeze(1)
        fade_w_smooth = 6 * (fade_w ** 5) - 15 * (fade_w ** 4) + 10 * (fade_w ** 3)

        # 平滑融合边界
        next_chunk[:blend_len, 4:7] = (
            current_motion[i * stride : i * stride + blend_len, 4:7] * (1 - fade_w_smooth)
            + next_chunk[:blend_len, 4:7] * fade_w_smooth
        )

        q_curr = axis_angle_to_quaternion(
            ax_from_6v(current_motion[i * stride : i * stride + blend_len, 7:].reshape(blend_len, 24, 6))
        )
        q_next = axis_angle_to_quaternion(
            ax_from_6v(next_chunk[:blend_len, 7:].reshape(blend_len, 24, 6))
        )
        q_blended = quat_slerp(q_curr, q_next, fade_w)
        next_chunk[:blend_len, 7:] = matrix_to_rotation_6d(
            quaternion_to_matrix(q_blended)
        ).reshape(blend_len, 144)

        current_motion = torch.cat([current_motion[: i * stride], next_chunk], dim=0)

    output_np = current_motion[:audio_seq_len].cpu().float().numpy()
    
    target_traj_np = physical_target_traj[0, :audio_seq_len].cpu().float().numpy() if has_traj else None
    output_np = apply_trajectory_postprocess(
        output_np,
        target_traj=target_traj_np,
        mode=args.trajectory_post_mode,
        device=device,
    )

    if bool(getattr(args, "motion_spike_filter", False)):
        output_np, spike_frames, spike_threshold = suppress_motion_spikes(
            output_np,
            threshold=getattr(args, "motion_spike_threshold", 0.10),
            radius=getattr(args, "motion_spike_radius", 16),
            protected_frames=protected_filter_frames,
        )
        smooth_strength = float(getattr(args, "motion_global_smooth_strength", 0.80))
        if smooth_strength > 0:
            output_np = smooth_motion_jitter(
                output_np,
                window=getattr(args, "motion_global_smooth_window", 13),
                strength=smooth_strength,
                protected_frames=protected_filter_frames,
            )
        if spike_frames:
            print(
                f"🧹 已执行局部动作尖峰滤波: frames={spike_frames}, "
                f"threshold={spike_threshold:.4f}, radius={args.motion_spike_radius}, "
                f"global_smooth={smooth_strength:.2f}"
            )

    root_smooth_strength = float(getattr(args, "motion_root_smooth_strength", 0.0))
    if root_smooth_strength > 0:
        output_np = smooth_root_trajectory_jitter(
            output_np,
            window=getattr(args, "motion_root_smooth_window", 7),
            strength=root_smooth_strength,
        )
        print(
            f"🧭 已执行 root X/Z 轨迹微平滑: "
            f"window={args.motion_root_smooth_window}, strength={root_smooth_strength:.2f}"
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, output_np)
    print(f"🎉 生成完毕！已在物理空间完成无缝拼接，保存完整时长物理张量至: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多模态控制编舞推理")
    parser.add_argument("--ckpt", type=str, required=True, help="阶段性权重路径")
    parser.add_argument("--audio", type=str, required=True, help="测试音乐 .wav 路径")
    parser.add_argument("--out", type=str, default="output/test_driven.npy", help="输出文件路径")
    parser.add_argument("--trajectory", type=str, default="", help="2D (X,Z) 空间走位轨迹")
    parser.add_argument("--start_pose", type=str, default="", help="起始关键帧张量文件路径 (.npy)")
    parser.add_argument("--end_pose", type=str, default="", help="终止关键帧张量文件路径 (.npy)")
    parser.add_argument("--endpoint_pose_width", type=int, default=3, help="起止姿态约束帧宽，建议与 Stage1/2 训练保持 3")
    parser.add_argument("--mid_poses", type=str, default="", help="逗号或分号分隔的中间关键帧 .npy 文件列表")
    parser.add_argument("--mid_pose_frames", type=str, default="", help="中间关键帧所在输出帧号，例如 180,360")
    parser.add_argument("--mid_pose_ratios", type=str, default="", help="中间关键帧所在相对位置，例如 0.33,0.66；未提供时均匀分布")
    parser.add_argument("--mid_pose_width", type=int, default=15, help="每个中间关键帧约束的羽化帧宽，15 帧约 0.5 秒")
    parser.add_argument("--mid_pose_strength", type=float, default=0.55, help="中间关键帧软约束峰值强度")
    parser.add_argument("--pose_path_strength", type=float, default=0.12, help="首/中/尾姿态之间的弱连续姿态路径引导强度")
    parser.add_argument("--pose_path_hold_frames", type=int, default=0, help="每段姿态路径起点保留帧数，用于避免开头立刻跳到下一姿态")
    parser.add_argument("--pose_path_hold_release_frames", type=int, default=0, help="起始保留后逐步释放的帧数，用于避免保留窗口结束处突然跳变")
    parser.add_argument("--pose_path_end_lead_frames", type=int, default=0, help="最后一段提前收束到终点姿态的帧数，用于避免最后 1 秒突变")
    parser.add_argument("--chunk_batch_size", type=int, default=1, help="一次采样的长序列分块数量；20 秒音频可设为 7 以同步块间重叠区")
    parser.add_argument("--chunk_overlap_blend_frames", type=int, default=75, help="长音频分块拼接的重叠淡入淡出帧数；75 为 150 帧窗口的一半")
    parser.add_argument("--motion_spike_filter", action="store_true", help="对异常大的单帧姿态尖峰做局部平滑，不修改 X/Z 轨迹")
    parser.add_argument("--motion_spike_threshold", type=float, default=0.10, help="动作尖峰滤波阈值；会结合速度和加速度自适应阈值")
    parser.add_argument("--motion_spike_radius", type=int, default=16, help="动作尖峰局部平滑半径")
    parser.add_argument("--motion_global_smooth_strength", type=float, default=0.80, help="全局旋转弱平滑强度；不修改 X/Z 轨迹，调低可保留更多动作锐度")
    parser.add_argument("--motion_global_smooth_window", type=int, default=13, help="全局旋转弱平滑窗口，奇数更稳定")
    parser.add_argument("--motion_root_smooth_strength", type=float, default=0.0, help="可选 root X/Z 轨迹微平滑强度；会轻微偏离严格轨迹，用于消除轨迹折点造成的展示卡顿")
    parser.add_argument("--motion_root_smooth_window", type=int, default=7, help="root X/Z 轨迹微平滑窗口，奇数更稳定")
    parser.add_argument("--mid_pose_include_root_xz", action="store_true", help="中间姿态也约束 root XZ；默认关闭以避免轨迹跳变")
    parser.add_argument("--mid_pose_include_contacts", action="store_true", help="中间姿态也约束脚部接触通道；默认关闭以减少抖动")
    parser.add_argument(
        "--trajectory_post_mode",
        type=str,
        default="optimize",
        choices=TRAJECTORY_POST_MODES,
        help="trajectory postprocess mode: hard is strict path, optimize balances trajectory and foot contact",
    )
    parser.add_argument("--use_tto", action="store_true", help="enable lightweight test-time optimization during inpainting sampling")
    parser.add_argument("--enable_rag", action="store_true", help="enable retrieved Dunhuang motion priors at audio peaks")
    parser.add_argument("--disable_rag", action="store_true", help="disable retrieved Dunhuang motion priors at audio peaks")
    parser.add_argument("--rag_db_dir", type=str, default="data/dunhuang_rag_db", help="directory containing Dunhuang RAG prior .npy files")
    parser.add_argument("--rag_clip_len", type=int, default=15, help="odd-length pose-only RAG clip injected around strong audio peaks")
    parser.add_argument("--rag_pose_strength", type=float, default=0.35, help="soft mask strength for optional RAG pose anchors")
    args = parser.parse_args()
    
    run_music_driven_inference(args)
    # (打印成功的信息已经在 run_music_driven_inference 内部执行过了，这里直接结束即可)
