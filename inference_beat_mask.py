import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
import librosa
import torch.nn.functional as F
from EDGE import EDGE
from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract

def run_beat_mask_inference():
    # 参数配置
    ckpt_path = "runs/train/exp16/weights/train-300.pt"
    wav_path = "custom_music/dunhuang_pipa.wav" 
    out_dir = "output/beat_masked_renders"
    os.makedirs(out_dir, exist_ok=True)

    print("🚀 启动节拍掩码驱动推理 (Beat Masking Strategy)...")

    # 1. 加载模型
    model = EDGE(feature_type="hybrid", checkpoint_path=ckpt_path, audio_dim=803)
    model.eval()

    # 2. 提取原始高维音频特征
    raw_feat, _ = hybrid_extract(wav_path)
    
    # 3. 🌟 核心逻辑：提取节拍并生成掩码
    y, sr = librosa.load(wav_path, sr=None)
    # 计算起始能量强度 (Onset Strength)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # 将包络线对齐到 30 FPS 的帧数
    duration = librosa.get_duration(y=y, sr=sr)
    target_frames = int(duration * 30)
    
    # 线性插值包络线
    onset_resampled = F.interpolate(
        torch.from_numpy(onset_env).float().view(1, 1, -1),
        size=target_frames,
        mode='linear'
    ).view(-1).numpy()

    # 生成硬掩码：强度大于均值的部分保留，其余区域强制归零（模拟微调时的静音状态）
    threshold = np.mean(onset_resampled) * 1.5 
    beat_mask = (onset_resampled > threshold).astype(np.float32)

    # 4. 应用掩码
    # 对原始特征进行 30FPS 对齐并应用节拍掩码
    raw_feat_t = torch.from_numpy(raw_feat).float().unsqueeze(0).transpose(1, 2)
    aligned_feat = F.interpolate(raw_feat_t, size=target_frames, mode='linear').transpose(1, 2).squeeze(0)
    
    # 乘性注入：将非节拍区域的特征“静音化”
    masked_feat = aligned_feat * torch.from_numpy(beat_mask).unsqueeze(-1)
    
    # 代码/inference_beat_mask.py
    audio_tensor = masked_feat[:150].unsqueeze(0).to(model.accelerator.device).to(torch.bfloat16)
    traj_tensor = torch.zeros(1, 150, 2, device=model.accelerator.device, dtype=torch.bfloat16)
    
    if model.normalizer is not None:
        traj_tensor[..., 0] = (traj_tensor[..., 0] - model.normalizer.mean[4]) / (model.normalizer.std[4] + 1e-6)
        traj_tensor[..., 1] = (traj_tensor[..., 1] - model.normalizer.mean[6]) / (model.normalizer.std[6] + 1e-6)
        
    cond = {"audio": audio_tensor, "trajectory": traj_tensor}
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # 🌟 修复：严格对齐 EDGE.render_sample 的 4 元组解包要求 (x, cond, name, wav)
            model.render_sample(
                (None, cond, ["beat_mask_result"], [wav_path]), 
                label="beat_mask_result", 
                render_dir=out_dir
            )

if __name__ == "__main__":
    run_beat_mask_inference()