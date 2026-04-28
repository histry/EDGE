import os
import torch
import numpy as np
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion
from vis import SMPLSkeleton # ✨ 修复：正确导入物理引擎

# 1. 替换原函数
def apply_hard_mask(mask, value, target_pose_norm, peak_idx, is_start=True, window_size=3):
    """
    ✨ 统一架构：采用 1.0 的绝对硬约束 (Hard Masking)
    废弃软化掩码，保护扩散模型去噪过程的方差不变性 (Diffusion Variance)。
    过渡的平滑任务完全交由模型自身的去噪网络解决。
    """
    device = mask.device
    dtype = mask.dtype
    target_pose = target_pose_norm.to(device).to(dtype)
    seq_len = mask.shape[1]
    
    for i in range(window_size):
        frame_idx = peak_idx + i if is_start else peak_idx - i
        if 0 <= frame_idx < seq_len:
            mask[:, frame_idx, :] = 1.0  # 绝对硬锁定
            value[:, frame_idx, :] = target_pose 
    return mask, value

def run_pure_motion_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    checkpoint_path = "runs/train/exp16/weights/train-300.pt"

    model = DanceDecoder(
        nfeats=151,           
        seq_len=150,          
        cond_feature_dim=803, 
        latent_dim=512,   
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    normalizer = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('ema_state_dict')))
        
        # 🌟 修复：反序列化提取 Normalizer
        norm_data = checkpoint.get("normalizer")
        if isinstance(norm_data, dict) and "mean" in norm_data:
            from dataset.preprocess import Normalizer
            dummy = torch.zeros((1, 1, 151))
            normalizer = Normalizer(dummy)
            normalizer.mean = np.array(norm_data["mean"])
            normalizer.std = np.array(norm_data["std"])
        else:
            normalizer = norm_data

        if normalizer is None:
            raise ValueError("❌ 找不到 Normalizer，无法进行物理空间对齐！")
            
        print(f"✅ 成功加载训练权重及 Normalizer: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"⚠️ 警告: 找不到权重文件 {checkpoint_path}")
        
    model.eval()

    dummy_audio = torch.zeros(1, 150, 803).to(device).to(dtype) 

    n_frames = 150 
    mask = torch.zeros(1, n_frames, 1).to(device).to(dtype)
    value = torch.zeros(1, n_frames, 151).to(device).to(dtype)
    
    print("✨ 激活 Decay Mask 缓冲：使用全 0 物理姿态作为起止关键帧，并施加衰减掩码。")
    # decay_weights = [1.0, 0.85, 0.6, 0.35, 0.1]
    
    zero_prior_norm = normalizer.normalize(torch.zeros(1, 1, 151)).squeeze(0).squeeze(0)

    mask, value = apply_hard_mask(mask, value, zero_prior_norm, 0, is_start=True, window_size=3)
    mask, value = apply_hard_mask(mask, value, zero_prior_norm, n_frames - 1, is_start=False, window_size=3)

    # ✨ 核心修复：必须先实例化 SMPL
    active_smpl = SMPLSkeleton(device=device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=150,           
        repr_dim=151,          
        smpl=active_smpl,      # 👈 修复：传入实例而不是类       
        n_timestep=1000,       
        predict_epsilon=False  
    ).to(device)
    
    diffusion.normalizer = normalizer

    print("🚀 开始执行 AIST++ 纯动作空间插值推理 (Dummy Audio 模式)...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            constraint_dict = {
                "mask": mask,          
                "value": value   
            }
            output_motion = diffusion.inpaint_loop(
                shape=(1, n_frames, 151), 
                cond=dummy_audio,
                constraint=constraint_dict
            )

    output_motion_physical = normalizer.unnormalize(output_motion)
    output_np = output_motion_physical.squeeze(0).cpu().numpy()
    
    print("✨ 纯动作先验插值完成！(输出物理张量 Shape: {})".format(output_np.shape))
    
    os.makedirs("output", exist_ok=True)
    output_npy = "output/aist_pure_motion_test.npy"
    np.save(output_npy, output_np)
    print(f"🎉 成功导出物理动作参数张量: {output_npy}")

if __name__ == "__main__":
    run_pure_motion_inference()