import os
import torch
import numpy as np
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion

def run_pure_motion_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    checkpoint_path = "runs/train/exp15/weights/train-1000.pt"  # 👈 指向你正在训的权重

    model = DanceDecoder(
        nfeats=151,           # <--- ✅ 151 维
        seq_len=150,          
        cond_feature_dim=803, 
        latent_dim=512,   
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint.get('ema_state_dict')))
        print(f"成功加载训练权重: {checkpoint_path}")
    else:
        print(f"⚠️ 警告: 找不到权重文件 {checkpoint_path}，使用随机初始化参数做测试。")
        
    model.eval()

    dummy_audio = torch.zeros(1, 150, 803).to(device).to(dtype) 

    n_frames = 150 
    ref_motions = torch.zeros(1, n_frames, 151).to(device).to(dtype) # <--- ✅ 151 维
    mask = torch.zeros(1, n_frames, 1).to(device).to(dtype)
    
    print("⚠️ 注意：当前使用全 0 姿态作为 AIST++ 的起止关键帧。")
    mask[:, 0, :] = 1.0 
    mask[:, 149, :] = 1.0 

    diffusion = GaussianDiffusion(
        model=model,
        horizon=150,           
        repr_dim=151,          # <--- ✅ 151 维
        smpl=None,             
        n_timestep=1000,       
        predict_epsilon=False  
    ).to(device)
    
    print("开始执行 AIST++ 纯动作空间插值推理 (Dummy Audio 模式)...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            constraint_dict = {
                "mask": mask,          
                "value": ref_motions   
            }
            output_motion = diffusion.inpaint_loop(
                shape=(1, n_frames, 151), # <--- ✅ 151 维
                cond=dummy_audio,
                constraint=constraint_dict
            )

    output_np = output_motion.squeeze(0).cpu().numpy()
    print("纯动作先验插值完成！(输出张量 Shape: {})".format(output_np.shape))
    
    os.makedirs("output", exist_ok=True)
    output_npy = "output/aist_pure_motion_test.npy"
    np.save(output_npy, output_np)
    print(f"成功导出动作参数张量: {output_npy}")

if __name__ == "__main__":
    run_pure_motion_inference()
