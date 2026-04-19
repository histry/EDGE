import os
import torch
import numpy as np
from model.model import DanceDecoder
from model.diffusion import GaussianDiffusion

# 【精准导入】使用你写好的 extract 函数
from data.audio_extraction.wav2vec_librosa_features import extract 

def run_music_driven_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    # 1. 准备音乐特征
    audio_path = "custom_music/test_music.wav" # 测试用的音频路径
    print(f"🎵 正在使用 Wav2Vec2 + Librosa 提取音乐特征: {audio_path}")
    
    audio_feat_np, _ = extract(audio_path) 
    audio_seq_len = audio_feat_np.shape[0]
    
    cond_audio = torch.tensor(audio_feat_np, device=device, dtype=dtype).unsqueeze(0)
    print(f"✅ 音乐特征提取完成！特征张量维度: {cond_audio.shape}")

    # 2. 加载模型
    checkpoint_path = "runs/train/exp15/weights/train-1000.pt" # 👈 请确保这里指向你最新的权重
    
    model = DanceDecoder(
        nfeats=151,            # <--- ✅ 恢复为 AIST++ 标准的 151 维
        seq_len=audio_seq_len, 
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
        print(f"✅ 成功加载权重: {checkpoint_path}")
    else:
        print(f"⚠️ 警告: 找不到权重 {checkpoint_path}，当前使用随机初始化参数做连通性测试。")
        
    model.eval()

    # 3. 构建起止关键帧约束
    ref_motions = torch.zeros(1, audio_seq_len, 151).to(device).to(dtype) # <--- ✅ 151 维
    mask = torch.zeros(1, audio_seq_len, 1).to(device).to(dtype)
    
    print("⚠️ 注意：当前使用全 0 姿态作为 AIST++ 的起止关键帧。")
    mask[:, 0, :] = 1.0 
    mask[:, audio_seq_len - 1, :] = 1.0 

    # 4. 初始化扩散模型
    diffusion = GaussianDiffusion(
        model=model,
        horizon=audio_seq_len, 
        repr_dim=151,          # <--- ✅ 151 维
        smpl=None,             
        n_timestep=1000,       
        predict_epsilon=False  
    ).to(device)
    
    print(f"🚀 开始执行【音乐驱动】扩散生成 ({audio_seq_len} 帧)...")
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            constraint_dict = {"mask": mask, "value": ref_motions}
            output_motion = diffusion.inpaint_loop(
                shape=(1, audio_seq_len, 151), # <--- ✅ 151 维
                cond=cond_audio,
                constraint=constraint_dict
            )

    output_np = output_motion.squeeze(0).cpu().numpy()
    
    os.makedirs("output", exist_ok=True)
    output_npy = "output/aist_music_driven.npy"
    np.save(output_npy, output_np)
    print(f"🎉 听音起舞生成完毕！已保存原始张量至: {output_npy}")

if __name__ == "__main__":
    run_music_driven_inference()