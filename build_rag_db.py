import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from data.audio_extraction.wav2vec_librosa_features import extract
from model.mmr_model import CrossModalMMR
import pickle
from dataset.quaternion import ax_to_6v
from dataset.preprocess import vectorize_many
from vis import SMPLSkeleton

def build_automated_rag_database():
    output_dir = "data/dunhuang_rag_db"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("🚀 初始化 MMR 跨模态编码器...")
    mmr_model = CrossModalMMR(motion_dim=151, audio_dim=803, latent_dim=256).to(device)
    # 确保权重路径与你的实际路径匹配
    pretrained_path = "weights/mmr_pretrained.pt"
    if os.path.exists(pretrained_path):
        mmr_model.load_state_dict(torch.load(pretrained_path, map_location=device))
    else:
        print(f"⚠️ 找不到 {pretrained_path}，请确保已提供预训练模型！")
    mmr_model.eval()

    # 1. 预提取所有可用候选 proxy music 的隐空间特征。
    # 这些音乐只作为候选弱条件，不表示真实的敦煌音乐-动作配对标签。
    audio_bank_paths = glob.glob("proxy_music/*.wav")
    audio_bank_features = []
    print(f"🎵 正在建立音乐特征池 (共 {len(audio_bank_paths)} 首)...")
    for audio_path in audio_bank_paths:
        feat_np, _ = extract(audio_path)
        feat_t = torch.tensor(feat_np).float().unsqueeze(0).to(device)
        with torch.no_grad():
            latent = mmr_model.encode_audio(feat_t).squeeze(0) # [256]
        audio_bank_features.append({"path": audio_path, "latent": latent, "raw_feat": feat_np})

    if not audio_bank_features:
        print("⚠️ 警告：未找到任何代理音乐，请检查 proxy_music/ 目录。")
        return

    # 2. 遍历所有处理好的动作数据进行候选检索。
    # 这里得到的是 proxy candidate，不作为真实监督配对 claim。
    motion_paths = glob.glob("data/dunhuang_bvh/processed/*.pkl")
    print(f"🕺 开始对 {len(motion_paths)} 个动作切片进行 proxy music 候选检索...")
    
    smpl = SMPLSkeleton()
    
    for m_path in motion_paths:
        data = pickle.load(open(m_path, "rb"))
        pos_t = torch.Tensor(data["pos"]).unsqueeze(0) 
        q_t = torch.Tensor(data["q"]).unsqueeze(0)     
        
        bs, sq, c = q_t.shape
        q_t_reshaped = q_t.reshape((bs, sq, -1, 3))
        
        # 计算 4 维脚部物理接触点
        positions = smpl.forward(q_t_reshaped, pos_t)
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(q_t_reshaped)

        # 将欧拉角转化为 144 维 6D 旋转，拼凑成 151 维张量
        q_6v = ax_to_6v(q_t_reshaped)
        l = [contacts, pos_t, q_6v]
        motion_np = vectorize_many(l).float().detach().squeeze(0).numpy()
        
        motion_t = torch.tensor(motion_np).float().unsqueeze(0).to(device)
        
        # 提取动作的隐空间语义
        with torch.no_grad():
            motion_latent = mmr_model.encode_motion(motion_t).squeeze(0) # [256]
            
        # 3. 计算余弦相似度，寻找 Top-1 proxy music 候选
        best_sim = -1.0
        best_audio = None
        for audio_item in audio_bank_features:
            sim = F.cosine_similarity(motion_latent.unsqueeze(0), audio_item["latent"].unsqueeze(0)).item()
            if sim > best_sim:
                best_sim = sim
                best_audio = audio_item
                
        # 设定阈值，确保是真正气质相符的片段
        if best_sim > 0.85:  
            print(f"   ✅ [候选命中] 动作: {os.path.basename(m_path)} <==> proxy music: {os.path.basename(best_audio['path'])} (相似度: {best_sim:.3f})")
            
            audio_feat_np = best_audio["raw_feat"]
            seq_len = audio_feat_np.shape[0]
            
            # 对齐长度
            motion_save = motion_np.copy()
            if motion_save.shape[0] > seq_len:
                motion_save = motion_save[:seq_len]
            elif motion_save.shape[0] < seq_len:
                pad_len = seq_len - motion_save.shape[0]
                motion_save = np.pad(motion_save, ((0, pad_len), (0, 0)), mode='edge')
            
            save_path = os.path.join(output_dir, f"auto_rag_{os.path.basename(m_path).split('.')[0]}.npy")
            np.save(save_path, {"audio_feat": audio_feat_np, "motion": motion_save})

    print(f"\n🎉 RAG 候选库构建完成！现已存储于: {output_dir}")

if __name__ == "__main__":
    build_automated_rag_database()
