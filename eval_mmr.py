import os
import glob
import torch
import torch.nn.functional as F
import numpy as np

# ✨ 修复：新增引入骨架实例、四元数运算与平滑混合工具
from vis import SMPLSkeleton
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d
from dataset.quaternion import quat_slerp, ax_from_6v

class DunhuangRAGSystem:
    def __init__(self, db_dir="data/dunhuang_rag_db"):
        self.db_dir = db_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 【核心修复 3】：加载对比学习 MMR 编码器，打破维度诅咒
        self.mmr_model = CrossModalMMR(motion_dim=151, audio_dim=803, latent_dim=256).to(self.device)
        mmr_ckpt_path = "weights/mmr_pretrained.pt"
        if os.path.exists(mmr_ckpt_path):
            self.mmr_model.load_state_dict(torch.load(mmr_ckpt_path, map_location=self.device))
            print("✅ RAG 引擎: MMR 降维投影器加载成功。")
        self.mmr_model.eval()
        
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
        
    def _get_latent_semantics(self, audio_feat):
        """利用 MMR 模型将 803 维时序特征压缩至 256 维高语义隐空间"""
        feat_t = audio_feat.to(self.device).unsqueeze(0) # (1, seq_len, 803)
        with torch.no_grad():
            latent = self.mmr_model.encode_audio(feat_t) # (1, 256)
        return latent.squeeze(0)

    def retrieve_prior(self, test_audio_feat, top_k=3):
        if not self.db_records:
            return None
        
        scores = []
        # 将测试音频映射到 256 维紧凑流形
        test_latent = self._get_latent_semantics(test_audio_feat)
        
        for record in self.db_records:
            # 将先验库音频映射到同一隐空间
            db_latent = self._get_latent_semantics(record['audio_feat'])
            # 在低维空间计算语义相似度，显著提升判别力
            sim = F.cosine_similarity(test_latent.unsqueeze(0), db_latent.unsqueeze(0)).item()
            scores.append((sim, record['motion']))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_motion = scores[0] 
        print(f"🔍 [RAG 检索] 命中最高语义匹配，隐空间余弦相似度: {best_score:.4f}")
        return best_motion

# ✨ 修复：补充切片函数，安全地将变长音频切割为模型熟悉的 150 帧定长
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

def run_music_driven_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    audio_path = "custom_music/test_music.wav"
    print(f"🎵 正在使用 Wav2Vec2 + Librosa 提取音乐特征: {audio_path}")
    
    audio_feat_np, _ = extract(audio_path) 
    audio_seq_len = audio_feat_np.shape[0]
    
    cond_audio = torch.tensor(audio_feat_np, device=device, dtype=dtype).unsqueeze(0)
    print(f"✅ 音乐特征提取完成！特征张量维度: {cond_audio.shape}")

    checkpoint_path = "runs/train/exp16/weights/train-300.pt"
    
    # === 原代码 DanceDecoder 实例化及之后的逻辑全部替换 ===
    
    horizon = 150   # 锁定模型感受野
    stride = horizon // 2

    # ✨ 修复：强制模型处于 150 帧序列长度
    model = DanceDecoder(
        nfeats=151,            
        seq_len=horizon, 
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
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('ema_state_dict'))
        
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
            
        if 'null_cond_embed' in state_dict:
            saved_shape = state_dict['null_cond_embed'].shape 
            target_shape = model.null_cond_embed.shape        
            if saved_shape != target_shape:
                saved_embed = state_dict['null_cond_embed'].permute(0, 2, 1) 
                resized_embed = F.interpolate(saved_embed, size=target_shape[1], mode='linear', align_corners=False)
                state_dict['null_cond_embed'] = resized_embed.permute(0, 2, 1)
        
        model.load_state_dict(state_dict)
        print(f"✅ 成功加载权重及 Normalizer: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"⚠️ 找不到权重文件 {checkpoint_path}")
        
    model.eval()

    rag_system = DunhuangRAGSystem(db_dir="data/dunhuang_rag_db")
    retrieved_motion = rag_system.retrieve_prior(cond_audio[0])

    mask = torch.zeros(1, audio_seq_len, 1).to(device).to(dtype)
    value = torch.zeros(1, audio_seq_len, 151).to(device).to(dtype)
    
    if retrieved_motion is not None and retrieved_motion.shape[0] >= 15:
        print("✨ 激活 RAG 增强：提取 15 帧 (0.5秒) 建立物理动量先验，消除边界抖动。")
        retrieved_motion_norm = normalizer.normalize(retrieved_motion.unsqueeze(0)).squeeze(0)
        mask[:, 0:15, :] = 1.0  
        value[:, 0:15, :] = retrieved_motion_norm[0:15].to(device).to(dtype)
        mask[:, -15:, :] = 1.0  
        retrieved_last_idx = min(audio_seq_len, retrieved_motion_norm.shape[0])
        value[:, -15:, :] = retrieved_motion_norm[retrieved_last_idx-15:retrieved_last_idx].to(device).to(dtype)
    else:
        print("⚠️ 提示：未检索到有效 RAG 动作先验，回退到首尾 1 帧的静止先验。")
        mask[:, 0, :] = 1.0 
        mask[:, -1, :] = 1.0 
        zero_prior = normalizer.normalize(torch.zeros(1, 1, 151)).squeeze(0)
        value[:, 0, :] = zero_prior.to(device).to(dtype)
        value[:, -1, :] = zero_prior.to(device).to(dtype)

    # ✨ 修复：进入切块流程
    N_cond_audio = chunk_tensor(cond_audio, horizon, stride, audio_seq_len)
    N_mask = chunk_tensor(mask, horizon, stride, audio_seq_len, is_mask=True)
    N_value = chunk_tensor(value, horizon, stride, audio_seq_len)
    
    N = N_cond_audio.shape[0]
    constraint_dict = {"mask": N_mask, "value": N_value}

    active_smpl = SMPLSkeleton(device=device)

    diffusion = GaussianDiffusion(
        model=model,
        horizon=horizon, 
        repr_dim=151,          
        smpl=active_smpl,             
        n_timestep=1000,       
        predict_epsilon=False  
    ).to(device)
    diffusion.normalizer = normalizer
    
    print(f"🚀 开始执行【检索增强】长序列分块扩散生成 (共切割为 {N} 块)...")
    
    all_outputs = []
    overlap_frames = 15
    
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            for i in range(N):
                sub_cond_dict = {"audio": N_cond_audio[i:i+1]}
                sub_constraint_mask = constraint_dict["mask"][i:i+1].clone()
                sub_constraint_value = constraint_dict["value"][i:i+1].clone()
                
                # 滑动窗口：提取上一块末尾重叠部分的实际生成结果，锁死为当前块的开局
                if i > 0 and len(all_outputs) > 0:
                    prev_last_chunk = all_outputs[-1][-1:].to(device).clone()
                    sub_constraint_mask[0, :overlap_frames, :] = 1.0 
                    sub_constraint_value[0, :overlap_frames, :] = prev_last_chunk[0, stride:stride+overlap_frames, :]
                
                output_motion = diffusion.long_inpaint_loop(
                    shape=(1, horizon, 151), 
                    cond=sub_cond_dict,
                    constraint={"mask": sub_constraint_mask, "value": sub_constraint_value},
                    use_tto=False
                )
                all_outputs.append(output_motion.detach().cpu())

    # 聚合并反归一化回物理空间
    output_motion = torch.cat(all_outputs, dim=0)
    output_motion_physical = normalizer.unnormalize(output_motion)
    
    # ✨ 修复：分块间的物理坐标混合修正 (Slerp Blending)
    current_motion = output_motion_physical[0].clone()
    for i in range(1, N):
        next_chunk = output_motion_physical[i].clone()
        
        # 1. 根节点位移硬对齐
        delta_x = current_motion[i * stride, 4] - next_chunk[0, 4]
        delta_z = current_motion[i * stride, 6] - next_chunk[0, 6]
        next_chunk[:, 4] += delta_x
        next_chunk[:, 6] += delta_z

        # 2. 构造融合过渡的平滑权重 (五次多项式)
        fade_w = torch.linspace(0, 1, stride, device=next_chunk.device).unsqueeze(1)
        fade_w_smooth = 6 * (fade_w ** 5) - 15 * (fade_w ** 4) + 10 * (fade_w ** 3)
        
        # 3. 平滑混合位置坐标
        next_chunk[:stride, 4:7] = current_motion[i * stride : i * stride + stride, 4:7] * (1 - fade_w_smooth) + next_chunk[:stride, 4:7] * fade_w_smooth
        
        # 4. Slerp 平滑混合骨骼旋转四元数
        q_curr = axis_angle_to_quaternion(ax_from_6v(current_motion[i * stride : i * stride + stride, 7:].reshape(stride, 24, 6)))
        q_next = axis_angle_to_quaternion(ax_from_6v(next_chunk[:stride, 7:].reshape(stride, 24, 6)))
        q_blended = quat_slerp(q_curr, q_next, fade_w.squeeze(1)) 
        next_chunk[:stride, 7:] = matrix_to_rotation_6d(quaternion_to_matrix(q_blended)).reshape(stride, 144)

        current_motion = torch.cat([current_motion[:i * stride], next_chunk], dim=0)

    # 裁减到精准音频帧数
    output_np = current_motion[:audio_seq_len].cpu().numpy()
    
    os.makedirs("output", exist_ok=True)
    output_npy = "output/aist_music_driven.npy"
    np.save(output_npy, output_np)
    print(f"🎉 RAG 节拍锚点推理完成！已保存反归一化的物理张量至: {output_npy}")

if __name__ == "__main__":
    run_music_driven_inference()
