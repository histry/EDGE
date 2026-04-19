import pickle
import torch
import os
import numpy as np
from vis import SMPLSkeleton, skeleton_render

# ==========================================
# 1. 读取你刚刚洗好的敦煌 PKL 数据
# ==========================================
# ⚠️ 请把这里替换为你刚刚生成的敦煌 pkl 文件的路径
pkl_path = "data/dunhuang/motions_sliced/test_dunhuang.pkl" 

print(f"🔄 正在读取转换后的敦煌数据: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

pos = torch.tensor(data["pos"]).float() # (Seq_len, 3) 全局位移
q = torch.tensor(data["q"]).float()     # (Seq_len, 72) Axis-Angle 旋转

# ==========================================
# 2. 将 72 维数据重塑为 SMPL 需要的 (Seq_len, 24, 3) 格式
# ==========================================
q_reshaped = q.reshape(q.shape[0], 24, 3)

# ==========================================
# 3. 通过 SMPL 运动学模型算出 3D 骨架
# ==========================================
print("🦴 正在通过 SMPL 计算 3D 骨架坐标...")
smpl = SMPLSkeleton()
poses_3d = smpl.forward(q_reshaped.unsqueeze(0), pos.unsqueeze(0)).squeeze(0).numpy()

# ==========================================
# 4. 调用最新版 vis.py 渲染出图！
# ==========================================
out_dir = "runs/verify_dunhuang"
os.makedirs(out_dir, exist_ok=True)
print(f"🎬 正在渲染视频，请稍候...")

skeleton_render(
    poses_3d, 
    epoch="dunhuang_test", 
    out=out_dir, 
    name="test_dh.mp4", 
    sound=False
)

print(f"✅ 渲染完成！快去 {out_dir} 看看敦煌的飞天是不是完美站立了！")