import torch
import os
from dataset.dance_dataset import AISTPPDataset
from dataset.quaternion import ax_from_6v
from vis import SMPLSkeleton, skeleton_render

# ==========================================
# 1. 加载最新的纯净数据集 (它会触发生成新的 .pkl)
# ==========================================
print("🔄 正在加载并处理纯净的 AIST++ 数据...")
# 注意：这里的 data_path 请替换为你实际存放 AIST++ 数据的根目录（通常是 "data/"）
dataset = AISTPPDataset(data_path="data/", train=True, seq_len=150) 

# ==========================================
# 2. 抽取第 1 条数据的 Ground Truth (真实标签)
# ==========================================
pose_norm, _, filename, _ = dataset[0]
print(f"📦 成功抽取测试数据，源文件: {filename}")

# ==========================================
# 3. 反归一化还原真实物理尺度
# ==========================================
# pose_norm 形状是 (150, 151)，增加 batch 维度给 normalizer
pose = dataset.normalizer.unnormalize(pose_norm.unsqueeze(0)).squeeze(0)

# ==========================================
# 4. 拆解这 151 维“天书”
# ==========================================
contact = pose[:, :4]           # 前 4 维是脚部接触点
pos = pose[:, 4:7]              # 第 4~7 维是 Root 根节点位移
q_6v = pose[:, 7:].reshape(-1, 24, 6) # 后 144 维是 6D 旋转矩阵
q = ax_from_6v(q_6v)            # 转回轴角 (Axis-Angle) 表示

# ==========================================
# 5. 用 SMPL 运动学模型算出 3D 坐标
# ==========================================
print("🦴 正在通过 SMPL 计算 3D 骨架坐标...")
smpl = SMPLSkeleton()
poses_3d = smpl.forward(q.unsqueeze(0), pos.unsqueeze(0)).squeeze(0).numpy()

# 因为是验证真实数据，我们需要把我们之前针对模型乱跑加的“锁死代码”逻辑去掉
# 真实数据是不需要 [..., 4:7] = 0.0 来锁死的，它本来就带有正确的位移
# 但是！我们依然需要应用我们最新版 vis.py 里的视角和旋转逻辑！

# ==========================================
# 6. 调用最新版的 vis.py 渲染出图！
# ==========================================
out_dir = "runs/verify_data"
os.makedirs(out_dir, exist_ok=True)
print(f"🎬 正在渲染视频，请稍候...")

skeleton_render(
    poses_3d, 
    epoch="pure_gt", 
    out=out_dir, 
    name="test_gt.mp4", 
    sound=False # 我们只看动作，不听声音
)

print(f"✅ 渲染完成！快去 {out_dir} 看看你的小人是不是恢复正常了！")