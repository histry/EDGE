import os
import glob
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# ⚠️ 1. 骨骼拓扑映射字典 (Joint Mapping) ⚠️
# Key: SMPL 24标准关节名
# Value: 敦煌 BVH 文件中对应的真实关节名
# ==============================================================================
SMPL_TO_BVH_MAPPING = {
    "root": "Hips",                 # 0  - 根节点
    "lhip": "LeftUpLeg",            # 1  - 左大腿根
    "rhip": "RightUpLeg",           # 2  - 右大腿根
    "belly": "Spine",               # 3  - 下腰
    "lknee": "LeftLeg",             # 4  - 左膝
    "rknee": "RightLeg",            # 5  - 右膝
    "spine": "Spine1",              # 6  - 中脊椎
    "lankle": "LeftFoot",           # 7  - 左脚踝
    "rankle": "RightFoot",          # 8  - 右脚踝
    "chest": "Spine2",              # 9  - 胸/上脊椎 (忽略 Spine3 以对齐数量)
    "ltoes": "LeftToeBase",         # 10 - 左脚尖
    "rtoes": "RightToeBase",        # 11 - 右脚尖
    "neck": "Neck",                 # 12 - 脖子
    "linshoulder": "LeftShoulder",  # 13 - 左内肩 (锁骨)
    "rinshoulder": "RightShoulder", # 14 - 右内肩 (锁骨)
    "head": "Head",                 # 15 - 头
    "lshoulder": "LeftArm",         # 16 - 左大臂
    "rshoulder": "RightArm",        # 17 - 右大臂
    "lelbow": "LeftForeArm",        # 18 - 左小臂
    "relbow": "RightForeArm",       # 19 - 右小臂
    "lwrist": "LeftHand",           # 20 - 左手腕
    "rwrist": "RightHand",          # 21 - 右手腕
    "lhand": None,                  # 22 - 左手尖 (丢弃繁琐的手指细节)
    "rhand": None,                  # 23 - 右手尖 (丢弃繁琐的手指细节)
}

# SMPL 24 关节的标准顺序，严格对齐 72 维数据
SMPL_JOINT_ORDER = [
    "root", "lhip", "rhip", "belly", "lknee", "rknee", "spine", "lankle", 
    "rankle", "chest", "ltoes", "rtoes", "neck", "linshoulder", "rinshoulder", 
    "head", "lshoulder", "rshoulder", "lelbow", "relbow", "lwrist", "rwrist", 
    "lhand", "rhand"
]


def load_bvh_data(bvh_path):
    """
    原生纯 Python BVH 解析器。
    自动解析 Hierarchy 层级、通道映射关系以及各关节欧拉角旋转顺序，
    然后从 Motion 块中提取全序列浮点数。
    """
    print(f"🔄 正在深度解析 BVH 文件: {bvh_path}")
    
    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    joint_channels = {}
    joint_rot_orders = {}
    joint_names = []
    
    current_joint = None
    channel_index = 0
    is_motion = False
    motion_lines = []
    seq_len = 0

    # 1. 逐行解析 BVH 文件
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("ROOT") or line.startswith("JOINT"):
            current_joint = line.split()[1]
            joint_names.append(current_joint)
            
        elif line.startswith("CHANNELS"):
            parts = line.split()
            num_channels = int(parts[1])
            channels = parts[2:]
            
            # 记录该关节在 Motion 数据行中的切片起始和结束索引
            joint_channels[current_joint] = (channel_index, channel_index + num_channels)
            channel_index += num_channels

            # 动态获取旋转顺序 (比如 Yrotation Xrotation Zrotation -> 'yxz')
            rot_order = ""
            for ch in channels:
                if "rotation" in ch.lower():
                    rot_order += ch[0].lower()
            if rot_order:
                joint_rot_orders[current_joint] = rot_order

        elif line.startswith("MOTION"):
            is_motion = True
            continue

        if is_motion:
            if line.startswith("Frames:"):
                seq_len = int(line.split()[1])
            elif line.startswith("Frame Time:"):
                pass  # FPS 信息暂时忽略
            else:
                motion_lines.append([float(x) for x in line.split()])

    # 2. 将字符串数据转换为 numpy 矩阵
    data = np.array(motion_lines) # shape: (seq_len, total_channels)
    assert data.shape[0] == seq_len, "BVH 声明的帧数与实际读取行数不符！"

    # 3. 精准提取我们需要的数据
    root_name = joint_names[0]
    root_start, root_end = joint_channels[root_name]
    
    # 假设 Root 的前 3 个通道是 XYZ Position
    bvh_root_pos = data[:, root_start:root_start+3]

    bvh_joint_rotations = {}
    bvh_joint_orders = {}

    for smpl_name, bvh_name in SMPL_TO_BVH_MAPPING.items():
        if bvh_name and bvh_name in joint_channels:
            start, end = joint_channels[bvh_name]
            
            # 如果通道数是 6（通常是 Root，前3是位移，后3是旋转），切出后3个
            if end - start == 6:
                rot_data = data[:, start+3:end]
            else:
                rot_data = data[:, start:end]
                
            bvh_joint_rotations[bvh_name] = rot_data
            bvh_joint_orders[bvh_name] = joint_rot_orders.get(bvh_name, "zxy") # 默认给个 zxy 保底
            
    return bvh_root_pos, bvh_joint_rotations, bvh_joint_orders, seq_len


def process_single_bvh(bvh_file, out_path):
    bvh_root_pos, bvh_joint_rotations, bvh_joint_orders, seq_len = load_bvh_data(bvh_file)
    
    # ==============================================================================
    # 🎯 核心一：固化坐标系转换 (Bake Coordinate Alignment)
    # 把之前在 dataset.py 里的黑魔法挪到这里！绕 X 轴旋转 90 度，使小人完美站立。
    # ==============================================================================
    align_rot = R.from_euler('x', 90, degrees=True)
    
    # 1. 旋转全局位移 (Root Translation)
    # ⚠️ 极其关键：从你给的 BVH offset 来看 (如 43.68)，数据单位大概率是【厘米(cm)】
    # 而 AIST++ 的模型单位是【米(m)】！这里强制乘以 0.01 进行等比缩小。
    bvh_root_pos = bvh_root_pos * 0.01
    aligned_pos = align_rot.apply(bvh_root_pos)
    
    # ==============================================================================
    # 🎯 核心二：重定向与 72 维 Axis-Angle 组装
    # ==============================================================================
    q_72 = np.zeros((seq_len, 24 * 3)) # (seq_len, 72)
    
    for i, smpl_joint in enumerate(SMPL_JOINT_ORDER):
        bvh_joint = SMPL_TO_BVH_MAPPING[smpl_joint]
        
        if bvh_joint is None or bvh_joint not in bvh_joint_rotations:
            continue # 没有对应关节则保持静止 0
            
        euler_angles = bvh_joint_rotations[bvh_joint]
        order = bvh_joint_orders[bvh_joint] # 自动获取，比如 'yxz'
        
        # 将原生欧拉角转为 Scipy Rotation
        joint_rot = R.from_euler(order, euler_angles, degrees=True) 
        
        # 2. 根节点的旋转也要施加全局坐标系对齐 (站立校正)
        if smpl_joint == "root":
            joint_rot = align_rot * joint_rot
            
        # 转换为 SMPL 需要的 Axis-Angle (旋转向量)
        axis_angle = joint_rot.as_rotvec()
        
        q_72[:, i*3 : (i+1)*3] = axis_angle

    # ==============================================================================
    # 🎯 核心三：保存为 AIST++ 兼容的 381 维等效 .pkl 格式
    # ==============================================================================
    out_data = {
        "pos": aligned_pos.astype(np.float32),
        "q": q_72.astype(np.float32)
    }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(out_data, f)
        
    print(f"✅ 转换成功！已保存为: {out_path}")
    print(f"   维度检查 -> pos: {out_data['pos'].shape}, q: {out_data['q'].shape}")


if __name__ == "__main__":
    # 你需要把存放敦煌原始 bvh 文件的目录放在这里
    raw_bvh_dir = "data/dunhuang_raw_bvh/"
    processed_dir = "data/dunhuang/motions_sliced/"
    
    os.makedirs(processed_dir, exist_ok=True)
    bvh_files = glob.glob(os.path.join(raw_bvh_dir, "*.bvh"))
    
    if not bvh_files:
        print(f"⚠️ 警告: 在 {raw_bvh_dir} 下没有找到任何 .bvh 文件，请检查路径。")
    
    for bvh_file in bvh_files:
        filename = os.path.basename(bvh_file).replace(".bvh", ".pkl")
        out_file = os.path.join(processed_dir, filename)
        process_single_bvh(bvh_file, out_file)