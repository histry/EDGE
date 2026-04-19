import os
import glob
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

# ==============================================================================
# ⚠️ 1. 骨骼拓扑映射字典 (Joint Mapping) ⚠️
# 这里的 Key 是 SMPL 的 24 个标准关节名字，绝对不能改！
# 这里的 Value 请填入你【敦煌 BVH 文件】中对应的真实关节名字。
# 如果敦煌数据没有某个关节（比如脚趾），填入 None，代码会自动用 0 填充。
# ==============================================================================
SMPL_TO_BVH_MAPPING = {
    "root": "Hips",           # 0  - 根节点 (必须有)
    "lhip": "LeftUpLeg",      # 1
    "rhip": "RightUpLeg",     # 2
    "belly": "Spine",         # 3
    "lknee": "LeftLeg",       # 4
    "rknee": "RightLeg",      # 5
    "spine": "Spine1",        # 6
    "lankle": "LeftFoot",     # 7
    "rankle": "RightFoot",    # 8
    "chest": "Spine2",        # 9
    "ltoes": "LeftToeBase",   # 10 (如果没有可填 None)
    "rtoes": "RightToeBase",  # 11 (如果没有可填 None)
    "neck": "Neck",           # 12
    "linshoulder": "LeftShoulder",  # 13 (内肩/锁骨)
    "rinshoulder": "RightShoulder", # 14
    "head": "Head",           # 15
    "lshoulder": "LeftArm",   # 16 (外肩/大臂)
    "rshoulder": "RightArm",  # 17
    "lelbow": "LeftForeArm",  # 18
    "relbow": "RightForeArm", # 19
    "lwrist": "LeftHand",     # 20
    "rwrist": "RightHand",    # 21
    "lhand": None,            # 22 (通常丢弃手指细节)
    "rhand": None,            # 23
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
    这是一个极简的 BVH 旋转和位移提取器。
    在实际科研中，建议使用 `bvh` 库或 `pymo` 库来更精确地解析层级。
    这里假设你已经能提取出 BVH 每一帧的 Euler 角和 Root 位移。
    """
    print(f"🔄 正在读取 BVH 文件: {bvh_path}")
    
    # ---------------------------------------------------------
    # ⚠️ 2. 请在这里接入你的 BVH 读取逻辑 ⚠️
    # 你需要返回两个 numpy 数组：
    # bvh_root_pos: shape (seq_len, 3) 每一帧 Root 的全局坐标
    # bvh_joint_rotations: dict, Key是BVH关节名, Value是 shape (seq_len, 3) 的欧拉角(Euler angles)
    # ---------------------------------------------------------
    
    # 以下为占位符假数据，请用你的真实解析代码替换！
    seq_len = 150 
    bvh_root_pos = np.zeros((seq_len, 3))
    bvh_joint_rotations = {name: np.zeros((seq_len, 3)) for name in SMPL_TO_BVH_MAPPING.values() if name}
    
    # 提示：用 scipy 读欧拉角时要注意旋转顺序（如 'ZXY', 'XYZ' 等，取决于你的动捕软件）
    
    return bvh_root_pos, bvh_joint_rotations, seq_len


def process_single_bvh(bvh_file, out_path):
    bvh_root_pos, bvh_joint_rotations, seq_len = load_bvh_data(bvh_file)
    
    # ==============================================================================
    # 🎯 核心一：固化坐标系转换 (Bake Coordinate Alignment)
    # 把之前在 dataset.py 里的黑魔法挪到这里！绕 X 轴旋转 90 度。
    # 这样进入模型的数据，天生就是符合 AIST++ 物理规律的 Y-Up 数据。
    # ==============================================================================
    align_rot = R.from_euler('x', 90, degrees=True)
    
    # 1. 旋转全局位移 (Root Translation)
    # AIST++ 的缩放比例通常是米(m)，如果你的 BVH 是厘米(cm)，请在这里除以 100！
    bvh_root_pos = bvh_root_pos * 1.0  # ⚠️ 如果需要厘米转米，改成 * 0.01
    aligned_pos = align_rot.apply(bvh_root_pos)
    
    # ==============================================================================
    # 🎯 核心二：重定向与 72 维 Axis-Angle 组装
    # ==============================================================================
    q_72 = np.zeros((seq_len, 24 * 3)) # (seq_len, 72)
    
    for i, smpl_joint in enumerate(SMPL_JOINT_ORDER):
        bvh_joint = SMPL_TO_BVH_MAPPING[smpl_joint]
        
        if bvh_joint is None or bvh_joint not in bvh_joint_rotations:
            # 如果没有这个关节，保持为 0 (静止状态)
            continue
            
        # 获取该关节在这段序列里所有的 Euler 角 (假设 BVH 默认是 ZXY 顺序)
        euler_angles = bvh_joint_rotations[bvh_joint]
        
        # 转换为 scipy 的 Rotation 对象
        # ⚠️ 注意：这里的 'zxy' 或 'xyz' 必须与你导出 BVH 时的设置严格一致！
        joint_rot = R.from_euler('zxy', euler_angles, degrees=True) 
        
        # 2. 对 Root 节点的旋转也要施加全局坐标系对齐！
        if smpl_joint == "root":
            joint_rot = align_rot * joint_rot
            
        # 将旋转转换为 SMPL 标准的 Axis-Angle (轴角) 表示格式 (也就是 scipy 的 rotvec)
        axis_angle = joint_rot.as_rotvec()
        
        # 填入对应的 72 维切片中
        q_72[:, i*3 : (i+1)*3] = axis_angle

    # ==============================================================================
    # 🎯 核心三：保存为 AIST++ 标准 .pkl 格式
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
    # 批量处理你的敦煌数据
    raw_bvh_dir = "data/dunhuang_raw_bvh/"
    processed_dir = "data/dunhuang/motions_sliced/"
    
    bvh_files = glob.glob(os.path.join(raw_bvh_dir, "*.bvh"))
    if not bvh_files:
        print("⚠️ 没有找到 BVH 文件，请检查路径。")
    
    for bvh_file in bvh_files:
        filename = os.path.basename(bvh_file).replace(".bvh", ".pkl")
        out_file = os.path.join(processed_dir, filename)
        process_single_bvh(bvh_file, out_file)