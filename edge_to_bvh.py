import os
import numpy as np
from scipy.spatial.transform import Rotation as R
# 导入你写好的拓扑映射表
from convert_dunhuang_bvh import SMPL_TO_BVH_MAPPING, SMPL_JOINT_ORDER

def rot6d_to_matrix(r6d):
    """
    将 6D 旋转表示还原为 3x3 旋转矩阵 (Gram-Schmidt 正交化)
    r6d shape: (N, 6)
    """
    x_raw = r6d[:, 0:3]
    y_raw = r6d[:, 3:6]
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    z_raw = np.cross(x, y_raw)
    z = z_raw / (np.linalg.norm(z_raw, axis=-1, keepdims=True) + 1e-8)
    y = np.cross(z, x)
    matrices = np.stack([x, y, z], axis=-1)
    return matrices

def tensor_to_bvh(motion_tensor, template_bvh_path, output_bvh_path, fps=30):
    if not os.path.exists(template_bvh_path):
        raise FileNotFoundError(f"找不到模板文件 {template_bvh_path}")

    # === 1. 读取模板 BVH 的头部骨架信息，并记录真实的层级顺序 ===
    bvh_joints = []
    with open(template_bvh_path, 'r') as f:
        lines = f.readlines()
        
    header_lines = []
    for line in lines:
        header_lines.append(line)
        if line.strip().startswith("ROOT") or line.strip().startswith("JOINT"):
            bvh_joints.append(line.strip().split()[1])
        if line.strip().startswith("Frame Time:"):
            break 

    # === 2. 解析张量为欧拉角 ===
    seq_len = motion_tensor.shape[0]
    root_pos = motion_tensor[:, 4:7]          
    q_6d = motion_tensor[:, 7:]               
    
    num_joints = q_6d.shape[1] // 6
    q_6d_flat = q_6d.reshape(seq_len * num_joints, 6)
    rot_matrices = rot6d_to_matrix(q_6d_flat)
    euler_angles_flat = R.from_matrix(rot_matrices).as_euler('yxz', degrees=True)
    euler_angles = euler_angles_flat.reshape(seq_len, num_joints, 3)

    # 🌟 核心修复：建立 BVH 真实层级到 SMPL 索引的反向路由映射
    BVH_TO_SMPL_IDX = {}
    for i, smpl_name in enumerate(SMPL_JOINT_ORDER):
        bvh_name = SMPL_TO_BVH_MAPPING.get(smpl_name)
        if bvh_name:
            BVH_TO_SMPL_IDX[bvh_name] = i

    # === 3. 组装并写入新的 BVH ===
    os.makedirs(os.path.dirname(output_bvh_path), exist_ok=True)
    
    with open(output_bvh_path, 'w') as f:
        for line in header_lines:
            if line.strip().startswith("Frames:"):
                f.write(f"Frames: {seq_len}\n")
            elif line.strip().startswith("Frame Time:"):
                f.write(f"Frame Time: {1.0 / fps:.6f}\n")
            else:
                f.write(line)
                
        for i in range(seq_len):
            frame_data = []
            frame_data.extend(root_pos[i].tolist())
            
            # 🌟 核心修复：严格遵循 BVH Hierarchy 的深度优先顺序写入欧拉角
            for bvh_joint in bvh_joints:
                if bvh_joint in BVH_TO_SMPL_IDX:
                    smpl_idx = BVH_TO_SMPL_IDX[bvh_joint]
                    frame_data.extend(euler_angles[i, smpl_idx].tolist())
                else:
                    # 对于未被 SMPL 涵盖的末端关节 (如手指)，填入零旋转保持静止
                    frame_data.extend([0.0, 0.0, 0.0])
                    
            line_str = " ".join([f"{val:.5f}" for val in frame_data])
            f.write(line_str + "\n")
            
    print(f"🎬 视觉还原成功！动画文件已生成: {output_bvh_path}")