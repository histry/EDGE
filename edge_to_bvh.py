import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def rot6d_to_matrix(r6d):
    """
    将 6D 旋转表示还原为 3x3 旋转矩阵 (Gram-Schmidt 正交化)
    r6d shape: (N, 6)
    """
    x_raw = r6d[:, 0:3]
    y_raw = r6d[:, 3:6]
    
    # 归一化 x
    x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + 1e-8)
    
    # 计算 z = x cross y
    z_raw = np.cross(x, y_raw)
    z = z_raw / (np.linalg.norm(z_raw, axis=-1, keepdims=True) + 1e-8)
    
    # 计算精确的 y = z cross x
    y = np.cross(z, x)
    
    # 将 x, y, z 作为列向量堆叠成旋转矩阵 (N, 3, 3)
    matrices = np.stack([x, y, z], axis=-1)
    return matrices

def tensor_to_bvh(motion_tensor, template_bvh_path, output_bvh_path, fps=30):
    """
    核心反向导出函数：
    1. 从模板 BVH 复制骨骼层级 (Hierarchy)
    2. 将 381 维张量 (3位移 + 63关节*6D) 还原为欧拉角
    3. 写入新的 BVH 文件
    """
    if not os.path.exists(template_bvh_path):
        raise FileNotFoundError(f"找不到模板文件 {template_bvh_path}，必须提供一个真实 BVH 来获取骨架层级树！")

    # === 1. 读取模板 BVH 的头部骨架信息 ===
    with open(template_bvh_path, 'r') as f:
        lines = f.readlines()
        
    header_lines = []
    for line in lines:
        header_lines.append(line)
        if line.strip().startswith("Frame Time:"):
            break # 截取到此为止，下方就是原本的动作数据了

    # === 2. 解析张量 ===
    # motion_tensor shape: (seq_len, 381)
    seq_len = motion_tensor.shape[0]
    
    root_pos = motion_tensor[:, 0:3]          # 前 3 维是根节点 (Root) 的全局 XYZ 位移
    q_6d = motion_tensor[:, 3:]               # 后面 378 维是 63 个关节的 6D 旋转
    
    num_joints = q_6d.shape[1] // 6
    assert num_joints == 63, f"预期有 63 个关节，但解析出 {num_joints} 个！请检查张量维度。"

    # 将 (seq_len, 378) reshape 为 (seq_len * 63, 6) 以便批量计算
    q_6d_flat = q_6d.reshape(seq_len * num_joints, 6)
    
    # 6D -> 旋转矩阵
    rot_matrices = rot6d_to_matrix(q_6d_flat)
    
    # 旋转矩阵 -> 欧拉角 (严格使用 yxz 小写内旋)
    euler_angles_flat = R.from_matrix(rot_matrices).as_euler('yxz', degrees=True)
    
    # 恢复形状 (seq_len, num_joints, 3)
    euler_angles = euler_angles_flat.reshape(seq_len, num_joints, 3)

    # === 3. 组装并写入新的 BVH ===
    os.makedirs(os.path.dirname(output_bvh_path), exist_ok=True)
    
    with open(output_bvh_path, 'w') as f:
        # 写入复制来的骨架头部
        for line in header_lines:
            # 顺手把帧数修改为我们要生成的长度
            if line.strip().startswith("Frames:"):
                f.write(f"Frames: {seq_len}\n")
            elif line.strip().startswith("Frame Time:"):
                f.write(f"Frame Time: {1.0 / fps:.6f}\n")
            else:
                f.write(line)
                
        # 逐帧写入动作数据
        for i in range(seq_len):
            frame_data = []
            
            # Root XYZ 位移
            frame_data.extend(root_pos[i].tolist())
            
            # 所有关节的 Euler 角度
            for j in range(num_joints):
                frame_data.extend(euler_angles[i, j].tolist())
                
            # 用空格拼接并换行
            line_str = " ".join([f"{val:.5f}" for val in frame_data])
            f.write(line_str + "\n")
            
    print(f"🎬 视觉还原成功！动画文件已生成: {output_bvh_path}")