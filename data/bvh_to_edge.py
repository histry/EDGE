import os
import glob
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# 假设你的 BVH 导出的基础帧率为 60 或 30
TARGET_FPS = 30 

def euler_to_6d(euler_angles, order='yxz'):
    """将欧拉角转换为连续的 6D 旋转表示 (适合扩散模型学习)"""
    # euler_angles shape: (seq_len, num_joints, 3)
    seq_len, num_joints, _ = euler_angles.shape
    euler_flat = euler_angles.reshape(-1, 3)
    
    # 转换为旋转矩阵 (严格按照 Chang-E 数据的 yxz 内旋顺序)
    rot_matrices = R.from_euler(order, euler_flat, degrees=True).as_matrix() # (N, 3, 3)
    
    # 提取旋转矩阵的前两列构成 6D 表示 (N, 6)
    rot_6d = np.concatenate([rot_matrices[:, :, 0], rot_matrices[:, :, 1]], axis=-1)
    return rot_6d.reshape(seq_len, num_joints * 6)

def parse_bvh_simple(bvh_path):
    """轻量级 BVH 解析逻辑"""
    with open(bvh_path, 'r') as f:
        lines = f.readlines()

    # 寻找数据块起始点
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("MOTION"):
            data_start = i + 3 # 跳过 MOTION, Frames, Frame Time
            break
            
    # 解析数值数据 (seq_len, num_channels)
    data = []
    for line in lines[data_start:]:
        data.append([float(x) for x in line.strip().split()])
    data = np.array(data)

    num_channels = data.shape[1]
    num_joints = (num_channels - 3) // 3
    
    root_positions = data[:, :3] # (seq_len, 3)
    joint_rotations = data[:, 3:].reshape(-1, num_joints, 3) # (seq_len, J, 3)
    
    return root_positions, joint_rotations

def process_bvh_directory(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    bvh_files = glob.glob(os.path.join(src_dir, "*.bvh"))
    
    for bvh_path in tqdm(bvh_files, desc="Converting BVH to EDGE Format"):
        try:
            # 1. 解析 BVH
            root_pos, joint_euler = parse_bvh_simple(bvh_path)
            
            # 2. 转换为 6D 旋转 (强制使用 yxz 内旋!)
            q_6d = euler_to_6d(joint_euler, order='yxz')
            
            # 3. 构造与 EDGE AIST++ 一致的数据字典
            motion_data = {
                "pos": root_pos, # (seq_len, 3)
                "q": q_6d,       # (seq_len, J * 6)
                "fps": TARGET_FPS,
                "num_joints": joint_euler.shape[1]
            }
            
            # 保存为 pkl
            filename = Path(bvh_path).stem
            save_path = os.path.join(dest_dir, f"{filename}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(motion_data, f)
                
        except Exception as e:
            print(f"Error processing {bvh_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Path to folder containing BVH files")
    parser.add_argument("--dest", type=str, required=True, help="Path to save processed PKL files")
    args = parser.parse_args()
    process_bvh_directory(args.src, args.dest)