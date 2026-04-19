import numpy as np
from data.bvh_to_edge import parse_bvh_simple, euler_to_6d
from edge_to_bvh import tensor_to_bvh

print("正在测试底层 3D 转换管线 (BVH -> 张量 -> BVH)...")

# 1. 读取原动作并转为张量
bvh_path = "data/test_dunhuang_sample.bvh"
root_pos, joint_euler = parse_bvh_simple(bvh_path)
q_6d = euler_to_6d(joint_euler, order='yxz')
real_motion_np = np.concatenate([root_pos, q_6d], axis=-1)

# 2. 直接将张量反转回 BVH (完全不经过任何 AI 和神经网络)
output_bvh = "output/test_math_roundtrip.bvh"
tensor_to_bvh(real_motion_np, bvh_path, output_bvh, fps=30)

print(f"✅ 纯数学测试完成！已导出: {output_bvh}")