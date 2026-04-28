import argparse
import glob
import os
import pickle
import random  # 补充缺失的模块

import numpy as np
from tqdm import tqdm


def calc_physical_score(dir):
    scores = []
    names = []
    accelerations = []
    
    # ==========================================
    # 🌟 核心修复：SMPL 骨架的向上轴是 Y 轴 (1)
    # ==========================================
    up_dir = 1  # y is up (原先错误写成了 2)
    flat_dirs = [i for i in range(3) if i != up_dir]
    DT = 1 / 30

    it = glob.glob(os.path.join(dir, "*.pkl"))
    if len(it) > 1000:
        it = random.sample(it, 1000)
        
    for pkl in tqdm(it):
        info = pickle.load(open(pkl, "rb"))
        joint3d = info["full_pose"]
        
        root_v = (joint3d[1:, 0, :] - joint3d[:-1, 0, :]) / DT  # root velocity (S-1, 3)
        root_a = (root_v[1:] - root_v[:-1]) / DT  # (S-2, 3) root accelerations
        
        # clamp the up-direction of root acceleration
        root_a[:, up_dir] = np.maximum(root_a[:, up_dir], 0)  # (S-2, 3)
        # l2 norm
        root_a = np.linalg.norm(root_a, axis=-1)  # (S-2,)

        foot_idx = [7, 10, 8, 11]
        feet = joint3d[:, foot_idx]  # foot positions (S, 4, 3)
        
        # 🌟 修复：物理公式必须除以时间常数 DT，否则求出的是单帧位移而不是速度
        foot_v = np.linalg.norm(
            feet[2:, :, flat_dirs] - feet[1:-1, :, flat_dirs], axis=-1
        ) / DT  # (S-2, 4) horizontal velocity
        
        foot_mins = np.zeros((len(foot_v), 2))
        foot_mins[:, 0] = np.minimum(foot_v[:, 0], foot_v[:, 1]) # 左脚 (踝, 尖) 取最小滑步
        foot_mins[:, 1] = np.minimum(foot_v[:, 2], foot_v[:, 3]) # 右脚 (踝, 尖) 取最小滑步

        foot_loss = (
            foot_mins[:, 0] * foot_mins[:, 1] * root_a
        )  # min leftv * min rightv * root_a (S-2,)
        
        foot_loss = foot_loss.mean()
        scores.append(foot_loss)
        names.append(pkl)
        accelerations.append(foot_mins[:, 0].mean())

    out = np.mean(scores) * 10000
    print(f"📊 {dir} 目录下的平均 PFC 物理滑步得分为: {out:.4f} (越低越好)")


def parse_eval_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_path",
        type=str,
        default="renders/",
        help="Where to load saved motions",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_eval_opt()
    calc_physical_score(opt.motion_path)