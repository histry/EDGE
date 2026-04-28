import os
import argparse
import numpy as np
import torch
from vis import skeleton_render, SMPLSkeleton 
from dataset.quaternion import ax_from_6v

def main():
    parser = argparse.ArgumentParser(description="将生成的 npy 动作与音乐合成为视频")
    parser.add_argument("--motion", type=str, required=True, help="输入的 .npy 文件路径")
    parser.add_argument("--audio", type=str, required=True, help="输入的测试音乐 .wav 路径")
    parser.add_argument("--output", type=str, required=True, help="输出的 .mp4 文件路径")
    parser.add_argument(
        "--camera_mode",
        type=str,
        choices=["fixed", "follow"],
        default="fixed",
        help="fixed 用固定世界坐标观察轨迹；follow 用骨盆跟拍观察动作细节",
    )
    args = parser.parse_args()

    if not os.path.exists(args.motion):
        raise FileNotFoundError(f"❌ 找不到动作文件: {args.motion}")
    if not os.path.exists(args.audio):
        raise FileNotFoundError(f"❌ 找不到音乐文件: {args.audio}")

    print(f"🎬 正在读取动作张量: {args.motion}")
    motion_data = np.load(args.motion)
    
    # ==================== ✨ 修复核心：正向运动学 (FK) 还原 3D 关节坐标 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    motion_tensor = torch.tensor(motion_data, dtype=torch.float32, device=device).unsqueeze(0)
    
    # 拆分特征
    contacts = motion_tensor[0, :, 0:4].cpu().numpy()
    pos = motion_tensor[:, :, 4:7]
    seq_len = pos.shape[1]
    q_6d = motion_tensor[:, :, 7:].reshape(1, seq_len, 24, 6)
    
    # 6D 转 Axis-Angle
    q_ax = ax_from_6v(q_6d)
    
    # SMPL 前向推导 3D 坐标
    smpl = SMPLSkeleton(device=device)
    poses_3d = smpl.forward(q_ax, pos).detach().cpu().numpy()[0]
    # =========================================================================================
    
    # 提取输出目录
    out_dir = os.path.dirname(args.output)
    if out_dir == "": out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    
    # 为了适配 vis.py 的命名习惯，构造 epoch 前缀
    base_name = os.path.basename(args.output).replace(".mp4", "")
    
    print(f"🎥 开始渲染 3D 骨架并合成音频 (camera={args.camera_mode})...")
    skeleton_render(
        poses=poses_3d,       # ✨ 替换：传入运算好的 3D 坐标
        epoch=base_name,    
        out=out_dir,
        name=[args.audio],  
        sound=True,
        stitch=False,
        contact=contacts,     # ✨ 新增：传入脚部物理接触点数据，让地面支撑更直观（红/绿切换）
        render=True,
        camera_mode=args.camera_mode,
        output_path=args.output,
    )
    
    print(f"✅ 视频渲染完成！请查看: {args.output}")

if __name__ == "__main__":
    main()
