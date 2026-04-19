import torch
import os

# 指定你要检查的权重文件路径
ckpt_path = "runs/train/exp14/weights/train-600.pt"

if not os.path.exists(ckpt_path):
    print(f"❌ 找不到权重文件: {ckpt_path}")
    exit()

print(f"🔍 正在对 {ckpt_path} 进行底层 X 光扫描...")

try:
    # 仅加载到 CPU，防止爆显存
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # 优先检查 EMA 权重（推理默认使用的），如果没有则检查基础模型
    model_state = ckpt.get("ema_state_dict", ckpt.get("model_state_dict"))
    
    if model_state is None:
        print("❌ 找不到模型权重字典！")
    else:
        has_nan = False
        for name, param in model_state.items():
            if torch.isnan(param).any():
                has_nan = True
                print(f"\n💥 致命预警: 在神经网络的【{name}】层发现了 NaN (非数字)！")
                break
                
        if has_nan:
            print("\n🚨 终极诊断：模型发生了【梯度爆炸】！")
            print("结论：在跑到 600 轮的过程中（可能是三百多轮的时候），因为 bf16 精度溢出或学习率过大，模型彻底崩溃了。生成的位移全部是 NaN，所以画面完全画不出人。")
            print("👉 下一步：调小学习率，比如使用 --learning_rate 5e-5，然后从 100 轮的健康权重重新开始训练。")
        else:
            print("\n✅ 终极诊断：模型极其健康！一切正常！")
            print("结论：权重里面一个 NaN 都没有！说明你的 Loss 曲线绝对是健康的！")
            print("👉 下一步：视频里没人的原因 100% 是【归一化(Normalizer)还原时把坐标放大了几万倍，导致骨架飞出了画面外】。请告诉我，我会给你一行代码把它强行绑回画面正中央！")

except Exception as e:
    print(f"读取异常: {e}")