import json
import glob
import os

# 倒序排列，优先看最新的文件夹，找不到再往前找
run_dirs = sorted(glob.glob("wandb/offline-run-*"), reverse=True)
found = False

if not run_dirs:
    print("❌ 没有找到任何 wandb 日志文件夹。")
    exit()

print("🔍 正在逆向扫描历史日志...")

for run in run_dirs:
    summary_file = os.path.join(run, "files", "wandb-summary.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                # 确保这个 json 里面真的有我们想要的 Loss 数据
                if "Train Loss" in data:
                    print(f"\n=== 📁 找到历史可用数据: {os.path.basename(run)} ===")
                    print(f"📉 最新记录的 Loss 数据 (对应写入步数: {data.get('_step', '未知')}):")
                    print(f"  🔸 Train Loss (总损失): {data.get('Train Loss', '暂无')}")
                    print(f"  🔸 FK Loss (运动学):   {data.get('FK Loss', '暂无')}")
                    print(f"  🔸 Foot Loss (滑步):   {data.get('Foot Loss', '暂无')}")
                    print("===================================================\n")
                    found = True
                    break  # 找到最近的一个有效数据后就停止
        except json.JSONDecodeError:
            continue

if not found:
    print("⏳ 扫描了所有历史文件夹，都没有发现记录完好的 Loss 数据。")