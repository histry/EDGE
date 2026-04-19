import os
from pathlib import Path

def rescue_training_set():
    print("🚀 启动数据集抢救计划...")
    
    motions_dir = Path("data/aist/motions")
    splits_dir = Path("data/splits")
    
    train_file = splits_dir / "crossmodal_train.txt"
    test_file = splits_dir / "crossmodal_test.txt"
    
    if not motions_dir.exists():
        print(f"❌ 找不到动作目录: {motions_dir.absolute()}")
        return
        
    # 1. 获取所有真实的动作文件（排除 Mac 垃圾文件）
    real_motions = [f.stem for f in motions_dir.glob("*.pkl") if not f.name.startswith("._")]
    print(f"📦 案发现场清点完毕：真实动作文件共 {len(real_motions)} 个 (另外一半是Mac隐藏文件)。")
    
    # 2. 读取测试集名单（这 20 个必须保留给期末考试）
    test_set = set()
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            test_set = set([x.strip().replace(".pkl", "") for x in f.readlines() if x.strip()])
    
    # 3. 剥离出非测试集的文件，强行编入训练军团
    train_set = [m for m in real_motions if m not in test_set]
    
    # 4. 覆写官方那个“画大饼”的训练名单
    with open(train_file, "w", encoding="utf-8") as f:
        for m in train_set:
            f.write(m + "\n")
            
    print(f"🎉 抢救成功！已将 {len(train_set)} 个幸存动作重新编入训练大纲。")
    print("👉 你的 4090 已经可以开饭了，请立刻运行: python data/create_dataset.py")

if __name__ == "__main__":
    rescue_training_set()