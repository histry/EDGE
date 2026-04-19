import os
import pickle
import shutil
from pathlib import Path
import soundfile as sf  # 引入音频处理库

def fileToList(f):
    if not os.path.exists(f):
        print(f"❌ 找不到切分名单: {f}")
        return []
        
    with open(f, "r", encoding="utf-8") as file:
        out = file.readlines()
        
    # 安全清洗：剥离换行符和后缀
    out = [x.strip().replace(".pkl", "") for x in out if x.strip()]
    
    ignore_file = "splits/ignore_list.txt"
    if os.path.exists(ignore_file):
        with open(ignore_file, "r", encoding="utf-8") as file:
            filter_list = set([x.strip().replace(".pkl", "") for x in file.readlines() if x.strip()])
        # 剔除黑名单动作
        out = [x for x in out if x not in filter_list]
        
    return out

def split_data(dataset_folder):
    train_list = fileToList("splits/crossmodal_train.txt")
    test_list = fileToList("splits/crossmodal_test.txt")

    print(f"📊 准备分配: 训练集名单 {len(train_list)} 个, 测试集名单 {len(test_list)} 个")

    motion_dir = Path(dataset_folder) / "motions"
    wav_dir = Path(dataset_folder) / "wavs"

    for split_list, folder in zip([train_list, test_list], ["train", "test"]):
        os.makedirs(os.path.join(folder, "motions"), exist_ok=True)
        os.makedirs(os.path.join(folder, "wavs"), exist_ok=True)
        
        success_count = 0
        missing_count = 0
        
        for sequence in split_list:
            motion_path = motion_dir / f"{sequence}.pkl"
            
            try:
                base_wav_name = sequence.split("_")[-2] + ".wav"
            except:
                base_wav_name = f"{sequence}.wav"
                
            wav_path = wav_dir / base_wav_name
            
            # 容错：找不到就跳过
            if not motion_path.exists() or not wav_path.exists():
                missing_count += 1
                if missing_count == 1:
                    print(f"⚠️ [排查线索] 首个找不到的文件 -> 动作存在:{motion_path.exists()}, 音乐存在:{wav_path.exists()} | 目标名称:{sequence}")
                continue
            
            try:
                with open(motion_path, "rb") as f:
                    motion_data = pickle.load(f)
                    
                trans = motion_data["smpl_trans"]
                pose = motion_data["smpl_poses"]
                scale = motion_data["smpl_scaling"]
                
                out_data = {"pos": trans, "q": pose, "scale": scale}
                
                # ----- 【核心修复：动态裁剪音频使长度完美匹配】 -----
                # AIST++ 原始动作是 60 FPS，计算该段动作的实际秒数
                num_frames = pose.shape[0]
                duration_sec = num_frames / 60.0 
                
                # 读取长音频
                audio_data, sr = sf.read(str(wav_path))
                
                # 算出需要截取多少个音频样本点
                num_samples = int(duration_sec * sr)
                
                # 从头精准一刀切
                cropped_audio = audio_data[:min(num_samples, len(audio_data))]
                
                # 保存瘦身动作
                with open(f"{folder}/motions/{sequence}.pkl", "wb") as f_out:
                    pickle.dump(out_data, f_out)
                    
                # 保存裁剪后的纯净配套音频
                sf.write(f"{folder}/wavs/{sequence}.wav", cropped_audio, sr)
                # ----------------------------------------------------
                
                success_count += 1
            except Exception as e:
                print(f"⚠️ 读取 {sequence} 数据时出现异常: {e}")
                missing_count += 1
            
        print(f"✅ [{folder}] 数据集拆分完成: 成功对齐并裁剪 {success_count} 个, 缺失/跳过 {missing_count} 个。")
        
        # 针对 0 文件的严重警告
        if folder == "train" and success_count == 0:
            print("\n🚨🚨🚨 严重警告 🚨🚨🚨")
            print("训练集成功处理了 0 个文件！请务必检查你的 `data/aist/motions/` 文件夹！")
            print("里面可能只放了几十个文件，并没有把 motions.zip 里的 900 多个动作全部放进去。如果不补齐动作文件，之后的训练将无法启动！\n")