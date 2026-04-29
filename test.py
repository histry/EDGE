import glob
import os
# 🌟 核心防御：大规模批量测试必须开启显存段扩展
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random

import numpy as np
import torch
import torch.nn.functional as F
import librosa
from tqdm import tqdm

from args import parse_test_opt
from data.slice import slice_audio
from EDGE import EDGE
from data.audio_extraction.baseline_features import extract as baseline_extract
from data.audio_extraction.wav2vec_librosa_features import extract as hybrid_extract
from data.audio_extraction.jukebox_features import extract as juke_extract

# sort filenames that look like songname_slice{number}.ext
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb: return -1
    if aa > bb: return 1
    if ka < kb: return -1
    if ka > kb: return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)

def test(opt):
    feature_map = {
        "jukebox": juke_extract,
        "baseline": baseline_extract,
        "hybrid": hybrid_extract,
    }
    if opt.feature_type not in feature_map:
        raise ValueError(f"Unknown feature_type: {opt.feature_type}")
    feature_func = feature_map[opt.feature_type]
    
    sample_length = opt.out_length
    sample_size = int(sample_length / 2.5) - 1

    temp_dir_list = []
    all_cond = []
    all_filenames = []
    
    # ==========================================
    # 🎯 终极修复点 1：关闭 EMA，加载绝对健康的活跃权重！
    # ==========================================
    print("🚀 正在初始化模型 (已强制加载健康的 Active 权重，抛弃损坏的 EMA)...")
    model = EDGE(
        opt.feature_type,
        opt.checkpoint,
        audio_dim=opt.audio_dim,
        EMA=False,
        beat_guidance_weight=opt.beat_guidance_weight,
        hard_keyframe_project=opt.hard_keyframe_project,
    )
    model.eval()

    if opt.use_cached_features:
        print("Using precomputed features")
        dir_list = glob.glob(os.path.join(opt.feature_cache_dir, "*/"))
        for dir in dir_list:
            file_list = sorted(glob.glob(f"{dir}/*.wav"), key=stringintkey)
            juke_file_list = sorted(glob.glob(f"{dir}/*.npy"), key=stringintkey)
            assert len(file_list) == len(juke_file_list)
            rand_idx = random.randint(0, len(file_list) - sample_size)
            file_list = file_list[rand_idx : rand_idx + sample_size]
            juke_file_list = juke_file_list[rand_idx : rand_idx + sample_size]
            cond_list = [np.load(x) for x in juke_file_list]
            all_filenames.append(file_list)
            all_cond.append(torch.from_numpy(np.array(cond_list)))
    else:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(opt.music_dir, "*.wav")):
            if opt.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(opt.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                Path(opt.render_dir).mkdir(parents=True, exist_ok=True)
                temp_dir = TemporaryDirectory(dir=opt.render_dir, prefix="edge_test_slices_")
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
            
            # ==========================================
            # 🎯 终极修复点 2 & 3：提取全曲并强制物理 30FPS 对齐
            # ==========================================
            print(f"🎵 正在提取全曲特征以保留完整节拍...")
            raw_feat, _ = feature_func(wav_file)
            
            duration = librosa.get_duration(path=wav_file) # 适配 librosa >= 0.10
            target_frames = int(duration * 30)
            print(f"📊 提取器原始帧数: {raw_feat.shape[0]} | 强制对齐到 30FPS 物理帧数: {target_frames}")
            
            raw_feat_tensor = torch.from_numpy(raw_feat).float().unsqueeze(0).transpose(1, 2)
            aligned_feat = F.interpolate(raw_feat_tensor, size=target_frames, mode='linear', align_corners=False)
            full_reps = aligned_feat.transpose(1, 2).squeeze(0).numpy()

            print(f"✂️ Slicing audio files: {wav_file}")
            slice_audio(wav_file, 2.5, 5.0, dirname)
            file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)

            if len(file_list) < sample_size:
                sample_size = max(1, len(file_list))

            rand_idx = random.randint(0, len(file_list) - sample_size)
            cond_list = []
            
            print(f"📦 正在将 30FPS 完美特征映射到切片序列...")
            for idx, file in enumerate(tqdm(file_list)):
                if (not opt.cache_features) and (not (rand_idx <= idx < rand_idx + sample_size)):
                    continue
                
                start_frame = idx * 75
                end_frame = start_frame + 150
                
                if end_frame > full_reps.shape[0]:
                    slice_reps = full_reps[start_frame:]
                    if slice_reps.shape[0] < 150:
                        pad_len = 150 - slice_reps.shape[0]
                        slice_reps = np.pad(slice_reps, ((0, pad_len), (0, 0)), mode='constant')
                else:
                    slice_reps = full_reps[start_frame:end_frame]

                if opt.cache_features:
                    featurename = os.path.splitext(file)[0] + ".npy"
                    np.save(featurename, slice_reps)
                
                if rand_idx <= idx < rand_idx + sample_size:
                    cond_list.append(slice_reps)
                    
            cond_list = torch.from_numpy(np.array(cond_list))
            all_cond.append(cond_list)
            all_filenames.append(file_list[rand_idx : rand_idx + sample_size])

    fk_out = opt.motion_save_dir if opt.save_motions else None

    print("Generating dances")
    for i in range(len(all_cond)):
        # 封装为字典推入显卡，匹配模型 bf16 精度
        # ❌ 原代码：
        # cond_dict = {"audio": all_cond[i].to(model.accelerator.device).to(torch.bfloat16)}
            
        # ✅ 替换为：
        audio_tensor = all_cond[i].to(model.accelerator.device).to(torch.bfloat16)
        traj_tensor = torch.zeros(audio_tensor.shape[0], audio_tensor.shape[1], 2, device=model.accelerator.device, dtype=torch.bfloat16)
            
        # 补齐在标准化流形下的绝对零点
        if model.normalizer is not None:
            mean_x = model.normalizer.mean[4]
            mean_z = model.normalizer.mean[6]
            std_x = model.normalizer.std[4]
            std_z = model.normalizer.std[6]
            traj_tensor[..., 0] = (traj_tensor[..., 0] - mean_x) / (std_x + 1e-6)
            traj_tensor[..., 1] = (traj_tensor[..., 1] - mean_z) / (std_z + 1e-6)
            
        cond_dict = {"audio": audio_tensor, "trajectory": traj_tensor}
        
        # 🌟 修复：将 all_filenames[i] 同时传给 name 和 wav 槽位，彻底唤醒底层的音频拼接引擎
        data_tuple = (None, cond_dict, all_filenames[i], all_filenames[i]) 
        
        # 严禁在推理期使用 autocast，避免精度坍塌
        with torch.no_grad():
            model.render_sample(
                data_tuple, "test", opt.render_dir, render_count=-1, fk_out=fk_out, render=not opt.no_render
            )
            
    print("Done")
    torch.cuda.empty_cache()
    for temp_dir in temp_dir_list:
        temp_dir.cleanup()

if __name__ == "__main__":
    opt = parse_test_opt()
    test(opt)
