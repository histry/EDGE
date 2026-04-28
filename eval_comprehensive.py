import argparse
import glob
import os
import pickle
import numpy as np
import librosa
from scipy.signal import argrelextrema
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import itertools

def calc_kinetic_features(joint3d, DT=1/30.0):
    """提取运动学特征，用于计算多样性 Diversity"""
    # joint3d: (seq_len, num_joints, 3)
    velocity = (joint3d[1:] - joint3d[:-1]) / DT
    acceleration = (velocity[1:] - velocity[:-1]) / DT
    
    # 提取特征向量: 速度均值、方差，加速度均值、方差
    v_mean = np.mean(velocity, axis=(0, 1))
    v_std = np.std(velocity, axis=(0, 1))
    a_mean = np.mean(acceleration, axis=(0, 1))
    a_std = np.std(acceleration, axis=(0, 1))
    
    feature_vec = np.concatenate([v_mean, v_std, a_mean, a_std])
    return feature_vec

def calc_diversity(pkl_files):
    """计算生成多样性 (Kinetic Feature Pairwise Distance)"""
    print("📊 正在计算生成多样性 (Diversity)...")
    features = []
    for pkl in pkl_files:
        data = pickle.load(open(pkl, "rb"))
        joint3d = data["full_pose"]
        features.append(calc_kinetic_features(joint3d))
        
    features = np.array(features)
    if len(features) < 2:
        return 0.0
        
    # 计算所有生成的舞蹈序列之间的平均欧式距离
    distances = []
    for f1, f2 in itertools.combinations(features, 2):
        distances.append(np.linalg.norm(f1 - f2))
        
    diversity_score = np.mean(distances)
    print(f"✅ Diversity Score: {diversity_score:.4f} (越高代表生成的舞蹈越不重复)")
    return diversity_score

def calc_beatalign(pkl_files, audio_dir, fps=30):
    """计算节拍对齐率 (BeatAlign Score)"""
    print("🥁 正在计算音频节拍对齐率 (BeatAlign)...")
    beat_scores = []
    
    for pkl in tqdm(pkl_files):
        data = pickle.load(open(pkl, "rb"))
        joint3d = data["full_pose"]
        
        # 1. 提取动作节拍 (Motion Beats)
        # 计算全局运动速度，速度的局部极小值通常代表动作的“顿点/发力点”
        velocity = np.linalg.norm(joint3d[1:] - joint3d[:-1], axis=-1).mean(axis=-1)
        motion_beats = argrelextrema(velocity, np.less)[0]
        
        # 2. 匹配音频并提取音乐节拍 (Audio Beats)
        base_name = os.path.basename(pkl)
        # 假设你的音频名字包含在 pkl 名字中，或者一一对应
        # 你可能需要根据实际命名规则修改这里的提取逻辑
        audio_name = base_name.split('_')[1] + ".wav" if len(base_name.split('_')) > 1 else base_name.replace(".pkl", ".wav")
        audio_path = os.path.join(audio_dir, audio_name)
        
        if not os.path.exists(audio_path):
            continue
            
        y, sr = librosa.load(audio_path, sr=None)
        _, audio_beats_time = librosa.beat.beat_track(y=y, sr=sr, units='time')
        audio_beats_frame = np.round(audio_beats_time * fps).astype(int)
        
        if len(audio_beats_frame) == 0 or len(motion_beats) == 0:
            continue
            
        # 3. 计算高斯加权对齐得分
        # 衡量每个动作节拍距离最近的音乐节拍有多近
        sigma = 3.0 # 容忍度约 3 帧 (0.1秒)
        score = 0
        for mb in motion_beats:
            dist = np.min(np.abs(audio_beats_frame - mb))
            score += np.exp(-(dist**2) / (2 * sigma**2))
            
        beat_scores.append(score / len(motion_beats))
        
    final_score = np.mean(beat_scores) if beat_scores else 0.0
    print(f"✅ BeatAlign Score: {final_score:.4f} (范围0-1，越高代表卡点越准)")
    return final_score

def calc_trajectory_error(pkl_files):
    """计算轨迹误差 (Dynamic Time Warping / 空间偏移量)"""
    print("🗺️ 正在计算空间轨迹误差 (Trajectory DTW Error)...")
    dtw_errors = []
    
    for pkl in pkl_files:
        data = pickle.load(open(pkl, "rb"))
        
        if "target_trajectory" not in data or data["target_trajectory"] is None:
            continue
            
        joint3d = data["full_pose"]
        target_traj = data["target_trajectory"]
        
        # Root 是第 0 个关节，提取 X 和 Z 坐标构成生成轨迹
        gen_root_x = joint3d[:, 0, 0]
        gen_root_z = joint3d[:, 0, 2]
        gen_traj = np.stack([gen_root_x, gen_root_z], axis=1)
        
        # 确保序列长度对齐
        min_len = min(len(gen_traj), len(target_traj))
        gen_traj = gen_traj[:min_len]
        target_traj = target_traj[:min_len]
        
        # 计算两条 2D 曲线的动态时间规整距离 (衡量形状和位移的综合误差)
        distance, _ = fastdtw(gen_traj, target_traj, dist=euclidean)
        # 归一化到每帧的平均偏移米数
        avg_error = distance / min_len
        dtw_errors.append(avg_error)
        
    if not dtw_errors:
        print("⚠️ 警告：PKL 文件中没有找到 target_trajectory，请确保已经修改了 diffusion.py 并重新生成！")
        return 0.0
        
    final_error = np.mean(dtw_errors)
    print(f"✅ Trajectory Error (DTW): {final_error:.4f} 米/帧 (越低代表走位越精准)")
    return final_error

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_dir", type=str, required=True, help="存放生成的 .pkl 文件夹路径")
    parser.add_argument("--audio_dir", type=str, required=True, help="存放对应测试 .wav 音乐的文件夹路径")
    args = parser.parse_args()

    pkl_files = glob.glob(os.path.join(args.motion_dir, "*.pkl"))
    if not pkl_files:
        print(f"❌ 在 {args.motion_dir} 中没有找到 PKL 文件！")
        exit()
        
    print(f"🔍 找到 {len(pkl_files)} 个动作文件，开始全量评估...\n")
    
    # 1. 评估生成多样性
    calc_diversity(pkl_files)
    print("-" * 40)
    
    # 2. 评估节拍对齐率
    calc_beatalign(pkl_files, args.audio_dir)
    print("-" * 40)
    
    # 3. 评估轨迹跟随误差
    calc_trajectory_error(pkl_files)
    print("-" * 40)
    print("🎉 评估全部完成！可将上述指标直接填入论文表格。")