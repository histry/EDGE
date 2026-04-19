import argparse
import os
import sys
import shutil
from pathlib import Path

# 【核心修复 1：路径锚定】强行将程序的工作目录切换到 create_dataset.py 所在的目录 (即 data/ 文件夹)
# 这样无论你在外面哪层目录敲命令，相对路径(如 "splits/ignore_list.txt")都能精准命中！
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from audio_extraction.baseline_features import extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from audio_extraction.wav2vec_librosa_features import extract_folder as hybrid_extract
from filter_split_data import *
from slice import *


def prepare_wavs(dataset_folder):
    # 【核心修复 2：音频搬运】EDGE 的切片脚本默认会去 aist/wavs 找原始音频
    # 我们之前提取特征时放在了 aist_audio/raw，这里写个保护逻辑自动把它复制过去
    target_wav_dir = Path(dataset_folder) / "wavs"
    source_wav_dir = Path("aist_audio/raw")
    
    if not target_wav_dir.exists() and source_wav_dir.exists():
        print(f"🔄 自动将音频从 {source_wav_dir} 复制到 {target_wav_dir} 以适配切片逻辑...")
        shutil.copytree(source_wav_dir, target_wav_dir)
        print("✅ 音频就位！")


def create_dataset(opt):
    prepare_wavs(opt.dataset_folder)
    
    # split the data according to the splits files
    print("Creating train / test split")
    split_data(opt.dataset_folder)
    
    # slice motions/music into sliding windows to create training dataset
    print("Slicing train data")
    slice_aistpp(f"train/motions", f"train/wavs")
    print("Slicing test data")
    slice_aistpp(f"test/motions", f"test/wavs")
    
    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract("train/wavs_sliced", "train/baseline_feats")
        baseline_extract("test/wavs_sliced", "test/baseline_feats")
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract("train/wavs_sliced", "train/jukebox_feats")
        jukebox_extract("test/wavs_sliced", "test/jukebox_feats")
    if opt.extract_hybrid:
        print("Extracting hybrid wav2vec+librosa features (Train & Test)")
        hybrid_extract("train/wavs_sliced", "train/hybrid_feats")
        hybrid_extract("test/wavs_sliced", "test/hybrid_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    
    # 【核心修复 3：名称对齐】把默认寻找的动作文件夹改成了你真实的文件夹名字 "aist"
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="aist",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    
    # 【核心修复 4：默认开启提取】因为我们需要用 hybrid 特征训练，直接设为 True
    parser.add_argument("--extract-hybrid", action="store_true", default=True)
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)