#!/usr/bin/env python3
"""
V16D Refiner 数据集构建
输入：V16C 输出短语 .npy 或 .mp4 + 原始 motion unit
输出：phrase-level dataset，用于 Refiner 蒸馏训练
"""
import os
import argparse
import numpy as np
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v16c_input_dir", required=True, help="V16C 输出目录")
    ap.add_argument("--motion_unit_dir", required=True, help="原始 motion unit")
    ap.add_argument("--out_dir", required=True, help="Refiner dataset 输出目录")
    ap.add_argument("--phrase_len", type=int, default=45, help="短语长度")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    v16c_files = sorted(Path(args.v16c_input_dir).glob("*.npy"))
    for i, f in enumerate(v16c_files):
        data = np.load(f)  # [1,T,151] 或 [T,151]
        if data.ndim == 3:
            data = data[0]
        T = data.shape[0]
        for start in range(0, T - args.phrase_len + 1, args.phrase_len):
            phrase = data[start:start+args.phrase_len]
            np.save(Path(args.out_dir)/f"{f.stem}_phrase{start}.npy", phrase)
    print(f"生成 {len(v16c_files)} 个 V16C 输入短语数据集完成")

if __name__ == "__main__":
    main()
