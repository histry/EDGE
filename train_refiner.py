#!/usr/bin/env python3
"""
V16D Refiner 训练主程序
可选择蒸馏 V16C 输出或 reconstruction
可选音乐 onset embedding
"""
import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model.phrase_refiner import PhraseRefiner

class RefinerDataset(Dataset):
    def __init__(self, data_dir, use_onset=False):
        self.files = list(Path(data_dir).glob("*.npy"))
        self.use_onset = use_onset
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        motion = np.load(self.files[idx]).astype('float32')
        if self.use_onset:
            onset = np.zeros((motion.shape[0],16),dtype='float32')  # 占位
            return motion, onset, motion
        return motion, motion  # input, target

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--use_onset", type=int, default=0)
    ap.add_argument("--scope", type=str, default='phrase')  # phrase / local
    return ap.parse_args()

def main():
    args = parse_args()
    dataset = RefinerDataset(args.data_dir, use_onset=bool(args.use_onset))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhraseRefiner(use_onset=bool(args.use_onset)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        for batch in loader:
            if args.use_onset:
                x,onset,y = batch
                x = x.to(device); onset=onset.to(device); y=y.to(device)
                pred = model(x,onset)
            else:
                x,y = batch
                x = x.to(device); y = y.to(device)
                pred = model(x)
            loss = loss_fn(pred,y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {epoch+1}/{args.epochs} Loss: {loss.item():.6f}")
    torch.save(model.state_dict(),"phrase_refiner.pt")
    print("训练完成，模型保存为 phrase_refiner.pt")

if __name__ == "__main__":
    main()
