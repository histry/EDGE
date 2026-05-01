"""Motion-domain adaptation for MMR using unpaired Dunhuang BVH/motion clips.

Why this exists:
- AIST++ can train true paired audio-motion MMR.
- Dunhuang BVH has no paired music, so we should not use strong audio-motion loss.
- Instead, freeze AudioEncoder and adapt MotionEncoder with motion-only contrastive
  learning plus distillation to preserve the AIST++ shared space.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model.mmr_encoder import load_mmr_model, save_mmr_checkpoint
from losses.mmr_loss import symmetric_infonce_loss
from mmr_data_utils import iter_motion_files, load_motion_151, resample_sequence, load_normalizer_from_checkpoint, normalize_motion_if_needed


class DunhuangMotionClipDataset(Dataset):
    def __init__(self, motion_dir: str, seq_len=150, stride=30, checkpoint="", pose_space="normalized"):
        self.seq_len = int(seq_len)
        self.items = []
        normalizer = load_normalizer_from_checkpoint(checkpoint) if checkpoint else None
        for path in iter_motion_files(motion_dir):
            try:
                motion = load_motion_151(path)
                motion = normalize_motion_if_needed(motion, normalizer, pose_space)
            except Exception as exc:
                print("skip", path, exc)
                continue
            if len(motion) < 2:
                continue
            if len(motion) < self.seq_len:
                self.items.append((str(path), 0, motion))
            else:
                for start in range(0, len(motion) - self.seq_len + 1, int(stride)):
                    self.items.append((str(path), start, motion[start : start + self.seq_len]))
        if not self.items:
            raise RuntimeError(f"No motion clips found in {motion_dir}")

    def __len__(self):
        return len(self.items)

    def augment(self, clip: np.ndarray) -> np.ndarray:
        clip = clip.copy().astype(np.float32)
        # random temporal crop/resample
        if len(clip) > 20:
            scale = random.uniform(0.85, 1.15)
            new_len = max(20, int(round(len(clip) * scale)))
            clip = resample_sequence(clip, new_len)
            clip = resample_sequence(clip, self.seq_len)
        # small feature noise excluding contact/root mostly
        noise = np.random.randn(*clip[:, 7:151].shape).astype(np.float32) * 0.005
        clip[:, 7:151] += noise
        # root xz translation invariance
        clip[:, [4, 6]] -= clip[:1, [4, 6]]
        return clip.astype(np.float32)

    def __getitem__(self, idx):
        _, _, clip = self.items[idx]
        clip = resample_sequence(clip, self.seq_len)
        v1 = self.augment(clip)
        v2 = self.augment(clip)
        return {"motion1": torch.from_numpy(v1).float(), "motion2": torch.from_numpy(v2).float()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mmr_checkpoint", required=True)
    p.add_argument("--dunhuang_motion_dir", required=True)
    p.add_argument("--edge_checkpoint", default="", help="Optional EDGE checkpoint normalizer when Dunhuang files are physical-space.")
    p.add_argument("--pose_space", default="normalized", choices=["normalized", "physical"])
    p.add_argument("--out", default="runs/mmr/mmr_aist_dunhuang_motion_adapt.pt")
    p.add_argument("--seq_len", type=int, default=150)
    p.add_argument("--stride", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--distill_weight", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_mmr_model(args.mmr_checkpoint, device=device)
    teacher = load_mmr_model(args.mmr_checkpoint, device=device)
    teacher.eval()
    for p in model.audio_encoder.parameters():
        p.requires_grad = False
    for p in teacher.parameters():
        p.requires_grad = False

    ds = DunhuangMotionClipDataset(args.dunhuang_motion_dir, args.seq_len, args.stride, args.edge_checkpoint, args.pose_space)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for batch in tqdm(loader, desc=f"adapt {epoch}"):
            m1 = batch["motion1"].to(device)
            m2 = batch["motion2"].to(device)
            z1 = model.encode_motion(m1)
            z2 = model.encode_motion(m2)
            logits = model.logit_scale.exp().clamp(max=100.0) * z1 @ z2.t()
            loss_con = symmetric_infonce_loss(logits)
            with torch.no_grad():
                t1 = teacher.encode_motion(m1)
                t2 = teacher.encode_motion(m2)
            loss_distill = 0.5 * ((z1 - t1).pow(2).mean() + (z2 - t2).pow(2).mean())
            loss = loss_con + float(args.distill_weight) * loss_distill
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
            count += 1
        print(f"Epoch {epoch} | loss={total / max(count,1):.6f}")
        if epoch % 5 == 0 or epoch == args.epochs:
            save_mmr_checkpoint(args.out, model, {"args": vars(args), "epoch": epoch})
            print("saved:", args.out)


if __name__ == "__main__":
    main()
