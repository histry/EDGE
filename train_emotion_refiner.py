#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 Emotion-conditioned Seam Residual Refiner

最适合本项目的训练方式：
- 输入：V16C/V17 的稳定短语 + 音乐情感语义条件 + seam mask
- 目标：重建 V16C teacher，重点修复 seam 附近扰动
- Loss：
  1. seam-weighted reconstruction loss
  2. velocity continuity loss
  3. acceleration / jitter loss
  4. root X/Z in-place lock loss
  5. style-preservation loss outside seam
  6. residual regularization
  7. music-emotion alignment loss（弱权重，避免机械卡点）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model.emotion_conditioned_refiner import EmotionConditionedSeamRefiner

CONTACT = slice(0, 4)
ROOT_X = 4
ROOT_Y = 5
ROOT_Z = 6
ROT_START = 7


class EmotionRefinerDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz samples found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        z = np.load(self.files[idx], allow_pickle=False)
        return {
            "noisy": torch.from_numpy(z["noisy"].astype("float32")),
            "target": torch.from_numpy(z["target"].astype("float32")),
            "music": torch.from_numpy(z["music"].astype("float32")),
            "seam_mask": torch.from_numpy(z["seam_mask"].astype("float32")),
        }


def body_region_energy(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    # x: [B,T,151]
    rot = x[:, :, ROT_START:].reshape(x.shape[0], x.shape[1], 24, 6)
    d = rot[:, 1:] - rot[:, :-1]

    lower = torch.linalg.norm(d[:, :, 0:8].reshape(x.shape[0], x.shape[1]-1, -1), dim=-1)
    torso = torch.linalg.norm(d[:, :, 8:14].reshape(x.shape[0], x.shape[1]-1, -1), dim=-1)
    upper = torch.linalg.norm(d[:, :, 14:24].reshape(x.shape[0], x.shape[1]-1, -1), dim=-1)
    full = torch.linalg.norm(d.reshape(x.shape[0], x.shape[1]-1, -1), dim=-1)
    return {"lower": lower, "torso": torso, "upper": upper, "full": full}


def normalize_seq(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # normalize per batch item
    mn = x.amin(dim=1, keepdim=True)
    mx = x.amax(dim=1, keepdim=True)
    return (x - mn) / (mx - mn + eps)


def music_emotion_alignment_loss(pred: torch.Tensor, music: torch.Tensor) -> torch.Tensor:
    """
    弱音乐情感语义对齐：
    - arousal/tension 高的音乐段，对应 upper/torso motion energy 更高；
    - calmness 高的音乐段，对应 full-body energy 更平缓。
    不强制 beat 卡点，避免破坏敦煌舞动作连续性。
    """
    if music.shape[-1] < 8 or pred.shape[1] < 3:
        return pred.new_tensor(0.0)

    energy = body_region_energy(pred)
    upper = normalize_seq(energy["upper"])
    torso = normalize_seq(energy["torso"])
    full = normalize_seq(energy["full"])

    music_mid = music[:, 1:, :]
    arousal = normalize_seq(music_mid[:, :, 4])
    tension = normalize_seq(music_mid[:, :, 6])
    calmness = normalize_seq(music_mid[:, :, 7])

    expressive = normalize_seq(0.60 * upper + 0.40 * torso)

    loss_arousal = F.smooth_l1_loss(expressive, arousal)
    loss_tension = F.smooth_l1_loss(normalize_seq(0.55 * upper + 0.45 * full), tension)

    # calmness 高时，不要求完全静止，而是鼓励能量曲线不要过度尖锐。
    calm_target = 1.0 - calmness
    loss_calm = F.smooth_l1_loss(full, calm_target)

    return loss_arousal + 0.7 * loss_tension + 0.3 * loss_calm


def compute_losses(pred, residual, batch, lambdas):
    target = batch["target"]
    noisy = batch["noisy"]
    music = batch["music"]
    seam = batch["seam_mask"]

    # 通道权重：contacts 不重写，root_xz 硬约束，root_y 低权重，rot 主权重。
    ch = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    ch[0:4] = 0.0
    ch[ROOT_X] = 0.0
    ch[ROOT_Z] = 0.0
    ch[ROOT_Y] = 0.35
    ch[ROT_START:] = 1.0
    ch = ch.view(1, 1, -1)

    # seam 附近权重更大，非 seam 也保留轻微重建，防止整体漂。
    time_w = 1.0 + 4.0 * seam
    rec = (F.smooth_l1_loss(pred * ch, target * ch, reduction="none") * time_w).mean()

    # 速度与加速度，保证边界连续性。
    vel_p = pred[:, 1:] - pred[:, :-1]
    vel_t = target[:, 1:] - target[:, :-1]
    vel_w = 1.0 + 3.0 * seam[:, 1:]
    vel = (F.smooth_l1_loss(vel_p * ch, vel_t * ch, reduction="none") * vel_w).mean()

    if pred.shape[1] > 2:
        acc_p = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        acc_t = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        acc = F.smooth_l1_loss(acc_p[:, :, ROT_START:], acc_t[:, :, ROT_START:])
    else:
        acc = pred.new_tensor(0.0)

    # 非 seam 区域尽量保持输入，避免模型把整段动作平均化。
    outside = 1.0 - seam
    preserve = (F.smooth_l1_loss(pred * ch, noisy * ch, reduction="none") * outside).mean()

    # residual 正则，避免过度修改 V16C teacher。
    residual_reg = torch.mean(torch.abs(residual[:, :, ROT_START:]))

    # 原地约束。
    root_lock = (pred[:, :, ROOT_X].abs().mean() + pred[:, :, ROOT_Z].abs().mean())

    music_loss = music_emotion_alignment_loss(pred, music)

    total = (
        lambdas["rec"] * rec
        + lambdas["vel"] * vel
        + lambdas["acc"] * acc
        + lambdas["preserve"] * preserve
        + lambdas["residual"] * residual_reg
        + lambdas["root"] * root_lock
        + lambdas["music"] * music_loss
    )

    return {
        "total": total,
        "rec": rec.detach(),
        "vel": vel.detach(),
        "acc": acc.detach(),
        "preserve": preserve.detach(),
        "residual": residual_reg.detach(),
        "root": root_lock.detach(),
        "music": music_loss.detach(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--residual_scale", type=float, default=0.20)
    ap.add_argument("--lambda_rec", type=float, default=1.0)
    ap.add_argument("--lambda_vel", type=float, default=0.6)
    ap.add_argument("--lambda_acc", type=float, default=0.15)
    ap.add_argument("--lambda_preserve", type=float, default=0.25)
    ap.add_argument("--lambda_residual", type=float, default=0.03)
    ap.add_argument("--lambda_root", type=float, default=5.0)
    ap.add_argument("--lambda_music", type=float, default=0.04)
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = EmotionRefinerDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    model = EmotionConditionedSeamRefiner(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)

    lambdas = {
        "rec": args.lambda_rec,
        "vel": args.lambda_vel,
        "acc": args.lambda_acc,
        "preserve": args.lambda_preserve,
        "residual": args.lambda_residual,
        "root": args.lambda_root,
        "music": args.lambda_music,
    }

    config = vars(args)
    config["lambdas"] = lambdas
    config["dataset_size"] = len(ds)
    (out_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best = 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        meters = {k: 0.0 for k in ["total", "rec", "vel", "acc", "preserve", "residual", "root", "music"]}
        n = 0

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            pred, residual = model(batch["noisy"], batch["music"], batch["seam_mask"], return_residual=True)
            losses = compute_losses(pred, residual, batch, lambdas)

            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = batch["noisy"].shape[0]
            n += bs
            for k in meters:
                meters[k] += float(losses[k].detach().cpu()) * bs

        sched.step()
        meters = {k: v / max(n, 1) for k, v in meters.items()}
        msg = " ".join([f"{k}={v:.6f}" for k, v in meters.items()])
        print(f"[epoch {ep:04d}/{args.epochs}] lr={sched.get_last_lr()[0]:.2e} {msg}", flush=True)

        if meters["total"] < best:
            best = meters["total"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / "best.pt")

        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / f"epoch_{ep:04d}.pt")

    torch.save({"model": model.state_dict(), "config": config, "epoch": args.epochs}, ckpt_dir / "final.pt")
    print(f"saved_final: {ckpt_dir / 'final.pt'}")
    print(f"saved_best:  {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
