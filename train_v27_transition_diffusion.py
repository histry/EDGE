#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train V27 lightweight conditional diffusion in-betweening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.v27_transition_diffusion import TransitionDenoiser, _linear_beta_schedule


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.target = np.asarray(data["target"], dtype=np.float32)
        self.mask = np.asarray(data["mask"], dtype=np.float32)
        self.start = np.asarray(data["start"], dtype=np.float32)
        self.end = np.asarray(data["end"], dtype=np.float32)
        self.music = np.asarray(data["music"], dtype=np.float32)
        self.length = np.asarray(data["length"], dtype=np.float32)
        self.sample_weight = (
            np.asarray(data["sample_weight"], dtype=np.float32)
            if "sample_weight" in data.files
            else np.ones((len(self.target),), dtype=np.float32)
        )
        self.meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}

    def __len__(self) -> int:
        return int(len(self.target))

    def __getitem__(self, idx: int):
        return {
            "target": self.target[idx],
            "mask": self.mask[idx],
            "start": self.start[idx],
            "end": self.end[idx],
            "music": self.music[idx],
            "length": self.length[idx],
            "sample_weight": self.sample_weight[idx],
        }


def masked_weighted_mse(pred: torch.Tensor, noise: torch.Tensor, mask: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    per_sample = (((pred - noise) ** 2) * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp_min(1.0)
    weight = sample_weight.reshape(-1).clamp_min(1e-4)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-4)


def train_epoch(model, loader, opt, device, diffusion_steps: int) -> float:
    beta, alpha, alpha_bar = _linear_beta_schedule(diffusion_steps, device)
    model.train()
    losses = []
    for batch in loader:
        target = batch["target"].to(device)
        mask = batch["mask"].to(device).unsqueeze(-1)
        start = batch["start"].to(device)
        end = batch["end"].to(device)
        music = batch["music"].to(device)
        length = batch["length"].to(device).reshape(-1, 1)
        sample_weight = batch["sample_weight"].to(device).reshape(-1)
        b, k, _ = target.shape
        idx = torch.randint(0, diffusion_steps, (b,), device=device)
        noise = torch.randn_like(target)
        ab = alpha_bar[idx].reshape(b, 1, 1)
        noisy = torch.sqrt(ab) * target + torch.sqrt(1.0 - ab) * noise
        t = idx.float() / max(diffusion_steps - 1, 1)
        pos = torch.linspace(1.0 / (k + 1), k / (k + 1), k, device=device).reshape(1, k, 1).expand(b, -1, -1)
        pred = model(noisy, t, start, end, music, (length / 120.0).clamp(0, 1), pos)
        loss = masked_weighted_mse(pred, noise, mask, sample_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def eval_epoch(model, loader, device, diffusion_steps: int) -> float:
    beta, alpha, alpha_bar = _linear_beta_schedule(diffusion_steps, device)
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            mask = batch["mask"].to(device).unsqueeze(-1)
            start = batch["start"].to(device)
            end = batch["end"].to(device)
            music = batch["music"].to(device)
            length = batch["length"].to(device).reshape(-1, 1)
            sample_weight = batch["sample_weight"].to(device).reshape(-1)
            b, k, _ = target.shape
            idx = torch.randint(0, diffusion_steps, (b,), device=device)
            noise = torch.randn_like(target)
            ab = alpha_bar[idx].reshape(b, 1, 1)
            noisy = torch.sqrt(ab) * target + torch.sqrt(1.0 - ab) * noise
            t = idx.float() / max(diffusion_steps - 1, 1)
            pos = torch.linspace(1.0 / (k + 1), k / (k + 1), k, device=device).reshape(1, k, 1).expand(b, -1, -1)
            pred = model(noisy, t, start, end, music, (length / 120.0).clamp(0, 1), pos)
            loss = masked_weighted_mse(pred, noise, mask, sample_weight)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ds = TransitionDataset(args.data)
    n_val = max(1, int(round(len(ds) * args.val_ratio)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionDenoiser(motion_dim=ds.target.shape[-1], music_dim=ds.music.shape[-1], hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = float("inf")
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, device, args.diffusion_steps)
        val_loss = eval_epoch(model, val_loader, device, args.diffusion_steps)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[v27 transition diffusion] epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if val_loss < best:
            best = val_loss
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "motion_dim": int(ds.target.shape[-1]),
                        "music_dim": int(ds.music.shape[-1]),
                        "hidden_dim": int(args.hidden_dim),
                        "diffusion_steps": int(args.diffusion_steps),
                        "dataset_meta": ds.meta,
                    },
                    "best_val_loss": float(best),
                    "epoch": int(epoch),
                },
                ckpt_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}")
                break

    (out_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "BEST_V27_TRANSITION_DIFFUSION_CKPT.txt").write_text(str(ckpt_dir / "best.pt"), encoding="utf-8")
    print(f"[SAVED] {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
