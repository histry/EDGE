#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from model.v21_music_router import V21MusicMotionRouter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--margin_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(args.data, allow_pickle=True)
    music = torch.from_numpy(np.asarray(z["music"], dtype=np.float32))
    positive = torch.from_numpy(np.asarray(z["positive"], dtype=np.float32))
    negative = torch.from_numpy(np.asarray(z["negative"], dtype=np.float32))
    dataset = TensorDataset(music, positive, negative)
    val_size = max(1, int(round(len(dataset) * 0.1)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V21MusicMotionRouter(
        music_dim=music.shape[1],
        motion_dim=positive.shape[1],
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "music_dim": int(music.shape[1]),
        "motion_dim": int(positive.shape[1]),
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
    }

    best_val = float("inf")
    history = []

    def run_epoch(loader, training: bool) -> float:
        model.train(training)
        total_loss = 0.0
        total_count = 0
        for m, p, n in loader:
            m, p, n = m.to(device), p.to(device), n.to(device)
            with torch.set_grad_enabled(training):
                me = model.encode_music(m)
                pe = model.encode_motion(p)
                ne = model.encode_motion(n)
                scale = model.logit_scale.exp().clamp(max=100.0)
                logits = scale * (me @ pe.t())
                labels = torch.arange(len(m), device=device)
                contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
                pos_sim = (me * pe).sum(dim=-1)
                neg_sim = (me * ne).sum(dim=-1)
                margin = F.softplus(neg_sim - pos_sim + 0.10).mean()
                loss = contrastive + args.margin_weight * margin
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total_loss += float(loss.detach()) * len(m)
            total_count += len(m)
        return total_loss / max(total_count, 1)

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, True)
        val_loss = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} best={best_val:.6f}", flush=True)

    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": args.epochs, "val_loss": history[-1]["val_loss"]}, ckpt_dir / "final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
