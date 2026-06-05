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

from model.v21_style_ranker import V21StyleRanker


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    z = np.load(args.data)
    positive = torch.from_numpy(np.asarray(z["positive"], dtype=np.float32))
    negative = torch.from_numpy(np.asarray(z["negative"], dtype=np.float32))
    dataset = TensorDataset(positive, negative)
    val_size = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V21StyleRanker(input_dim=positive.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {"input_dim": int(positive.shape[1]), "hidden_dim": args.hidden_dim, "dropout": args.dropout}
    best = float("inf")
    history = []

    def run(loader, training):
        model.train(training)
        total = count = 0
        for p, n in loader:
            p, n = p.to(device), n.to(device)
            with torch.set_grad_enabled(training):
                ps, ns = model(p), model(n)
                loss = F.softplus(ns - ps + 0.20).mean() + 0.05 * (ps.pow(2).mean() + ns.pow(2).mean())
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
            total += float(loss.detach()) * len(p)
            count += len(p)
        return total / max(count, 1)

    for epoch in range(1, args.epochs + 1):
        tr, va = run(train_loader, True), run(val_loader, False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        if va < best:
            best = va
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_loss": va}, ckpt_dir / "best.pt")
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train={tr:.6f} val={va:.6f} best={best:.6f}")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": args.epochs, "val_loss": history[-1]["val"]}, ckpt_dir / "final.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
