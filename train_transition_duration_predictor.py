#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.transition_duration_predictor import TransitionDurationPredictor, TRANSITION_BINS


class DPNDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz found in {data_dir}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        return {"x": torch.from_numpy(z["x"].astype("float32")), "y_cls": torch.tensor(int(z["y_cls"]), dtype=torch.long), "y_len": torch.tensor(float(z["y_len"]), dtype=torch.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ds = DPNDataset(args.data_dir)
    input_dim = int(ds[0]["x"].numel())
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransitionDurationPredictor(input_dim=input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    bins = torch.as_tensor(TRANSITION_BINS, device=device, dtype=torch.float32)
    best = 1e9
    config = vars(args); config.update({"input_dim": input_dim, "transition_bins": TRANSITION_BINS})
    (out_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    for ep in range(1, args.epochs + 1):
        model.train(); total = ce_total = reg_total = acc_total = n = 0
        for batch in dl:
            x = batch["x"].to(device); y = batch["y_cls"].to(device); y_len = batch["y_len"].to(device)
            out = model(x)
            ce = F.cross_entropy(out["logits"], y)
            pred_len = (torch.softmax(out["logits"], dim=-1) * bins[None]).sum(dim=-1)
            reg = F.smooth_l1_loss(pred_len, y_len)
            loss = ce + 0.05 * reg
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            bs = x.shape[0]; n += bs
            total += float(loss.detach().cpu()) * bs; ce_total += float(ce.detach().cpu()) * bs; reg_total += float(reg.detach().cpu()) * bs
            acc_total += float((out["logits"].argmax(dim=-1) == y).float().mean().detach().cpu()) * bs
        sched.step()
        meters = {"loss": total/max(n,1), "ce": ce_total/max(n,1), "reg": reg_total/max(n,1), "acc": acc_total/max(n,1)}
        print(f"[epoch {ep:04d}/{args.epochs}] lr={sched.get_last_lr()[0]:.2e} " + " ".join(f"{k}={v:.6f}" for k,v in meters.items()), flush=True)
        if meters["loss"] < best:
            best = meters["loss"]
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / "best.pt")
        if ep % 100 == 0 or ep == args.epochs:
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / f"epoch_{ep:04d}.pt")
    torch.save({"model": model.state_dict(), "config": config, "epoch": args.epochs}, ckpt_dir / "final.pt")
    print(f"saved_best: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
