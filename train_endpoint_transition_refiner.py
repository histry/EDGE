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

from model.endpoint_transition_refiner import EndpointTransitionRefiner, ROT, ROOT_X, ROOT_Z


class TransitionRefinerDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz found in {data_dir}")
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=True)
        return {k: torch.from_numpy(z[k].astype("float32")) for k in ["rough", "target", "valid_mask", "exit_pose", "entry_pose", "music_event"]}


def losses(pred, residual, batch, lambdas):
    target = batch["target"]; rough = batch["rough"]; mask = batch["valid_mask"]
    ch = torch.ones(pred.shape[-1], device=pred.device, dtype=pred.dtype)
    ch[0:4] = 0.0; ch[ROOT_X] = 0.0; ch[ROOT_Z] = 0.0; ch[5] = 0.35; ch[7:] = 1.0
    ch = ch.view(1,1,-1)
    rec = (F.smooth_l1_loss(pred * ch, target * ch, reduction="none") * mask).sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    if pred.shape[1] > 1:
        vel_p = pred[:,1:] - pred[:,:-1]; vel_t = target[:,1:] - target[:,:-1]; vm = mask[:,1:]
        vel = (F.smooth_l1_loss(vel_p * ch, vel_t * ch, reduction="none") * vm).sum() / (vm.sum() * pred.shape[-1]).clamp_min(1.0)
    else:
        vel = pred.new_tensor(0.0)
    if pred.shape[1] > 2:
        acc_p = pred[:,2:] - 2*pred[:,1:-1] + pred[:,:-2]
        acc_t = target[:,2:] - 2*target[:,1:-1] + target[:,:-2]
        am = mask[:,2:]
        acc = (F.smooth_l1_loss(acc_p[:,:,7:], acc_t[:,:,7:], reduction="none") * am).sum() / (am.sum() * (pred.shape[-1]-7)).clamp_min(1.0)
    else:
        acc = pred.new_tensor(0.0)
    endpoint = F.smooth_l1_loss(pred[:,0], batch["exit_pose"]) + F.smooth_l1_loss(pred[:,-1], batch["entry_pose"])
    preserve = (F.smooth_l1_loss(pred * ch, rough * ch, reduction="none") * (1.0 - mask * 0.5)).mean()
    residual_reg = torch.abs(residual[:,:,7:]).mean()
    root = pred[:,:,ROOT_X].abs().mean() + pred[:,:,ROOT_Z].abs().mean()
    total = lambdas["rec"]*rec + lambdas["vel"]*vel + lambdas["acc"]*acc + lambdas["endpoint"]*endpoint + lambdas["preserve"]*preserve + lambdas["residual"]*residual_reg + lambdas["root"]*root
    return {"total": total, "rec": rec.detach(), "vel": vel.detach(), "acc": acc.detach(), "endpoint": endpoint.detach(), "preserve": preserve.detach(), "residual": residual_reg.detach(), "root": root.detach()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=800); ap.add_argument("--batch_size", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--hidden_dim", type=int, default=256); ap.add_argument("--num_layers", type=int, default=4); ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--lambda_rec", type=float, default=1.0); ap.add_argument("--lambda_endpoint", type=float, default=2.0); ap.add_argument("--lambda_vel", type=float, default=0.8); ap.add_argument("--lambda_acc", type=float, default=0.2); ap.add_argument("--lambda_style_preserve", type=float, default=0.35); ap.add_argument("--lambda_residual", type=float, default=0.03); ap.add_argument("--lambda_root", type=float, default=5.0)
    ap.add_argument("--num_workers", type=int, default=2); ap.add_argument("--save_every", type=int, default=100)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True); ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    ds = TransitionRefinerDataset(args.data_dir); dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EndpointTransitionRefiner(hidden_dim=args.hidden_dim, num_layers=args.num_layers, num_heads=args.num_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs,1), eta_min=args.lr*0.05)
    lambdas = {"rec": args.lambda_rec, "endpoint": args.lambda_endpoint, "vel": args.lambda_vel, "acc": args.lambda_acc, "preserve": args.lambda_style_preserve, "residual": args.lambda_residual, "root": args.lambda_root}
    config = vars(args); config["lambdas"] = lambdas; config["dataset_size"] = len(ds)
    (out_dir / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    best = 1e9
    for ep in range(1, args.epochs+1):
        model.train(); meters = {k:0.0 for k in ["total","rec","vel","acc","endpoint","preserve","residual","root"]}; n=0
        for batch in dl:
            batch = {k:v.to(device) for k,v in batch.items()}
            pred, residual = model(batch["rough"], batch["exit_pose"], batch["entry_pose"], batch["music_event"], batch["valid_mask"], return_residual=True)
            loss = losses(pred, residual, batch, lambdas)
            opt.zero_grad(set_to_none=True); loss["total"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            bs = batch["rough"].shape[0]; n += bs
            for k in meters: meters[k] += float(loss[k].detach().cpu()) * bs
        sched.step(); meters = {k:v/max(n,1) for k,v in meters.items()}
        print(f"[epoch {ep:04d}/{args.epochs}] lr={sched.get_last_lr()[0]:.2e} " + " ".join(f"{k}={v:.6f}" for k,v in meters.items()), flush=True)
        if meters["total"] < best:
            best = meters["total"]; torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / "best.pt")
        if ep % args.save_every == 0 or ep == args.epochs:
            torch.save({"model": model.state_dict(), "config": config, "epoch": ep, "loss": meters}, ckpt_dir / f"epoch_{ep:04d}.pt")
    torch.save({"model": model.state_dict(), "config": config, "epoch": args.epochs}, ckpt_dir / "final.pt")
    print(f"saved_best: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
