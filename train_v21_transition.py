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

from model.v21_transition import TRANSITION_LENGTHS, V21EndpointTransitionRefiner, V21TransitionDurationPredictor


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return (value * mask).sum() / (mask.sum() * value.shape[-1] + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--dpn_hidden_dim", type=int, default=192)
    ap.add_argument("--refiner_hidden_dim", type=int, default=256)
    ap.add_argument("--residual_scale", type=float, default=0.18)
    ap.add_argument("--lambda_dpn", type=float, default=1.0)
    ap.add_argument("--lambda_recon", type=float, default=1.0)
    ap.add_argument("--lambda_velocity", type=float, default=0.6)
    ap.add_argument("--lambda_acceleration", type=float, default=0.15)
    ap.add_argument("--lambda_endpoint", type=float, default=2.0)
    ap.add_argument("--lambda_root", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    z = np.load(args.data, allow_pickle=True)
    tensors = [
        torch.from_numpy(np.asarray(z["rough"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["target"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["mask"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["start"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["end"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["music"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["dpn_features"], dtype=np.float32)),
        torch.from_numpy(np.asarray(z["dpn_label"], dtype=np.int64)),
    ]
    dataset = TensorDataset(*tensors)
    val_size = max(1, int(round(len(dataset) * 0.1)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dpn = V21TransitionDurationPredictor(
        input_dim=tensors[6].shape[1],
        hidden_dim=args.dpn_hidden_dim,
        num_classes=len(TRANSITION_LENGTHS),
    ).to(device)
    refiner = V21EndpointTransitionRefiner(
        motion_dim=151,
        music_dim=tensors[5].shape[1],
        hidden_dim=args.refiner_hidden_dim,
        residual_scale=args.residual_scale,
    ).to(device)
    params = list(dpn.parameters()) + list(refiner.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dpn_input_dim": int(tensors[6].shape[1]),
        "dpn_hidden_dim": args.dpn_hidden_dim,
        "refiner_hidden_dim": args.refiner_hidden_dim,
        "motion_dim": 151,
        "music_dim": int(tensors[5].shape[1]),
        "residual_scale": args.residual_scale,
        "transition_lengths": list(TRANSITION_LENGTHS),
        "dropout": 0.1,
    }

    def run_epoch(loader, training: bool):
        dpn.train(training)
        refiner.train(training)
        totals = {"loss": 0.0, "dpn": 0.0, "recon": 0.0}
        count = 0
        for rough, target, mask, start, end, music, dpn_feat, label in loader:
            rough, target, mask = rough.to(device), target.to(device), mask.to(device)
            start, end, music = start.to(device), end.to(device), music.to(device)
            dpn_feat, label = dpn_feat.to(device), label.to(device)
            with torch.set_grad_enabled(training):
                logits = dpn(dpn_feat)
                pred = refiner(rough, start, end, music, mask)
                dpn_loss = F.cross_entropy(logits, label)
                recon = masked_mean((pred - target) ** 2, mask)
                pvel = pred[:, 1:] - pred[:, :-1]
                tvel = target[:, 1:] - target[:, :-1]
                vmask = mask[:, 1:] * mask[:, :-1]
                vel_loss = masked_mean((pvel - tvel) ** 2, vmask)
                pacc = pvel[:, 1:] - pvel[:, :-1]
                tacc = tvel[:, 1:] - tvel[:, :-1]
                amask = vmask[:, 1:] * vmask[:, :-1]
                acc_loss = masked_mean((pacc - tacc) ** 2, amask)
                endpoint_loss = F.mse_loss(pred[:, 0], start) + F.mse_loss(pred[:, -1], end)
                root_loss = F.mse_loss(pred[..., [4, 6]], rough[..., [4, 6]])
                loss = (
                    args.lambda_dpn * dpn_loss
                    + args.lambda_recon * recon
                    + args.lambda_velocity * vel_loss
                    + args.lambda_acceleration * acc_loss
                    + args.lambda_endpoint * endpoint_loss
                    + args.lambda_root * root_loss
                )
                if training:
                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optim.step()
            b = len(rough)
            totals["loss"] += float(loss.detach()) * b
            totals["dpn"] += float(dpn_loss.detach()) * b
            totals["recon"] += float(recon.detach()) * b
            count += b
        return {k: v / max(count, 1) for k, v in totals.items()}

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "dpn_state_dict": dpn.state_dict(),
                    "refiner_state_dict": refiner.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "dpn_lo": np.asarray(z["dpn_lo"], dtype=np.float32),
                    "dpn_hi": np.asarray(z["dpn_hi"], dtype=np.float32),
                },
                ckpt_dir / "best.pt",
            )
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} dpn={val_metrics['dpn']:.6f} "
                f"recon={val_metrics['recon']:.6f} best={best_val:.6f}",
                flush=True,
            )

    torch.save(
        {
            "dpn_state_dict": dpn.state_dict(),
            "refiner_state_dict": refiner.state_dict(),
            "config": config,
            "epoch": args.epochs,
            "val_loss": history[-1]["val"]["loss"],
            "dpn_lo": np.asarray(z["dpn_lo"], dtype=np.float32),
            "dpn_hi": np.asarray(z["dpn_hi"], dtype=np.float32),
        },
        ckpt_dir / "final.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best:", ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
