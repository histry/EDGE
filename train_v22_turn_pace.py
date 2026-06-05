#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train V22 learned turn-aware temporal pace refiner."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from model.v22_turn_pace import V22TurnPaceRefiner
from tools.v22_turn_utils import torch_root_yaw_velocity_dps

EDIT_IDXS = torch.tensor([5] + list(range(7, 151)), dtype=torch.long)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(value)
    return (value * expanded).sum() / (expanded.sum() + 1e-8)


def group_split(source_ids: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    unique = np.unique(source_ids)
    rng = np.random.default_rng(seed)
    unique = unique.copy()
    rng.shuffle(unique)
    val_count = max(1, int(round(len(unique) * val_ratio)))
    val_sources = set(int(x) for x in unique[:val_count].tolist())
    train_sources = set(int(x) for x in unique[val_count:].tolist())
    train_idx = np.asarray([i for i, value in enumerate(source_ids) if int(value) in train_sources], dtype=np.int64)
    val_idx = np.asarray([i for i, value in enumerate(source_ids) if int(value) in val_sources], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Source-group split produced an empty train or val set")
    return train_idx, val_idx, sorted(train_sources), sorted(val_sources)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=2e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--residual_scale", type=float, default=0.22)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--patience", type=int, default=70)
    ap.add_argument("--lambda_recon", type=float, default=1.0)
    ap.add_argument("--lambda_velocity", type=float, default=0.80)
    ap.add_argument("--lambda_acceleration", type=float, default=0.30)
    ap.add_argument("--lambda_yaw", type=float, default=0.65)
    ap.add_argument("--lambda_peak", type=float, default=0.30)
    ap.add_argument("--lambda_unmasked", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=20260605)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    z = np.load(args.data, allow_pickle=True)
    required = ("corrupted", "target", "edit_mask", "condition", "source_id")
    for key in required:
        if key not in z.files:
            raise RuntimeError(f"Dataset missing array: {key}")

    corrupted_np = np.asarray(z["corrupted"], dtype=np.float32)
    target_np = np.asarray(z["target"], dtype=np.float32)
    mask_np = np.asarray(z["edit_mask"], dtype=np.float32)
    condition_np = np.asarray(z["condition"], dtype=np.float32)
    source_id_np = np.asarray(z["source_id"], dtype=np.int32)
    if corrupted_np.shape != target_np.shape or corrupted_np.ndim != 3 or corrupted_np.shape[-1] != 151:
        raise ValueError(f"Invalid motion arrays: corrupted={corrupted_np.shape}, target={target_np.shape}")
    if mask_np.shape != corrupted_np.shape[:2]:
        raise ValueError(f"Invalid mask shape: {mask_np.shape}")

    tensors = TensorDataset(
        torch.from_numpy(corrupted_np),
        torch.from_numpy(target_np),
        torch.from_numpy(mask_np),
        torch.from_numpy(condition_np),
    )
    train_idx, val_idx, train_sources, val_sources = group_split(source_id_np, args.val_ratio, args.seed)
    train_ds = Subset(tensors, train_idx.tolist())
    val_ds = Subset(tensors, val_idx.tolist())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V22TurnPaceRefiner(
        motion_dim=151,
        condition_dim=condition_np.shape[1],
        hidden_dim=args.hidden_dim,
        residual_scale=args.residual_scale,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    edit_idxs = EDIT_IDXS.to(device)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "motion_dim": 151,
        "condition_dim": int(condition_np.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "residual_scale": float(args.residual_scale),
        "dropout": float(args.dropout),
        "window_len": int(corrupted_np.shape[1]),
        "fps": 30.0,
    }
    split_report = {
        "num_samples": int(len(tensors)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_sources": train_sources,
        "val_sources": val_sources,
    }
    (out_dir / "split.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    def compute_losses(corrupted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, condition: torch.Tensor):
        pred = model(corrupted, mask, condition)
        pred_edit = pred.index_select(-1, edit_idxs)
        target_edit = target.index_select(-1, edit_idxs)
        corrupted_edit = corrupted.index_select(-1, edit_idxs)

        recon = masked_mean((pred_edit - target_edit) ** 2, mask)
        pvel = pred_edit[:, 1:] - pred_edit[:, :-1]
        tvel = target_edit[:, 1:] - target_edit[:, :-1]
        vmask = torch.maximum(mask[:, 1:], mask[:, :-1])
        velocity = masked_mean((pvel - tvel) ** 2, vmask)

        pacc = pvel[:, 1:] - pvel[:, :-1]
        tacc = tvel[:, 1:] - tvel[:, :-1]
        amask = torch.maximum(vmask[:, 1:], vmask[:, :-1])
        acceleration = masked_mean((pacc - tacc) ** 2, amask)

        pyaw = torch_root_yaw_velocity_dps(pred)
        tyaw = torch_root_yaw_velocity_dps(target)
        ymask = torch.maximum(mask[:, 1:], mask[:, :-1])
        yaw = masked_mean((pyaw - tyaw) ** 2 / (120.0**2), ymask)
        ppeak = torch.amax(torch.abs(pyaw) * ymask, dim=1)
        tpeak = torch.amax(torch.abs(tyaw) * ymask, dim=1)
        peak = F.smooth_l1_loss(ppeak / 120.0, tpeak / 120.0)

        unmasked = 1.0 - mask
        preserve = masked_mean((pred_edit - corrupted_edit) ** 2, unmasked)
        loss = (
            args.lambda_recon * recon
            + args.lambda_velocity * velocity
            + args.lambda_acceleration * acceleration
            + args.lambda_yaw * yaw
            + args.lambda_peak * peak
            + args.lambda_unmasked * preserve
        )
        return loss, {
            "recon": recon,
            "velocity": velocity,
            "acceleration": acceleration,
            "yaw": yaw,
            "peak": peak,
            "preserve": preserve,
        }

    def run_epoch(loader: DataLoader, training: bool) -> Dict[str, float]:
        model.train(training)
        totals: Dict[str, float] = {"loss": 0.0, "recon": 0.0, "velocity": 0.0, "acceleration": 0.0, "yaw": 0.0, "peak": 0.0, "preserve": 0.0}
        count = 0
        for corrupted, target, mask, condition in loader:
            corrupted = corrupted.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            condition = condition.to(device, non_blocking=True)
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, parts = compute_losses(corrupted, target, mask, condition)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            b = len(corrupted)
            totals["loss"] += float(loss.detach()) * b
            for key, value in parts.items():
                totals[key] += float(value.detach()) * b
            count += b
        return {key: value / max(count, 1) for key, value in totals.items()}

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    condition_lo = np.asarray(z["condition_lo"], dtype=np.float32) if "condition_lo" in z.files else np.zeros((condition_np.shape[1],), dtype=np.float32)
    condition_hi = np.asarray(z["condition_hi"], dtype=np.float32) if "condition_hi" in z.files else np.ones((condition_np.shape[1],), dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        scheduler.step()
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)

        if val_metrics["loss"] < best_val - 1e-7:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "condition_lo": condition_lo,
                    "condition_hi": condition_hi,
                    "split": split_report,
                },
                ckpt_dir / "best.pt",
            )
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} "
                f"val={val_metrics['loss']:.6f} yaw={val_metrics['yaw']:.6f} "
                f"peak={val_metrics['peak']:.6f} best={best_val:.6f}@{best_epoch}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop epoch={epoch} best={best_val:.6f}@{best_epoch}", flush=True)
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": history[-1]["epoch"],
            "val_loss": history[-1]["val"]["loss"],
            "condition_lo": condition_lo,
            "condition_hi": condition_hi,
            "split": split_report,
        },
        ckpt_dir / "final.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("best:", ckpt_dir / "best.pt")
    print("best_val:", best_val, "epoch:", best_epoch)


if __name__ == "__main__":
    main()
