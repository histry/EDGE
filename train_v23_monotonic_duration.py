#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.v23_monotonic_duration import (
    V23MonotonicDurationNet,
    root_yaw_velocity_dps,
    soft_turn_duration_ratio,
    warp_motion_so3,
)


class V23Dataset(Dataset):
    def __init__(self, arrays: Dict[str, np.ndarray]):
        self.arrays = arrays
        self.keys = (
            "corrupted", "target", "edit_mask", "condition", "target_tau",
            "target_duration_frames", "turn_input_start", "turn_input_end",
        )

    def __len__(self) -> int:
        return len(self.arrays["corrupted"])

    def __getitem__(self, index: int):
        return tuple(torch.from_numpy(np.asarray(self.arrays[k][index], dtype=np.float32)) for k in self.keys)


def group_split(source_id: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(source_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_sources = set(unique[:n_val].tolist())
    val = np.asarray([i for i, s in enumerate(source_id) if int(s) in val_sources], dtype=np.int64)
    train = np.asarray([i for i, s in enumerate(source_id) if int(s) not in val_sources], dtype=np.int64)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Invalid group split")
    return train, val


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    feature_multiplier = 1
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
        feature_multiplier *= value.shape[-1]
    return (value * mask).sum() / (mask.sum() * feature_multiplier + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=700)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--lambda_tau", type=float, default=2.0)
    ap.add_argument("--lambda_duration", type=float, default=1.2)
    ap.add_argument("--lambda_duration_consistency", type=float, default=0.8)
    ap.add_argument("--lambda_motion", type=float, default=0.8)
    ap.add_argument("--lambda_yaw", type=float, default=0.35)
    ap.add_argument("--lambda_smooth", type=float, default=0.08)
    ap.add_argument("--lambda_identity", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.data, allow_pickle=True) as z:
        required = (
            "corrupted", "target", "edit_mask", "condition", "target_tau", "source_id",
            "target_duration_frames", "turn_input_start", "turn_input_end", "speed_factor",
        )
        for key in required:
            if key not in z.files:
                raise RuntimeError(f"Dataset missing {key}")
        arrays = {key: np.asarray(z[key]) for key in z.files}

    n, t, d = arrays["corrupted"].shape
    if d != 151 or arrays["target"].shape != (n, t, d):
        raise ValueError("Invalid motion arrays")
    dataset = V23Dataset(arrays)
    train_idx, val_idx = group_split(np.asarray(arrays["source_id"]), args.val_ratio, args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()), batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()), batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V23MonotonicDurationNet(
        motion_dim=151,
        condition_dim=int(arrays["condition"].shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "samples": n,
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_sources": int(len(np.unique(arrays["source_id"][train_idx]))),
        "val_sources": int(len(np.unique(arrays["source_id"][val_idx]))),
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    def losses(batch):
        corrupted, target, mask, condition, target_tau, target_duration, turn_start, turn_end = [x.to(device, non_blocking=True) for x in batch]
        result = model(corrupted, mask, condition)
        tau = result["tau"]
        pred_duration_ratio = result["duration_ratio"]
        pred_motion = warp_motion_so3(corrupted, tau)

        tau_weight = 0.25 + mask
        tau_loss = ((tau - target_tau).abs() * tau_weight).sum() / tau_weight.sum().clamp_min(1.0)
        target_duration_ratio = target_duration / float(t)
        duration_loss = F.smooth_l1_loss(pred_duration_ratio, target_duration_ratio)
        soft_duration = soft_turn_duration_ratio(tau, turn_start, turn_end)
        duration_consistency = (
            F.smooth_l1_loss(soft_duration, target_duration_ratio)
            + 0.5 * F.smooth_l1_loss(pred_duration_ratio, soft_duration.detach())
        )

        motion_mask = 0.15 + mask
        motion_loss = masked_mean((pred_motion[..., 4:151] - target[..., 4:151]) ** 2, motion_mask)
        pyaw = root_yaw_velocity_dps(pred_motion)
        tyaw = root_yaw_velocity_dps(target)
        ymask = torch.maximum(mask[:, 1:], mask[:, :-1])
        yaw_loss = masked_mean(F.smooth_l1_loss(pyaw / 180.0, tyaw / 180.0, reduction="none"), ymask)

        increments = tau[:, 1:] - tau[:, :-1]
        smooth_loss = torch.mean((increments[:, 1:] - increments[:, :-1]) ** 2)
        # Identity supervision is identified by target_tau being almost uniform.
        uniform = torch.linspace(0.0, 1.0, t, device=device, dtype=tau.dtype)[None]
        identity_rows = (torch.mean(torch.abs(target_tau - uniform), dim=1) < 1e-4).to(tau.dtype)
        identity_loss = ((torch.abs(tau - uniform).mean(dim=1)) * identity_rows).sum() / identity_rows.sum().clamp_min(1.0)

        total = (
            args.lambda_tau * tau_loss
            + args.lambda_duration * duration_loss
            + args.lambda_duration_consistency * duration_consistency
            + args.lambda_motion * motion_loss
            + args.lambda_yaw * yaw_loss
            + args.lambda_smooth * smooth_loss
            + args.lambda_identity * identity_loss
        )
        parts = {
            "loss": total, "tau": tau_loss, "duration": duration_loss,
            "duration_consistency": duration_consistency, "motion": motion_loss,
            "yaw": yaw_loss, "smooth": smooth_loss, "identity": identity_loss,
            "duration_mae_frames": torch.mean(torch.abs(pred_duration_ratio * float(t) - target_duration)),
            "tau_mae": torch.mean(torch.abs(tau - target_tau)),
        }
        return total, parts

    def run_epoch(loader, training: bool) -> Dict[str, float]:
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        for batch in loader:
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, parts = losses(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            b = len(batch[0])
            count += b
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * b
        return {k: v / max(count, 1) for k, v in totals.items()}

    config = {
        "motion_dim": 151,
        "condition_dim": int(arrays["condition"].shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "window_len": t,
        "fps": 30.0,
    }
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]})
        score = val_metrics["loss"]
        if score < best_val - 1e-7:
            best_val, best_epoch, stale = score, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(), "config": config,
                    "epoch": epoch, "val_loss": best_val, "split": split,
                },
                ckpt_dir / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} val={score:.6f} "
                f"dur_mae={val_metrics['duration_mae_frames']:.3f} tau_mae={val_metrics['tau_mae']:.5f} "
                f"yaw={val_metrics['yaw']:.6f} best={best_val:.6f}@{best_epoch}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop epoch={epoch} best={best_val:.6f}@{best_epoch}", flush=True)
            break

    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "epoch": history[-1]["epoch"], "val_loss": history[-1]["val"]["loss"], "split": split},
        ckpt_dir / "final.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "BEST_V23_CKPT.txt").write_text(str((ckpt_dir / "best.pt").resolve()) + "\n", encoding="utf-8")
    print("best:", ckpt_dir / "best.pt")
    print("best_val:", best_val, "epoch:", best_epoch)


if __name__ == "__main__":
    main()
