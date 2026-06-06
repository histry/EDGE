#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
            "corrupted",
            "target",
            "edit_mask",
            "condition",
            "target_tau",
            "target_duration_frames",
            "turn_input_start",
            "turn_input_end",
            "is_identity",
            "duration_bin",
        )

    def __len__(self) -> int:
        return len(self.arrays["corrupted"])

    def __getitem__(self, index: int):
        values = []
        for key in self.keys:
            array = np.asarray(self.arrays[key][index])
            if key == "duration_bin":
                values.append(torch.tensor(array, dtype=torch.long))
            else:
                values.append(torch.from_numpy(np.asarray(array, dtype=np.float32)))
        return tuple(values)


def group_split(source_id: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(source_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_ratio)))
    val_sources = set(unique[:n_val].tolist())
    val = np.asarray([i for i, source in enumerate(source_id) if int(source) in val_sources], dtype=np.int64)
    train = np.asarray([i for i, source in enumerate(source_id) if int(source) not in val_sources], dtype=np.int64)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("Invalid source-level split")
    return train, val


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    feature_multiplier = 1
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
        feature_multiplier *= value.shape[-1]
    return (value * mask).sum() / (mask.sum() * feature_multiplier + 1e-8)


def per_sample_masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    feature_multiplier = 1
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
        feature_multiplier *= value.shape[-1]
    dims = tuple(range(1, value.ndim))
    numerator = (value * mask).sum(dim=dims)
    denominator = mask.sum(dim=dims) * feature_multiplier
    return numerator / denominator.clamp_min(1e-8)


def rotation_activity(motion: torch.Tensor) -> torch.Tensor:
    if motion.shape[1] < 2:
        return torch.zeros((motion.shape[0],), device=motion.device, dtype=motion.dtype)
    velocity = motion[:, 1:, 7:151] - motion[:, :-1, 7:151]
    return torch.linalg.vector_norm(velocity, dim=-1).mean(dim=1)


def duration_rank_loss(predicted: torch.Tensor, target: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    if len(predicted) < 2:
        return predicted.sum() * 0.0
    permutation = torch.randperm(len(predicted), device=predicted.device)
    pred_delta = predicted - predicted[permutation]
    target_delta = target - target[permutation]
    valid = torch.abs(target_delta) >= 3.0
    if not torch.any(valid):
        return predicted.sum() * 0.0
    sign = torch.sign(target_delta[valid])
    normalized_delta = pred_delta[valid] / 72.0
    return F.relu(float(margin) - sign * normalized_delta).mean()


def safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum())).clamp_min(1e-8)
    return (x_centered * y_centered).sum() / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--duration_min_frames", type=float, default=8.0)
    parser.add_argument("--duration_max_frames", type=float, default=56.0)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--balanced_sampler", type=int, default=1)

    parser.add_argument("--lambda_tau", type=float, default=2.0)
    parser.add_argument("--lambda_duration", type=float, default=1.2)
    parser.add_argument("--lambda_duration_linear", type=float, default=0.35)
    parser.add_argument("--lambda_duration_rank", type=float, default=0.15)
    parser.add_argument("--lambda_duration_consistency", type=float, default=0.8)
    parser.add_argument("--lambda_motion", type=float, default=0.8)
    parser.add_argument("--lambda_context", type=float, default=0.35)
    parser.add_argument("--lambda_velocity", type=float, default=0.30)
    parser.add_argument("--lambda_activity", type=float, default=0.20)
    parser.add_argument("--lambda_yaw", type=float, default=0.35)
    parser.add_argument("--lambda_peak_yaw", type=float, default=0.12)
    parser.add_argument("--lambda_smooth", type=float, default=0.06)
    parser.add_argument("--lambda_identity_tau", type=float, default=0.45)
    parser.add_argument("--lambda_identity_motion", type=float, default=0.35)
    parser.add_argument("--lambda_edit", type=float, default=0.15)
    parser.add_argument("--lambda_endpoint", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.data, allow_pickle=True) as archive:
        required = (
            "corrupted",
            "target",
            "edit_mask",
            "condition",
            "target_tau",
            "source_id",
            "target_duration_frames",
            "turn_input_start",
            "turn_input_end",
            "speed_factor",
            "is_identity",
            "duration_bin",
        )
        for key in required:
            if key not in archive.files:
                raise RuntimeError(f"Dataset missing {key}")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    n, time_steps, motion_dim = arrays["corrupted"].shape
    if motion_dim != 151 or arrays["target"].shape != (n, time_steps, motion_dim):
        raise ValueError("Invalid motion arrays")
    if arrays["condition"].shape[1] != 17:
        raise ValueError(f"V23-v2 expects 17D inference-safe condition, got {arrays['condition'].shape}")

    dataset = V23Dataset(arrays)
    train_indices, val_indices = group_split(np.asarray(arrays["source_id"]), args.val_ratio, args.seed)

    train_sampler = None
    train_shuffle = True
    if bool(args.balanced_sampler):
        train_bins = np.asarray(arrays["duration_bin"])[train_indices].astype(np.int64)
        train_identity = (np.asarray(arrays["is_identity"])[train_indices] > 0.5).astype(np.int64)
        class_id = train_identity * (int(train_bins.max()) + 1) + train_bins
        counts = np.bincount(class_id, minlength=int(class_id.max()) + 1).astype(np.float64)
        weights = 1.0 / np.sqrt(np.maximum(counts[class_id], 1.0))
        weights = weights / max(weights.mean(), 1e-8)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_indices),
            replacement=True,
        )
        train_shuffle = False

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=train_shuffle if train_sampler is None else False,
        sampler=train_sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = V23MonotonicDurationNet(
        motion_dim=151,
        condition_dim=int(arrays["condition"].shape[1]),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        duration_min_frames=args.duration_min_frames,
        duration_max_frames=args.duration_max_frames,
        window_len=time_steps,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )
    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "samples": int(n),
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "train_sources": int(len(np.unique(arrays["source_id"][train_indices]))),
        "val_sources": int(len(np.unique(arrays["source_id"][val_indices]))),
        "train_identity_ratio": float(np.mean(arrays["is_identity"][train_indices] > 0.5)),
        "val_identity_ratio": float(np.mean(arrays["is_identity"][val_indices] > 0.5)),
    }
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    def compute_losses(batch):
        (
            corrupted,
            target,
            mask,
            condition,
            target_tau,
            target_duration,
            turn_start,
            turn_end,
            is_identity,
            duration_bin,
        ) = [value.to(device, non_blocking=True) for value in batch]
        del duration_bin

        result = model(corrupted, mask, condition)
        tau = result["tau"]
        predicted_duration = result["duration_frames"]
        predicted_motion = warp_motion_so3(corrupted, tau)

        tau_weight = 0.20 + 1.80 * mask
        tau_loss = ((tau - target_tau).abs() * tau_weight).sum() / tau_weight.sum().clamp_min(1.0)

        duration_loss = F.smooth_l1_loss(
            torch.log(predicted_duration.clamp_min(1.0)),
            torch.log(target_duration.clamp_min(1.0)),
        )
        duration_linear_loss = F.smooth_l1_loss(
            predicted_duration / float(time_steps),
            target_duration / float(time_steps),
        )
        rank_loss = duration_rank_loss(predicted_duration, target_duration)

        soft_duration = soft_turn_duration_ratio(tau, turn_start, turn_end) * float(time_steps)
        duration_consistency = (
            F.smooth_l1_loss(soft_duration / float(time_steps), target_duration / float(time_steps))
            + 0.5 * F.smooth_l1_loss(predicted_duration / float(time_steps), soft_duration.detach() / float(time_steps))
        )

        motion_weight = 0.20 + 1.80 * mask
        motion_loss = masked_mean((predicted_motion[..., 4:151] - target[..., 4:151]) ** 2, motion_weight)
        outside = (1.0 - mask).clamp(0.0, 1.0)
        context_loss = masked_mean(
            F.smooth_l1_loss(predicted_motion[..., 4:151], target[..., 4:151], reduction="none"),
            0.10 + outside,
        )

        predicted_velocity = predicted_motion[:, 1:, 7:151] - predicted_motion[:, :-1, 7:151]
        target_velocity = target[:, 1:, 7:151] - target[:, :-1, 7:151]
        velocity_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
        velocity_loss = masked_mean(
            F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none"),
            0.20 + velocity_mask,
        )

        predicted_activity = rotation_activity(predicted_motion)
        target_activity = rotation_activity(target)
        activity_loss = F.smooth_l1_loss(
            torch.log1p(predicted_activity),
            torch.log1p(target_activity),
        )

        predicted_yaw = root_yaw_velocity_dps(predicted_motion)
        target_yaw = root_yaw_velocity_dps(target)
        yaw_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
        yaw_loss = masked_mean(
            F.smooth_l1_loss(predicted_yaw / 180.0, target_yaw / 180.0, reduction="none"),
            0.15 + yaw_mask,
        )
        predicted_peak = torch.quantile(torch.abs(predicted_yaw), 0.95, dim=1)
        target_peak = torch.quantile(torch.abs(target_yaw), 0.95, dim=1)
        peak_yaw_loss = F.smooth_l1_loss(predicted_peak / 180.0, target_peak / 180.0)

        increments = tau[:, 1:] - tau[:, :-1]
        increment_smoothness = torch.mean((increments[:, 1:] - increments[:, :-1]) ** 2)

        uniform_tau = torch.linspace(0.0, 1.0, time_steps, device=device, dtype=tau.dtype)[None]
        identity_rows = is_identity > 0.5
        identity_count = identity_rows.float().sum().clamp_min(1.0)
        identity_tau_per_sample = torch.abs(tau - uniform_tau).mean(dim=1)
        identity_tau_loss = (identity_tau_per_sample * identity_rows.float()).sum() / identity_count
        identity_motion_per_sample = torch.abs(predicted_motion[..., 4:151] - corrupted[..., 4:151]).mean(dim=(1, 2))
        identity_motion_loss = (identity_motion_per_sample * identity_rows.float()).sum() / identity_count

        edit_target = 1.0 - is_identity.clamp(0.0, 1.0)
        edit_loss = F.binary_cross_entropy_with_logits(result["edit_logit"], edit_target)
        endpoint_loss = torch.mean(torch.abs(tau[:, 0]) + torch.abs(tau[:, -1] - 1.0))

        total = (
            args.lambda_tau * tau_loss
            + args.lambda_duration * duration_loss
            + args.lambda_duration_linear * duration_linear_loss
            + args.lambda_duration_rank * rank_loss
            + args.lambda_duration_consistency * duration_consistency
            + args.lambda_motion * motion_loss
            + args.lambda_context * context_loss
            + args.lambda_velocity * velocity_loss
            + args.lambda_activity * activity_loss
            + args.lambda_yaw * yaw_loss
            + args.lambda_peak_yaw * peak_yaw_loss
            + args.lambda_smooth * increment_smoothness
            + args.lambda_identity_tau * identity_tau_loss
            + args.lambda_identity_motion * identity_motion_loss
            + args.lambda_edit * edit_loss
            + args.lambda_endpoint * endpoint_loss
        )

        duration_mae = torch.mean(torch.abs(predicted_duration - target_duration))
        rare_rows = target_duration < (args.duration_max_frames - 2.0)
        rare_count = rare_rows.float().sum().clamp_min(1.0)
        rare_duration_mae = (
            torch.abs(predicted_duration - target_duration) * rare_rows.float()
        ).sum() / rare_count
        duration_correlation = safe_corrcoef(predicted_duration, target_duration)
        activity_ratio = torch.mean(predicted_activity / target_activity.clamp_min(1e-6))
        identity_tau_metric = (identity_tau_per_sample * identity_rows.float()).sum() / identity_count
        identity_motion_metric = (identity_motion_per_sample * identity_rows.float()).sum() / identity_count
        edit_accuracy = ((torch.sigmoid(result["edit_logit"]) >= 0.5) == (edit_target >= 0.5)).float().mean()
        outside_drift = per_sample_masked_mean(
            torch.abs(predicted_motion[..., 4:151] - target[..., 4:151]),
            outside,
        ).mean()

        parts = {
            "loss": total,
            "tau": tau_loss,
            "duration": duration_loss,
            "duration_linear": duration_linear_loss,
            "duration_rank": rank_loss,
            "duration_consistency": duration_consistency,
            "motion": motion_loss,
            "context": context_loss,
            "velocity": velocity_loss,
            "activity": activity_loss,
            "yaw": yaw_loss,
            "peak_yaw": peak_yaw_loss,
            "smooth": increment_smoothness,
            "identity_tau": identity_tau_loss,
            "identity_motion": identity_motion_loss,
            "edit": edit_loss,
            "endpoint": endpoint_loss,
            "duration_mae_frames": duration_mae,
            "rare_duration_mae_frames": rare_duration_mae,
            "duration_correlation": duration_correlation,
            "tau_mae": torch.mean(torch.abs(tau - target_tau)),
            "activity_ratio": activity_ratio,
            "identity_tau_mae": identity_tau_metric,
            "identity_motion_drift": identity_motion_metric,
            "edit_accuracy": edit_accuracy,
            "outside_target_drift": outside_drift,
        }
        return total, parts

    def run_epoch(loader, training: bool) -> Dict[str, float]:
        model.train(training)
        totals: Dict[str, float] = {}
        count = 0
        for batch in loader:
            with torch.set_grad_enabled(training):
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, parts = compute_losses(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
            batch_size = len(batch[0])
            count += batch_size
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * batch_size
        return {key: value / max(count, 1) for key, value in totals.items()}

    config = {
        "version": "v23_v2_natural_duration",
        "motion_dim": 151,
        "condition_dim": int(arrays["condition"].shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "window_len": time_steps,
        "fps": 30.0,
        "duration_min_frames": args.duration_min_frames,
        "duration_max_frames": args.duration_max_frames,
    }
    best_score = float("inf")
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(train_loader, True)
        val_metrics = run_epoch(val_loader, False)
        scheduler.step()

        # Selection favors real duration generalization and safe identity behavior,
        # not merely the weighted training loss.
        selection_score = (
            val_metrics["loss"]
            + 0.015 * val_metrics["duration_mae_frames"] / float(time_steps)
            + 0.020 * val_metrics["rare_duration_mae_frames"] / float(time_steps)
            + 0.050 * abs(1.0 - val_metrics["activity_ratio"])
            + 0.100 * val_metrics["identity_tau_mae"]
            + 0.100 * val_metrics["identity_motion_drift"]
            + 0.020 * max(0.0, 0.85 - val_metrics["edit_accuracy"])
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_score": selection_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if selection_score < best_score - 1e-7:
            best_score = selection_score
            best_val = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "selection_score": best_score,
                    "val_metrics": val_metrics,
                    "split": split,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} train={train_metrics['loss']:.6f} val={val_metrics['loss']:.6f} "
                f"score={selection_score:.6f} dur={val_metrics['duration_mae_frames']:.3f} "
                f"rare={val_metrics['rare_duration_mae_frames']:.3f} corr={val_metrics['duration_correlation']:.3f} "
                f"tau={val_metrics['tau_mae']:.5f} activity={val_metrics['activity_ratio']:.3f} "
                f"id_tau={val_metrics['identity_tau_mae']:.5f} edit={val_metrics['edit_accuracy']:.3f} "
                f"best={best_score:.6f}@{best_epoch}",
                flush=True,
            )
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop epoch={epoch} best={best_score:.6f}@{best_epoch}", flush=True)
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": history[-1]["epoch"],
            "val_loss": history[-1]["val"]["loss"],
            "selection_score": history[-1]["selection_score"],
            "val_metrics": history[-1]["val"],
            "split": split,
        },
        checkpoint_dir / "final.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "BEST_V23_CKPT.txt").write_text(str((checkpoint_dir / "best.pt").resolve()) + "\n", encoding="utf-8")
    print("best:", checkpoint_dir / "best.pt")
    print("best val loss:", best_val, "selection score:", best_score, "epoch:", best_epoch)


if __name__ == "__main__":
    main()
