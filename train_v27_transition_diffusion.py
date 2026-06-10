#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the V29 temporal transition diffusion model.

The historical filename is retained so existing experiment scripts can be
upgraded by replacement.  V29 adds:
  * source-group-disjoint train/validation splits;
  * EMA inference weights;
  * temporal dilated attention denoising;
  * SO(3), velocity, acceleration, jerk, FK and endpoint losses;
  * deterministic validation noise for meaningful checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.v27_transition_diffusion import (
    TemporalTransitionDenoiser,
    _linear_beta_schedule,
)
from tools.v29_motion_geometry import (
    CONTACT,
    ROOT,
    ROOT_Y,
    ROT,
    angular_velocity_torch,
    geodesic_rotation_error_torch,
    motion_to_joint_positions_torch,
    project_motion_rotations_torch,
)


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
        self.sample_kind = (
            np.asarray(data["sample_kind"], dtype=object)
            if "sample_kind" in data.files
            else np.asarray(["unknown"] * len(self.target), dtype=object)
        )
        self.start_group = (
            np.asarray(data["start_group"], dtype=object)
            if "start_group" in data.files
            else np.asarray([f"sample:{i}" for i in range(len(self.target))], dtype=object)
        )
        self.end_group = (
            np.asarray(data["end_group"], dtype=object)
            if "end_group" in data.files
            else self.start_group.copy()
        )
        self.meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}

    def __len__(self) -> int:
        return int(len(self.target))

    def __getitem__(self, idx: int):
        length = int(self.length[idx])
        start_velocity = (
            self.target[idx, 0] - self.start[idx]
            if length > 0 else np.zeros_like(self.start[idx])
        )
        end_velocity = (
            self.end[idx] - self.target[idx, length - 1]
            if length > 0 else np.zeros_like(self.end[idx])
        )
        return {
            "target": self.target[idx],
            "mask": self.mask[idx],
            "start": self.start[idx],
            "end": self.end[idx],
            "music": self.music[idx],
            "length": self.length[idx],
            "sample_weight": self.sample_weight[idx],
            "start_velocity": start_velocity.astype(np.float32),
            "end_velocity": end_velocity.astype(np.float32),
            "index": idx,
        }


def source_disjoint_split(
    dataset: TransitionDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    all_groups = sorted(
        set(str(v) for v in dataset.start_group)
        | set(str(v) for v in dataset.end_group)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(all_groups)
    n_val_groups = max(1, int(round(len(all_groups) * val_ratio)))
    val_groups = set(all_groups[:n_val_groups])
    train_groups = set(all_groups[n_val_groups:])

    train_indices = []
    val_indices = []
    dropped_cross = []
    for i, (a, b) in enumerate(zip(dataset.start_group, dataset.end_group)):
        a, b = str(a), str(b)
        if a in val_groups and b in val_groups:
            val_indices.append(i)
        elif a in train_groups and b in train_groups:
            train_indices.append(i)
        else:
            dropped_cross.append(i)

    if len(train_indices) == 0 or len(val_indices) == 0:
        # Conservative fallback: split by primary group while still keeping
        # every identical primary group on one side.
        train_indices, val_indices = [], []
        for i, a in enumerate(dataset.start_group):
            (val_indices if str(a) in val_groups else train_indices).append(i)
        dropped_cross = []

    if len(train_indices) == 0 or len(val_indices) == 0:
        raise RuntimeError(
            "Unable to form a non-empty source-disjoint split. "
            "Increase dataset size or reduce --val_ratio."
        )
    meta = {
        "num_groups": len(all_groups),
        "num_train_groups": len(train_groups),
        "num_val_groups": len(val_groups),
        "num_train_samples": len(train_indices),
        "num_val_samples": len(val_indices),
        "num_cross_group_dropped": len(dropped_cross),
        "val_groups": sorted(val_groups),
    }
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(val_indices, dtype=np.int64),
        meta,
    )


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for key, value in state.items():
            if not torch.is_floating_point(value):
                self.shadow[key].copy_(value)
            else:
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}


def _masked_per_sample(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values).to(values.dtype)
    dims = tuple(range(1, values.ndim))
    return (values * expanded).sum(dim=dims) / expanded.sum(dim=dims).clamp_min(1.0)


def _weighted_mean(
    per_sample: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    weight = sample_weight.reshape(-1).clamp_min(1e-4)
    return (per_sample * weight).sum() / weight.sum().clamp_min(1e-4)


def _difference_mask(mask: torch.Tensor, order: int) -> torch.Tensor:
    result = mask
    for _ in range(order):
        result = result[:, 1:] * result[:, :-1]
    return result


def _diff(x: torch.Tensor, order: int) -> torch.Tensor:
    result = x
    for _ in range(order):
        result = result[:, 1:] - result[:, :-1]
    return result


def composite_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    diffusion_steps: int,
    weights: Dict[str, float],
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target = batch["target"].to(device)
    mask = batch["mask"].to(device)
    start = batch["start"].to(device)
    end = batch["end"].to(device)
    music = batch["music"].to(device)
    length = batch["length"].to(device).reshape(-1, 1)
    sample_weight = batch["sample_weight"].to(device).reshape(-1)
    start_velocity = batch["start_velocity"].to(device)
    end_velocity = batch["end_velocity"].to(device)
    b, k, _ = target.shape

    _, _, alpha_bar = _linear_beta_schedule(diffusion_steps, device)
    idx = torch.randint(
        0, diffusion_steps, (b,), device=device, generator=generator
    )
    noise = torch.randn(
        target.shape,
        device=device,
        dtype=target.dtype,
        generator=generator,
    )
    ab = alpha_bar[idx].reshape(b, 1, 1)
    noisy = torch.sqrt(ab) * target + torch.sqrt(1.0 - ab) * noise
    t = idx.float() / max(diffusion_steps - 1, 1)
    pos = torch.linspace(
        1.0 / (k + 1), k / (k + 1), k, device=device
    ).reshape(1, k, 1).expand(b, -1, -1)

    pred_noise = model(
        noisy,
        t,
        start,
        end,
        music,
        (length / float(k)).clamp(0.0, 1.0),
        pos,
        mask=mask,
        start_velocity=start_velocity,
        end_velocity=end_velocity,
    )
    noise_per = _masked_per_sample((pred_noise - noise) ** 2, mask)
    losses = {"noise": _weighted_mean(noise_per, sample_weight)}

    pred_x0 = (
        noisy - torch.sqrt(1.0 - ab) * pred_noise
    ) / torch.sqrt(ab).clamp_min(1e-6)
    pred_x0 = project_motion_rotations_torch(pred_x0)
    target_valid = project_motion_rotations_torch(target)

    geo = geodesic_rotation_error_torch(pred_x0, target_valid) ** 2
    losses["rotation"] = _weighted_mean(
        _masked_per_sample(geo, mask), sample_weight
    )

    root_values = (pred_x0[..., ROOT] - target_valid[..., ROOT]) ** 2
    losses["root"] = _weighted_mean(
        _masked_per_sample(root_values, mask), sample_weight
    )

    contact_values = (pred_x0[..., CONTACT] - target_valid[..., CONTACT]) ** 2
    losses["contact"] = _weighted_mean(
        _masked_per_sample(contact_values, mask), sample_weight
    )

    pred_pos = motion_to_joint_positions_torch(pred_x0)
    target_pos = motion_to_joint_positions_torch(target_valid)
    losses["fk"] = _weighted_mean(
        _masked_per_sample((pred_pos - target_pos) ** 2, mask),
        sample_weight,
    )

    for name, order in (("velocity", 1), ("acceleration", 2), ("jerk", 3)):
        if k <= order:
            losses[name] = pred_x0.new_tensor(0.0)
            continue
        dmask = _difference_mask(mask, order)
        pred_d = _diff(pred_pos, order)
        target_d = _diff(target_pos, order)
        losses[name] = _weighted_mean(
            _masked_per_sample(
                F.smooth_l1_loss(pred_d, target_d, reduction="none"),
                dmask,
            ),
            sample_weight,
        )

    angular_pred = angular_velocity_torch(pred_x0)
    angular_target = angular_velocity_torch(target_valid)
    if angular_pred.shape[-3] > 0:
        angular_mask = _difference_mask(mask, 1)
        losses["angular_velocity"] = _weighted_mean(
            _masked_per_sample(
                F.smooth_l1_loss(
                    angular_pred, angular_target, reduction="none"
                ),
                angular_mask,
            ),
            sample_weight,
        )
    else:
        losses["angular_velocity"] = pred_x0.new_tensor(0.0)

    lengths = mask.sum(dim=1).long().clamp_min(1)
    batch_indices = torch.arange(b, device=device)
    last_indices = lengths - 1
    first_pred = pred_x0[:, 0]
    last_pred = pred_x0[batch_indices, last_indices]
    endpoint_pose = (
        F.smooth_l1_loss(first_pred, target_valid[:, 0], reduction="none").mean(dim=-1)
        + F.smooth_l1_loss(
            last_pred, target_valid[batch_indices, last_indices], reduction="none"
        ).mean(dim=-1)
    )
    pred_start_vel = first_pred - start
    pred_end_vel = end - last_pred
    endpoint_velocity = (
        F.smooth_l1_loss(
            pred_start_vel, start_velocity, reduction="none"
        ).mean(dim=-1)
        + F.smooth_l1_loss(
            pred_end_vel, end_velocity, reduction="none"
        ).mean(dim=-1)
    )
    losses["endpoint"] = _weighted_mean(
        endpoint_pose + endpoint_velocity, sample_weight
    )

    total = pred_x0.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    device: torch.device,
    diffusion_steps: int,
    weights: Dict[str, float],
    amp: bool,
    grad_clip: float,
    eval_seed: int | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, list[float]] = {}
    scaler = torch.cuda.amp.GradScaler(enabled=amp and training and device.type == "cuda")
    generator = None
    if eval_seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(eval_seed)

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                loss, metrics = composite_loss(
                    model,
                    batch,
                    device,
                    diffusion_steps,
                    weights,
                    generator=generator,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(model)
            for name, value in metrics.items():
                totals.setdefault(name, []).append(value)
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=420)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_blocks", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=70)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)

    parser.add_argument("--w_noise", type=float, default=1.0)
    parser.add_argument("--w_rotation", type=float, default=0.30)
    parser.add_argument("--w_root", type=float, default=0.20)
    parser.add_argument("--w_contact", type=float, default=0.08)
    parser.add_argument("--w_fk", type=float, default=0.35)
    parser.add_argument("--w_velocity", type=float, default=0.45)
    parser.add_argument("--w_acceleration", type=float, default=0.22)
    parser.add_argument("--w_jerk", type=float, default=0.06)
    parser.add_argument("--w_angular_velocity", type=float, default=0.18)
    parser.add_argument("--w_endpoint", type=float, default=0.55)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = TransitionDataset(args.data)
    train_indices, val_indices, split_meta = source_disjoint_split(
        dataset, args.val_ratio, args.seed
    )
    train_ds = torch.utils.data.Subset(dataset, train_indices.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_indices.tolist())
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalTransitionDenoiser(
        motion_dim=dataset.target.shape[-1],
        music_dim=dataset.music.shape[-1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.08
    )
    ema = EMA(model, decay=args.ema_decay)

    weights = {
        "noise": args.w_noise,
        "rotation": args.w_rotation,
        "root": args.w_root,
        "contact": args.w_contact,
        "fk": args.w_fk,
        "velocity": args.w_velocity,
        "acceleration": args.w_acceleration,
        "jerk": args.w_jerk,
        "angular_velocity": args.w_angular_velocity,
        "endpoint": args.w_endpoint,
    }

    best = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, ema, device,
            args.diffusion_steps, weights, bool(args.amp),
            args.grad_clip,
        )
        val_metrics = run_epoch(
            model, val_loader, None, None, device,
            args.diffusion_steps, weights, False,
            args.grad_clip,
            eval_seed=args.seed + 9173,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": val_metrics,
            # Backward-compatible scalar fields.
            "train_loss": train_metrics.get("total", 0.0),
            "val_loss": val_metrics.get("total", 0.0),
        }
        history.append(row)
        print(
            f"[V29 transition] epoch={epoch:04d} "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"val_noise={val_metrics.get('noise', 0.0):.6f} "
            f"val_jerk={val_metrics.get('jerk', 0.0):.6f}",
            flush=True,
        )

        score = float(row["val_loss"])
        if score < best:
            best = score
            bad_epochs = 0
            config = {
                "architecture": "v29_temporal_dilated_attention",
                "motion_dim": int(dataset.target.shape[-1]),
                "music_dim": int(dataset.music.shape[-1]),
                "hidden_dim": args.hidden_dim,
                "num_blocks": args.num_blocks,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "diffusion_steps": args.diffusion_steps,
                "max_len": int(dataset.target.shape[1]),
                "loss_weights": weights,
                "dataset_meta": dataset.meta,
                "split_meta": split_meta,
                "seed": args.seed,
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema.state_dict(),
                    "config": config,
                    "best_val_loss": best,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                checkpoint_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}", flush=True)
                break

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "split.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best_path = checkpoint_dir / "best.pt"
    (out_dir / "BEST_V27_TRANSITION_DIFFUSION_CKPT.txt").write_text(
        str(best_path), encoding="utf-8"
    )
    (out_dir / "BEST_V29_TRANSITION_DIFFUSION_CKPT.txt").write_text(
        str(best_path), encoding="utf-8"
    )
    print(f"[SAVED] {best_path}")


if __name__ == "__main__":
    main()
