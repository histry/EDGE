#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train EDGE V30 continuous SO(3) INR and latent diffusion.

The historical filename is retained for existing automation.  Training has two
explicit stages:

A. continuous transition autoencoding: encode a variable-length real/synthetic
   interval into a compact latent and reconstruct it through an arbitrary-time
   SO(3) residual INR;
B. conditional latent diffusion: freeze the INR autoencoder and learn the
   distribution of transition latents conditioned on endpoints, endpoint
   velocities, music/event semantics and requested duration.

Source-group-disjoint validation, deterministic validation noise, EMA weights,
FK dynamics, angular dynamics and spectral losses are included by default.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.v29_motion_geometry import (
    CONTACT,
    ROOT,
    angular_velocity_torch,
    geodesic_rotation_error_torch,
    motion_to_joint_positions_torch,
    project_motion_rotations_torch,
)
from tools.v30_continuous_inr import (
    V30ContinuousTransitionSystem,
    V30INRConfig,
    linear_beta_schedule,
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
        self.sample_weight = np.asarray(
            data["sample_weight"] if "sample_weight" in data.files else np.ones((len(self.target),)),
            dtype=np.float32,
        )
        self.sample_kind = np.asarray(
            data["sample_kind"] if "sample_kind" in data.files else ["unknown"] * len(self.target),
            dtype=object,
        )
        self.start_group = np.asarray(
            data["start_group"] if "start_group" in data.files else [f"sample:{i}" for i in range(len(self.target))],
            dtype=object,
        )
        self.end_group = np.asarray(
            data["end_group"] if "end_group" in data.files else self.start_group,
            dtype=object,
        )
        self.real_target = np.asarray(
            data["real_target"] if "real_target" in data.files else np.ones((len(self.target),), dtype=np.bool_),
            dtype=np.bool_,
        )
        self.meta = json.loads(str(data["meta"].item())) if "meta" in data.files else {}

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray | float | int]:
        length = max(1, int(self.length[index]))
        start_velocity = self.target[index, 0] - self.start[index]
        end_velocity = self.end[index] - self.target[index, length - 1]
        return {
            "target": self.target[index],
            "mask": self.mask[index],
            "start": self.start[index],
            "end": self.end[index],
            "music": self.music[index],
            "length": self.length[index],
            "sample_weight": self.sample_weight[index],
            "start_velocity": start_velocity.astype(np.float32),
            "end_velocity": end_velocity.astype(np.float32),
            "real_target": float(self.real_target[index]),
            "index": int(index),
        }


def source_disjoint_split(
    dataset: TransitionDataset,
    validation_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    groups = sorted(
        set(str(value) for value in dataset.start_group)
        | set(str(value) for value in dataset.end_group)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(1, int(round(len(groups) * validation_ratio)))
    validation_groups = set(groups[:validation_count])
    training_groups = set(groups[validation_count:])
    train, validation, cross = [], [], []
    for index, (left, right) in enumerate(zip(dataset.start_group, dataset.end_group)):
        left, right = str(left), str(right)
        if left in validation_groups and right in validation_groups:
            validation.append(index)
        elif left in training_groups and right in training_groups:
            train.append(index)
        else:
            cross.append(index)
    if not train or not validation:
        raise RuntimeError(
            "Unable to form non-empty source-disjoint train/validation sets. "
            "Check source_group metadata or reduce --val_ratio."
        )
    meta = {
        "num_groups": len(groups),
        "num_train_groups": len(training_groups),
        "num_val_groups": len(validation_groups),
        "num_train_samples": len(train),
        "num_val_samples": len(validation),
        "num_cross_group_dropped": len(cross),
        "validation_groups": sorted(validation_groups),
    }
    return np.asarray(train, np.int64), np.asarray(validation, np.int64), meta


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    @torch.no_grad()
    def update(self, module: torch.nn.Module) -> None:
        for key, value in module.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {key: value.clone() for key, value in self.shadow.items()}


def _masked_per_sample(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values).to(values.dtype)
    dimensions = tuple(range(1, values.ndim))
    return (
        (values * expanded).sum(dim=dimensions)
        / expanded.sum(dim=dimensions).clamp_min(1.0)
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.reshape(-1).clamp_min(1e-4)
    return (values * weights).sum() / weights.sum().clamp_min(1e-4)


def _difference(values: torch.Tensor, order: int) -> torch.Tensor:
    result = values
    for _ in range(order):
        result = result[:, 1:] - result[:, :-1]
    return result


def _difference_mask(mask: torch.Tensor, order: int) -> torch.Tensor:
    result = mask
    for _ in range(order):
        result = result[:, 1:] * result[:, :-1]
    return result


def _coordinates(length: torch.Tensor, maximum_length: int) -> torch.Tensor:
    positions = torch.arange(
        1, maximum_length + 1,
        device=length.device,
        dtype=length.dtype,
    ).reshape(1, -1)
    return (positions / (length.reshape(-1, 1) + 1.0)).clamp(0.0, 1.0)[..., None]


def _spectral_loss(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variable-length log-spectrum and high-frequency excess losses."""
    spectral: List[torch.Tensor] = []
    high_excess: List[torch.Tensor] = []
    for batch_index in range(predicted_positions.shape[0]):
        length = int(lengths[batch_index].item())
        if length < 6:
            continue
        pred_velocity = torch.diff(
            predicted_positions[batch_index, :length], dim=0
        ).reshape(length - 1, -1)
        target_velocity = torch.diff(
            target_positions[batch_index, :length], dim=0
        ).reshape(length - 1, -1)
        pred_spectrum = torch.fft.rfft(pred_velocity, dim=0).abs()
        target_spectrum = torch.fft.rfft(target_velocity, dim=0).abs()
        spectral.append(F.smooth_l1_loss(
            torch.log1p(pred_spectrum),
            torch.log1p(target_spectrum),
        ))
        frequency_count = pred_spectrum.shape[0]
        high_start = max(1, int(math.floor(0.55 * frequency_count)))
        pred_high = pred_spectrum[high_start:].square().mean()
        target_high = target_spectrum[high_start:].square().mean()
        high_excess.append(F.relu(pred_high - 1.10 * target_high))
    if not spectral:
        zero = predicted_positions.new_tensor(0.0)
        return zero, zero
    return torch.stack(spectral).mean(), torch.stack(high_excess).mean()


def autoencoder_loss(
    system: V30ContinuousTransitionSystem,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    weights: Dict[str, float],
    deterministic_latent: bool,
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
    real_target = batch["real_target"].to(device).reshape(-1)
    # Real samples receive a modest extra weight without erasing synthetic
    # regularisation examples.
    sample_weight = sample_weight * (1.0 + 0.35 * real_target)

    condition = system.condition(
        start,
        end,
        start_velocity,
        end_velocity,
        music,
        length,
    )
    mean, logvar = system.encode(target, mask, condition)
    latent = system.reparameterize(
        mean, logvar, deterministic=deterministic_latent
    )
    coordinate = _coordinates(length, target.shape[1])
    predicted = system.decode(
        latent,
        start,
        end,
        start_velocity,
        end_velocity,
        condition,
        coordinate,
        length,
    )
    predicted = project_motion_rotations_torch(predicted)
    target = project_motion_rotations_torch(target)

    losses: Dict[str, torch.Tensor] = {}
    rotation = geodesic_rotation_error_torch(predicted, target).square()
    losses["rotation"] = _weighted_mean(
        _masked_per_sample(rotation, mask), sample_weight
    )
    losses["root"] = _weighted_mean(
        _masked_per_sample((predicted[..., ROOT] - target[..., ROOT]).square(), mask),
        sample_weight,
    )
    losses["contact"] = _weighted_mean(
        _masked_per_sample((predicted[..., CONTACT] - target[..., CONTACT]).square(), mask),
        sample_weight,
    )

    predicted_position = motion_to_joint_positions_torch(predicted)
    target_position = motion_to_joint_positions_torch(target)
    losses["fk"] = _weighted_mean(
        _masked_per_sample((predicted_position - target_position).square(), mask),
        sample_weight,
    )
    for name, order in (("velocity", 1), ("acceleration", 2), ("jerk", 3)):
        if target.shape[1] <= order:
            losses[name] = target.new_tensor(0.0)
            continue
        difference_mask = _difference_mask(mask, order)
        losses[name] = _weighted_mean(
            _masked_per_sample(
                F.smooth_l1_loss(
                    _difference(predicted_position, order),
                    _difference(target_position, order),
                    reduction="none",
                ),
                difference_mask,
            ),
            sample_weight,
        )

    predicted_angular = angular_velocity_torch(predicted)
    target_angular = angular_velocity_torch(target)
    losses["angular_velocity"] = _weighted_mean(
        _masked_per_sample(
            F.smooth_l1_loss(
                predicted_angular,
                target_angular,
                reduction="none",
            ),
            _difference_mask(mask, 1),
        ),
        sample_weight,
    )

    lengths = mask.sum(dim=1).long().clamp_min(1)
    batch_indices = torch.arange(len(target), device=device)
    last = lengths - 1
    predicted_first = predicted[:, 0]
    predicted_last = predicted[batch_indices, last]
    target_first = target[:, 0]
    target_last = target[batch_indices, last]
    endpoint_pose = (
        F.smooth_l1_loss(predicted_first, target_first, reduction="none").mean(dim=-1)
        + F.smooth_l1_loss(predicted_last, target_last, reduction="none").mean(dim=-1)
    )
    endpoint_velocity = (
        F.smooth_l1_loss(
            predicted_first - start,
            start_velocity,
            reduction="none",
        ).mean(dim=-1)
        + F.smooth_l1_loss(
            end - predicted_last,
            end_velocity,
            reduction="none",
        ).mean(dim=-1)
    )
    losses["endpoint"] = _weighted_mean(
        endpoint_pose + endpoint_velocity, sample_weight
    )

    spectral, high_excess = _spectral_loss(
        predicted_position, target_position, lengths
    )
    losses["spectral"] = spectral
    losses["high_frequency"] = high_excess
    losses["kl"] = 0.5 * torch.mean(
        torch.exp(logvar) + mean.square() - 1.0 - logvar
    )
    losses["latent"] = mean.square().mean()

    total = target.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    metrics = {name: float(value.detach().cpu()) for name, value in losses.items()}
    metrics["total"] = float(total.detach().cpu())
    return total, metrics


def _make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_autoencoder_epoch(
    system: V30ContinuousTransitionSystem,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: Dict[str, float],
    amp: bool,
    grad_clip: float,
) -> Dict[str, float]:
    training = optimizer is not None
    system.train(training)
    totals: Dict[str, List[float]] = {}
    scaler = _make_grad_scaler(amp and training and device.type == "cuda")
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
                loss, metrics = autoencoder_loss(
                    system,
                    batch,
                    device,
                    weights,
                    deterministic_latent=not training,
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            for name, value in metrics.items():
                totals.setdefault(name, []).append(value)
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in totals.items()
    }


@torch.no_grad()
def encode_subset(
    system: V30ContinuousTransitionSystem,
    dataset: TransitionDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> Dict[str, torch.Tensor]:
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    system.eval()
    rows: Dict[str, List[torch.Tensor]] = {
        "latent": [], "condition": [], "weight": [],
    }
    for batch in loader:
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        start = batch["start"].to(device)
        end = batch["end"].to(device)
        music = batch["music"].to(device)
        length = batch["length"].to(device).reshape(-1, 1)
        start_velocity = batch["start_velocity"].to(device)
        end_velocity = batch["end_velocity"].to(device)
        condition = system.condition(
            start, end, start_velocity, end_velocity, music, length
        )
        mean, _ = system.encode(target, mask, condition)
        rows["latent"].append(mean.cpu())
        rows["condition"].append(condition.cpu())
        rows["weight"].append(batch["sample_weight"].reshape(-1).float())
    return {key: torch.cat(value, dim=0) for key, value in rows.items()}


class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        self.latent = latent.float()
        self.condition = condition.float()
        self.weight = weight.float()

    def __len__(self) -> int:
        return len(self.latent)

    def __getitem__(self, index: int):
        return self.latent[index], self.condition[index], self.weight[index]


def latent_diffusion_epoch(
    system: V30ContinuousTransitionSystem,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    diffusion_steps: int,
    condition_dropout: float,
    device: torch.device,
    amp: bool,
    grad_clip: float,
    seed: int | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    system.diffusion.train(training)
    scaler = _make_grad_scaler(amp and training and device.type == "cuda")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    _, _, alpha_bar = linear_beta_schedule(diffusion_steps, device)
    losses: List[float] = []
    x0_losses: List[float] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for latent, condition, weight in loader:
            latent = latent.to(device)
            condition = condition.to(device)
            weight = weight.to(device).reshape(-1).clamp_min(1e-4)
            normalised = (latent - latent_mean) / latent_std
            timestep = torch.randint(
                0,
                diffusion_steps,
                (len(latent),),
                device=device,
                generator=generator,
            )
            noise = torch.randn(
                normalised.shape,
                device=device,
                dtype=normalised.dtype,
                generator=generator,
            )
            alpha = alpha_bar[timestep].reshape(-1, 1)
            noisy = torch.sqrt(alpha) * normalised + torch.sqrt(1.0 - alpha) * noise
            if training and condition_dropout > 0.0:
                keep = (
                    torch.rand((len(condition), 1), device=device) >= condition_dropout
                ).to(condition.dtype)
                model_condition = condition * keep
            else:
                model_condition = condition
            time = timestep.float() / max(diffusion_steps - 1, 1)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                predicted_noise = system.diffusion(noisy, time, model_condition)
                per_sample = (predicted_noise - noise).square().mean(dim=-1)
                noise_loss = (per_sample * weight).sum() / weight.sum()
                predicted_x0 = (
                    noisy - torch.sqrt(1.0 - alpha) * predicted_noise
                ) / torch.sqrt(alpha).clamp_min(1e-6)
                x0_per_sample = F.smooth_l1_loss(
                    predicted_x0, normalised, reduction="none"
                ).mean(dim=-1)
                x0_loss = (x0_per_sample * weight).sum() / weight.sum()
                loss = noise_loss + 0.10 * x0_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(system.diffusion.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(system.diffusion)
            losses.append(float(noise_loss.detach().cpu()))
            x0_losses.append(float(x0_loss.detach().cpu()))
    return {
        "noise": float(np.mean(losses)) if losses else 0.0,
        "x0": float(np.mean(x0_losses)) if x0_losses else 0.0,
        "total": float(np.mean(losses) + 0.10 * np.mean(x0_losses)) if losses else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stage", choices=["all", "autoencoder", "diffusion"], default="all")
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--ae_epochs", type=int, default=220)
    parser.add_argument("--diffusion_epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--latent_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--diffusion_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--encoder_hidden", type=int, default=320)
    parser.add_argument("--inr_hidden", type=int, default=320)
    parser.add_argument("--inr_layers", type=int, default=5)
    parser.add_argument("--fourier_bands", type=int, default=10)
    parser.add_argument("--diffusion_hidden", type=int, default=512)
    parser.add_argument("--diffusion_blocks", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--ae_patience", type=int, default=55)
    parser.add_argument("--diffusion_patience", type=int, default=70)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)

    parser.add_argument("--w_rotation", type=float, default=0.35)
    parser.add_argument("--w_root", type=float, default=0.20)
    parser.add_argument("--w_contact", type=float, default=0.06)
    parser.add_argument("--w_fk", type=float, default=0.40)
    parser.add_argument("--w_velocity", type=float, default=0.48)
    parser.add_argument("--w_acceleration", type=float, default=0.24)
    parser.add_argument("--w_jerk", type=float, default=0.08)
    parser.add_argument("--w_angular_velocity", type=float, default=0.20)
    parser.add_argument("--w_endpoint", type=float, default=0.75)
    parser.add_argument("--w_spectral", type=float, default=0.16)
    parser.add_argument("--w_high_frequency", type=float, default=0.08)
    parser.add_argument("--w_kl", type=float, default=2e-4)
    parser.add_argument("--w_latent", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = TransitionDataset(args.data)
    train_indices, validation_indices, split_meta = source_disjoint_split(
        dataset, args.val_ratio, args.seed
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, validation_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = V30INRConfig(
        motion_dim=int(dataset.target.shape[-1]),
        music_dim=int(dataset.music.shape[-1]),
        latent_dim=args.latent_dim,
        condition_dim=args.condition_dim,
        encoder_hidden=args.encoder_hidden,
        inr_hidden=args.inr_hidden,
        inr_layers=args.inr_layers,
        fourier_bands=args.fourier_bands,
        diffusion_hidden=args.diffusion_hidden,
        diffusion_blocks=args.diffusion_blocks,
        max_len=int(dataset.target.shape[1]),
        dropout=args.dropout,
    )
    system = V30ContinuousTransitionSystem(config).to(device)
    weights = {
        "rotation": args.w_rotation,
        "root": args.w_root,
        "contact": args.w_contact,
        "fk": args.w_fk,
        "velocity": args.w_velocity,
        "acceleration": args.w_acceleration,
        "jerk": args.w_jerk,
        "angular_velocity": args.w_angular_velocity,
        "endpoint": args.w_endpoint,
        "spectral": args.w_spectral,
        "high_frequency": args.w_high_frequency,
        "kl": args.w_kl,
        "latent": args.w_latent,
    }

    ae_best_path = checkpoint_dir / "best_autoencoder.pt"
    ae_history: List[Dict[str, object]] = []
    if args.stage in {"all", "autoencoder"}:
        ae_parameters = list(system.condition_encoder.parameters()) + list(system.encoder.parameters()) + list(system.inr.parameters())
        optimizer = torch.optim.AdamW(
            ae_parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(args.ae_epochs, 1), eta_min=args.lr * 0.08
        )
        best = float("inf")
        bad_epochs = 0
        for epoch in range(1, args.ae_epochs + 1):
            train_metrics = run_autoencoder_epoch(
                system, train_loader, optimizer, device, weights,
                bool(args.amp), args.grad_clip,
            )
            validation_metrics = run_autoencoder_epoch(
                system, validation_loader, None, device, weights,
                False, args.grad_clip,
            )
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "val": validation_metrics,
                "train_loss": train_metrics.get("total", 0.0),
                "val_loss": validation_metrics.get("total", 0.0),
            }
            ae_history.append(row)
            print(
                f"[V30 INR-AE] epoch={epoch:04d} "
                f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
                f"val_fk={validation_metrics.get('fk', 0.0):.6f} "
                f"val_jerk={validation_metrics.get('jerk', 0.0):.6f} "
                f"val_spectral={validation_metrics.get('spectral', 0.0):.6f}",
                flush=True,
            )
            if float(row["val_loss"]) < best:
                best = float(row["val_loss"])
                bad_epochs = 0
                torch.save(
                    {
                        "system": system.state_dict(),
                        "config": asdict(config),
                        "loss_weights": weights,
                        "best_val_loss": best,
                        "epoch": epoch,
                        "split_meta": split_meta,
                        "dataset_meta": dataset.meta,
                    },
                    ae_best_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= args.ae_patience:
                    print(f"[V30 INR-AE EARLY STOP] patience={args.ae_patience}")
                    break
        (output / "autoencoder_history.json").write_text(
            json.dumps(ae_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        ae_best_path = Path(args.autoencoder_ckpt)
        if not ae_best_path.is_file():
            raise FileNotFoundError("--autoencoder_ckpt is required for diffusion-only stage")

    ae_checkpoint = torch.load(ae_best_path, map_location=device, weights_only=False)
    system.load_state_dict(ae_checkpoint["system"])

    if args.stage == "autoencoder":
        (output / "BEST_V30_INR_AUTOENCODER_CKPT.txt").write_text(
            str(ae_best_path), encoding="utf-8"
        )
        print(f"[SAVED] {ae_best_path}")
        return

    # Freeze the continuous representation before latent diffusion training.
    for module in (system.condition_encoder, system.encoder, system.inr):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    train_encoded = encode_subset(
        system, dataset, train_indices, args.batch_size,
        device, args.num_workers,
    )
    validation_encoded = encode_subset(
        system, dataset, validation_indices, args.batch_size,
        device, args.num_workers,
    )
    latent_mean = train_encoded["latent"].mean(dim=0, keepdim=True).to(device)
    latent_std = train_encoded["latent"].std(dim=0, keepdim=True).clamp_min(1e-3).to(device)
    train_latent_dataset = LatentDataset(**train_encoded)
    validation_latent_dataset = LatentDataset(**validation_encoded)
    train_latent_loader = torch.utils.data.DataLoader(
        train_latent_dataset,
        batch_size=args.latent_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    validation_latent_loader = torch.utils.data.DataLoader(
        validation_latent_dataset,
        batch_size=args.latent_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    diffusion_optimizer = torch.optim.AdamW(
        system.diffusion.parameters(),
        lr=args.diffusion_lr,
        weight_decay=args.weight_decay,
    )
    diffusion_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        diffusion_optimizer,
        T_max=max(args.diffusion_epochs, 1),
        eta_min=args.diffusion_lr * 0.08,
    )
    ema = EMA(system.diffusion, args.ema_decay)
    best_path = checkpoint_dir / "best.pt"
    best = float("inf")
    bad_epochs = 0
    diffusion_history: List[Dict[str, object]] = []
    for epoch in range(1, args.diffusion_epochs + 1):
        train_metrics = latent_diffusion_epoch(
            system,
            train_latent_loader,
            diffusion_optimizer,
            ema,
            latent_mean,
            latent_std,
            args.diffusion_steps,
            args.condition_dropout,
            device,
            bool(args.amp),
            args.grad_clip,
        )
        validation_metrics = latent_diffusion_epoch(
            system,
            validation_latent_loader,
            None,
            None,
            latent_mean,
            latent_std,
            args.diffusion_steps,
            0.0,
            device,
            False,
            args.grad_clip,
            seed=args.seed + 9173,
        )
        diffusion_scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(diffusion_optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": validation_metrics,
            "train_loss": train_metrics["total"],
            "val_loss": validation_metrics["total"],
        }
        diffusion_history.append(row)
        print(
            f"[V30 latent diffusion] epoch={epoch:04d} "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"val_noise={validation_metrics['noise']:.6f}",
            flush=True,
        )
        if float(row["val_loss"]) < best:
            best = float(row["val_loss"])
            bad_epochs = 0
            checkpoint_config = {
                "architecture": "v30_continuous_so3_inr_latent_diffusion",
                "model": asdict(config),
                "diffusion_steps": args.diffusion_steps,
                "condition_dropout": args.condition_dropout,
                "loss_weights": weights,
                "split_meta": split_meta,
                "dataset_meta": dataset.meta,
                "seed": args.seed,
            }
            torch.save(
                {
                    "system": system.state_dict(),
                    "ema_diffusion": ema.state_dict(),
                    "config": checkpoint_config,
                    "latent_mean": latent_mean.detach().cpu().numpy(),
                    "latent_std": latent_std.detach().cpu().numpy(),
                    "best_val_loss": best,
                    "epoch": epoch,
                    "autoencoder_checkpoint": str(ae_best_path),
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.diffusion_patience:
                print(
                    f"[V30 latent diffusion EARLY STOP] patience={args.diffusion_patience}"
                )
                break

    (output / "diffusion_history.json").write_text(
        json.dumps(diffusion_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "history.json").write_text(
        json.dumps(diffusion_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "split.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "BEST_V27_TRANSITION_DIFFUSION_CKPT.txt").write_text(
        str(best_path), encoding="utf-8"
    )
    (output / "BEST_V30_CONTINUOUS_INR_DIFFUSION_CKPT.txt").write_text(
        str(best_path), encoding="utf-8"
    )
    (output / "BEST_V30_INR_AUTOENCODER_CKPT.txt").write_text(
        str(ae_best_path), encoding="utf-8"
    )
    print(f"[SAVED] {best_path}")


if __name__ == "__main__":
    main()
