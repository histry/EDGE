#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train V31 band-limited SO(3) coefficient diffusion.

There is no neural INR autoencoder in V31. Real motion residuals are fitted to
an analytic C2-zero basis with weighted ridge regression. PCA compresses the
coefficient vectors, and conditional diffusion models only this compact,
band-limited latent. Checkpoint selection includes decoded SO(3), FK velocity
and jerk losses rather than latent noise loss alone.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.v29_motion_geometry import (
    geodesic_rotation_error_torch,
    motion_to_joint_positions_torch,
    project_motion_rotations_torch,
)
from tools.v31_bandlimited_transition import (
    RESIDUAL_DIM,
    V31Config,
    V31TransitionModel,
    decode_coefficients,
    fit_coefficients,
    linear_beta_schedule,
)


REAL_KINDS = {
    "intra_event_real",
    "source_gap_real",
    "source_boundary_mask_real",
}


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, include_synthetic: bool) -> None:
        z = np.load(path, allow_pickle=True)
        all_kind = (
            np.asarray(z["sample_kind"], dtype=object)
            if "sample_kind" in z.files
            else np.asarray(["unknown"] * len(z["target"]), dtype=object)
        )
        real_target = (
            np.asarray(z["real_target"], dtype=np.bool_)
            if "real_target" in z.files
            else np.asarray(
                [str(kind) in REAL_KINDS for kind in all_kind],
                dtype=np.bool_,
            )
        )
        keep = np.ones((len(all_kind),), dtype=np.bool_)
        if not include_synthetic:
            keep = np.asarray(
                [
                    bool(real_target[i]) and str(all_kind[i]) in REAL_KINDS
                    for i in range(len(all_kind))
                ],
                dtype=np.bool_,
            )
        indices = np.flatnonzero(keep)
        if len(indices) == 0:
            raise RuntimeError(
                "No eligible V31 samples. Build real full-sequence boundary "
                "masks or pass --include_synthetic 1 only for an ablation."
            )

        def array(name: str, default=None):
            if name in z.files:
                return np.asarray(z[name])[indices]
            if default is None:
                raise KeyError(name)
            return np.asarray(default)[indices]

        self.target = array("target").astype(np.float32)
        self.mask = array("mask").astype(np.float32)
        self.start = array("start").astype(np.float32)
        self.end = array("end").astype(np.float32)
        self.music = array(
            "music",
            np.zeros((len(all_kind), 12), np.float32),
        ).astype(np.float32)
        self.length = array("length").astype(np.float32)
        self.sample_weight = array(
            "sample_weight",
            np.ones((len(all_kind),), np.float32),
        ).astype(np.float32)
        self.kind = all_kind[indices]
        self.real_target = real_target[indices]
        self.start_group = array(
            "start_group",
            np.asarray([f"sample:{i}" for i in range(len(all_kind))], object),
        ).astype(object)
        self.end_group = array(
            "end_group",
            np.asarray([f"sample:{i}" for i in range(len(all_kind))], object),
        ).astype(object)
        self.meta = (
            json.loads(str(z["meta"].item()))
            if "meta" in z.files else {}
        )
        self.original_indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> Dict[str, object]:
        length = int(self.length[index])
        start_velocity = (
            self.target[index, 0] - self.start[index]
            if length > 0 else np.zeros_like(self.start[index])
        )
        end_velocity = (
            self.end[index] - self.target[index, length - 1]
            if length > 0 else np.zeros_like(self.end[index])
        )
        return {
            "target": self.target[index],
            "mask": self.mask[index],
            "start": self.start[index],
            "end": self.end[index],
            "music": self.music[index],
            "length": self.length[index],
            "weight": self.sample_weight[index],
            "start_velocity": start_velocity.astype(np.float32),
            "end_velocity": end_velocity.astype(np.float32),
            "index": index,
        }


def source_disjoint_split(
    dataset: TransitionDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    groups = sorted(
        set(str(x) for x in dataset.start_group)
        | set(str(x) for x in dataset.end_group)
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = max(1, min(len(groups) - 1, int(round(len(groups) * val_ratio))))
    validation = set(groups[:count])
    train, val, dropped = [], [], []
    for index, (a, b) in enumerate(
        zip(dataset.start_group, dataset.end_group)
    ):
        a, b = str(a), str(b)
        if a in validation and b in validation:
            val.append(index)
        elif a not in validation and b not in validation:
            train.append(index)
        else:
            dropped.append(index)
    if not train or not val:
        raise RuntimeError(
            "Cannot form a source-disjoint split. Provide more source groups."
        )
    return (
        np.asarray(train, np.int64),
        np.asarray(val, np.int64),
        {
            "num_groups": len(groups),
            "validation_groups": sorted(validation),
            "train_samples_before_fit_filter": len(train),
            "validation_samples_before_fit_filter": len(val),
            "cross_group_dropped": len(dropped),
        },
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
        for key, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[key].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)


@torch.no_grad()
def fit_all_coefficients(
    dataset: TransitionDataset,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    basis_count: int,
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray]:
    coefficients, errors = [], []
    for start_index in range(0, len(indices), batch_size):
        selected = indices[start_index : start_index + batch_size]
        target = torch.from_numpy(dataset.target[selected]).to(device)
        mask = torch.from_numpy(dataset.mask[selected]).to(device)
        start = torch.from_numpy(dataset.start[selected]).to(device)
        end = torch.from_numpy(dataset.end[selected]).to(device)
        length = torch.from_numpy(dataset.length[selected]).to(device).reshape(-1, 1)
        first = target[:, 0]
        last_indices = mask.sum(dim=1).long().clamp_min(1) - 1
        batch_indices = torch.arange(len(selected), device=device)
        last = target[batch_indices, last_indices]
        start_velocity = first - start
        end_velocity = end - last
        coeff, error = fit_coefficients(
            target, mask, start, end,
            start_velocity, end_velocity, length,
            basis_count=basis_count, ridge=ridge,
        )
        coefficients.append(coeff.cpu().numpy())
        errors.append(error.cpu().numpy())
    return (
        np.concatenate(coefficients, axis=0).astype(np.float32),
        np.concatenate(errors, axis=0).astype(np.float32),
    )


def fit_pca(
    train_coefficients: np.ndarray,
    pca_dim: int,
) -> Dict[str, np.ndarray]:
    flat = train_coefficients.reshape(len(train_coefficients), -1).astype(np.float64)
    mean = flat.mean(axis=0, keepdims=True)
    centred = flat - mean
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    dimension = min(int(pca_dim), len(order), len(flat) - 1)
    components = vectors[:, order[:dimension]].T
    score = centred @ components.T
    score_mean = score.mean(axis=0, keepdims=True)
    score_std = score.std(axis=0, keepdims=True) + 1e-5
    explained = values[order[:dimension]].sum() / max(values.clip(0).sum(), 1e-12)
    coefficient_limit = np.percentile(np.abs(flat), 99.5, axis=0)
    coefficient_limit = np.maximum(coefficient_limit, 1e-4)
    return {
        "coefficient_mean": mean.astype(np.float32).reshape(-1),
        "pca_components": components.astype(np.float32),
        "score_mean": score_mean.astype(np.float32).reshape(-1),
        "score_std": score_std.astype(np.float32).reshape(-1),
        "coefficient_abs_limit": coefficient_limit.astype(np.float32),
        "explained_variance_ratio": np.asarray([explained], np.float32),
    }


def transform_pca(coefficients: np.ndarray, pca: Dict[str, np.ndarray]) -> np.ndarray:
    flat = coefficients.reshape(len(coefficients), -1)
    score = (
        (flat - pca["coefficient_mean"][None])
        @ pca["pca_components"].T
    )
    return (
        (score - pca["score_mean"][None])
        / pca["score_std"][None]
    ).astype(np.float32)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source: TransitionDataset,
        indices: np.ndarray,
        normalized_scores: np.ndarray,
        coefficients: np.ndarray,
    ) -> None:
        self.source = source
        self.indices = np.asarray(indices, np.int64)
        self.scores = np.asarray(normalized_scores, np.float32)
        self.coefficients = np.asarray(coefficients, np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, object]:
        index = int(self.indices[item])
        row = self.source[index]
        row["score"] = self.scores[item]
        row["coefficient"] = self.coefficients[item]
        return row


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values).to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def inverse_pca_torch(
    normalized: torch.Tensor,
    pca: Dict[str, torch.Tensor],
    config: V31Config,
) -> torch.Tensor:
    score = normalized * pca["score_std"] + pca["score_mean"]
    flat = pca["coefficient_mean"] + score @ pca["pca_components"]
    limit = pca["coefficient_abs_limit"]
    flat = torch.maximum(torch.minimum(flat, limit), -limit)
    return flat.reshape(
        flat.shape[0], config.basis_count, config.coefficient_dim
    )


def run_epoch(
    model: V31TransitionModel,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    pca: Dict[str, torch.Tensor],
    device: torch.device,
    diffusion_steps: int,
    condition_dropout: float,
    decoded_weight: float,
    decoded_batch_limit: int,
    amp: bool,
    grad_clip: float,
    seed: int | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, List[float]] = {}
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp and training and device.type == "cuda"
    )
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    _, _, alpha_bar = linear_beta_schedule(diffusion_steps, device)

    for batch in loader:
        score = batch["score"].to(device).float()
        target = batch["target"].to(device).float()
        mask = batch["mask"].to(device).float()
        start = batch["start"].to(device).float()
        end = batch["end"].to(device).float()
        music = batch["music"].to(device).float()
        length = batch["length"].to(device).float().reshape(-1, 1)
        weight = batch["weight"].to(device).float().reshape(-1)
        start_velocity = batch["start_velocity"].to(device).float()
        end_velocity = batch["end_velocity"].to(device).float()
        batch_size = len(score)

        timestep = torch.randint(
            0, diffusion_steps, (batch_size,),
            device=device, generator=generator,
        )
        noise = torch.randn(
            score.shape, device=device, generator=generator
        )
        ab = alpha_bar[timestep].reshape(batch_size, 1)
        noisy = torch.sqrt(ab) * score + torch.sqrt(1.0 - ab) * noise
        diffusion_time = timestep.float() / max(diffusion_steps - 1, 1)

        condition = model.condition(
            start, end, start_velocity, end_velocity, music, length
        )
        if training and condition_dropout > 0.0:
            drop = (
                torch.rand((batch_size, 1), device=device)
                < float(condition_dropout)
            )
            condition = torch.where(drop, torch.zeros_like(condition), condition)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            predicted_noise = model.diffusion(
                noisy, diffusion_time, condition
            )
            predicted_x0 = (
                noisy - torch.sqrt(1.0 - ab) * predicted_noise
            ) / torch.sqrt(ab).clamp_min(1e-6)
            noise_per = ((predicted_noise - noise) ** 2).mean(dim=-1)
            x0_per = F.smooth_l1_loss(
                predicted_x0, score, reduction="none"
            ).mean(dim=-1)
            sample = weight / weight.mean().clamp_min(1e-5)
            noise_loss = (noise_per * sample).mean()
            x0_loss = (x0_per * sample).mean()

        # SO(3) logarithms, SVD projection and FK are evaluated in float32.
        # Running these operations under fp16 autocast can itself create noisy
        # gradients and numerical rotation spikes.
        with torch.autocast(device_type=device.type, enabled=False):
            decoded_count = min(batch_size, max(1, int(decoded_batch_limit)))
            predicted_x0_float = predicted_x0[:decoded_count].float()
            predicted_coeff = inverse_pca_torch(
                predicted_x0_float,
                {key: value.float() for key, value in pca.items()},
                model.config,
            )
            maximum = target.shape[1]
            coordinate = (
                torch.arange(
                    maximum, device=device, dtype=torch.float32
                ).reshape(1, maximum, 1)
                + 1.0
            ) / (length[:decoded_count].float().reshape(-1, 1, 1) + 1.0)
            coordinate = coordinate.clamp(0.0, 1.0)
            predicted_motion = decode_coefficients(
                predicted_coeff,
                start[:decoded_count].float(), end[:decoded_count].float(),
                start_velocity[:decoded_count].float(),
                end_velocity[:decoded_count].float(),
                coordinate, length[:decoded_count].float(), model.config,
            )
            predicted_motion = project_motion_rotations_torch(predicted_motion)
            target_projected = project_motion_rotations_torch(
                target[:decoded_count].float()
            )

            rotation = _masked_mean(
                geodesic_rotation_error_torch(
                    predicted_motion, target_projected
                ) ** 2,
                mask[:decoded_count].float(),
            )
            predicted_position = motion_to_joint_positions_torch(predicted_motion)
            target_position = motion_to_joint_positions_torch(target_projected)
            fk = _masked_mean(
                (predicted_position - target_position) ** 2,
                mask[:decoded_count].float()
            )
            velocity_mask = _difference_mask(
                mask[:decoded_count].float(), 1
            )
            velocity = _masked_mean(
                F.smooth_l1_loss(
                    _difference(predicted_position, 1),
                    _difference(target_position, 1),
                    reduction="none",
                ),
                velocity_mask,
            )
            jerk_mask = _difference_mask(
                mask[:decoded_count].float(), 3
            )
            if predicted_position.shape[1] > 3:
                jerk = _masked_mean(
                    F.smooth_l1_loss(
                        _difference(predicted_position, 3),
                        _difference(target_position, 3),
                        reduction="none",
                    ),
                    jerk_mask,
                )
            else:
                jerk = predicted_position.new_tensor(0.0)

            decoded = rotation + 0.70 * fk + 0.35 * velocity + 0.12 * jerk
        loss = noise_loss.float() + 0.25 * x0_loss.float() + float(decoded_weight) * decoded

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

        values = {
            "total": loss,
            "noise": noise_loss,
            "x0": x0_loss,
            "rotation": rotation,
            "fk": fk,
            "velocity": velocity,
            "jerk": jerk,
            "decoded": decoded,
        }
        for key, value in values.items():
            totals.setdefault(key, []).append(float(value.detach().cpu()))

    return {
        key: float(np.mean(values)) if values else 0.0
        for key, values in totals.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fit_batch_size", type=int, default=64)
    parser.add_argument("--basis_count", type=int, default=6)
    parser.add_argument("--pca_dim", type=int, default=96)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--diffusion_blocks", type=int, default=6)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--rotation_residual_cap", type=float, default=0.16)
    parser.add_argument("--root_y_residual_cap", type=float, default=0.045)
    parser.add_argument("--ridge", type=float, default=2e-3)
    parser.add_argument("--max_fit_rmse", type=float, default=0.14)
    parser.add_argument("--include_synthetic", type=int, default=0)
    parser.add_argument("--weighted_sampling", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1.8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--decoded_weight", type=float, default=0.10)
    parser.add_argument("--decoded_batch_limit", type=int, default=12)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset = TransitionDataset(
        args.data, include_synthetic=bool(args.include_synthetic)
    )
    train_indices, validation_indices, split_meta = source_disjoint_split(
        dataset, args.val_ratio, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_indices = np.concatenate([train_indices, validation_indices])
    coefficients, fit_error = fit_all_coefficients(
        dataset, all_indices, device,
        args.fit_batch_size, args.basis_count, args.ridge,
    )
    error_by_index = {
        int(index): float(error)
        for index, error in zip(all_indices, fit_error)
    }
    coefficient_by_index = {
        int(index): coefficient
        for index, coefficient in zip(all_indices, coefficients)
    }
    train_indices = np.asarray([
        index for index in train_indices
        if error_by_index[int(index)] <= args.max_fit_rmse
    ], np.int64)
    validation_indices = np.asarray([
        index for index in validation_indices
        if error_by_index[int(index)] <= args.max_fit_rmse
    ], np.int64)
    if len(train_indices) < 64 or len(validation_indices) < 16:
        raise RuntimeError(
            "Too few band-limited real samples survived fitting: "
            f"train={len(train_indices)}, val={len(validation_indices)}. "
            "Inspect data instead of raising residual bandwidth."
        )

    train_coeff = np.stack([
        coefficient_by_index[int(index)] for index in train_indices
    ]).astype(np.float32)
    val_coeff = np.stack([
        coefficient_by_index[int(index)] for index in validation_indices
    ]).astype(np.float32)
    pca = fit_pca(train_coeff, args.pca_dim)
    actual_pca_dim = int(pca["pca_components"].shape[0])
    train_score = transform_pca(train_coeff, pca)
    val_score = transform_pca(val_coeff, pca)

    config = V31Config(
        motion_dim=int(dataset.target.shape[-1]),
        music_dim=int(dataset.music.shape[-1]),
        basis_count=args.basis_count,
        coefficient_dim=RESIDUAL_DIM,
        pca_dim=actual_pca_dim,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        diffusion_blocks=args.diffusion_blocks,
        diffusion_steps=args.diffusion_steps,
        dropout=args.dropout,
        max_len=int(dataset.target.shape[1]),
        rotation_residual_cap=args.rotation_residual_cap,
        root_y_residual_cap=args.root_y_residual_cap,
    )
    model = V31TransitionModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.08
    )
    ema = EMA(model, args.ema_decay)

    train_latent_dataset = LatentDataset(
        dataset, train_indices, train_score, train_coeff
    )
    sampler = None
    shuffle = True
    if bool(args.weighted_sampling):
        sampling_weight = torch.as_tensor(
            dataset.sample_weight[train_indices], dtype=torch.double
        ).clamp_min(1e-3)
        sampler = torch.utils.data.WeightedRandomSampler(
            sampling_weight,
            num_samples=len(train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        shuffle = False
    train_loader = torch.utils.data.DataLoader(
        train_latent_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        LatentDataset(dataset, validation_indices, val_score, val_coeff),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    pca_torch = {
        key: torch.from_numpy(value).to(device)
        for key, value in pca.items()
        if key != "explained_variance_ratio"
    }

    split_meta.update({
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(validation_indices)),
        "fit_rmse_limit": float(args.max_fit_rmse),
        "fit_rmse_mean": float(np.mean(fit_error)),
        "fit_rmse_p95": float(np.percentile(fit_error, 95)),
        "pca_dim": actual_pca_dim,
        "pca_explained_variance": float(
            pca["explained_variance_ratio"][0]
        ),
    })
    (output / "fit_report.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = float("inf")
    bad = 0
    history: List[Dict[str, object]] = []
    best_path = checkpoint_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, ema, pca_torch, device,
            args.diffusion_steps, args.condition_dropout,
            args.decoded_weight, args.decoded_batch_limit,
            bool(args.amp), args.grad_clip,
        )
        validation_metrics = run_epoch(
            model, val_loader, None, None, pca_torch, device,
            args.diffusion_steps, 0.0,
            args.decoded_weight, args.decoded_batch_limit,
            False, args.grad_clip,
            seed=args.seed + 9173,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "val": validation_metrics,
            "train_loss": train_metrics["total"],
            "val_loss": validation_metrics["total"],
        }
        history.append(row)
        print(
            f"[V31 coefficient diffusion] epoch={epoch:04d} "
            f"train={train_metrics['total']:.6f} "
            f"val={validation_metrics['total']:.6f} "
            f"val_decoded={validation_metrics['decoded']:.6f} "
            f"val_jerk={validation_metrics['jerk']:.6f}",
            flush=True,
        )
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            bad = 0
            checkpoint_config = {
                "architecture": "v31_bandlimited_so3_coefficient_diffusion",
                "model": asdict(config),
                "diffusion_steps": args.diffusion_steps,
                "dataset_meta": dataset.meta,
                "split_meta": split_meta,
                "seed": args.seed,
            }
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema_model": ema.shadow,
                    "config": checkpoint_config,
                    "coefficient_mean": pca["coefficient_mean"],
                    "pca_components": pca["pca_components"],
                    "score_mean": pca["score_mean"],
                    "score_std": pca["score_std"],
                    "coefficient_abs_limit": pca["coefficient_abs_limit"],
                    "best_val_loss": best,
                    "val_metrics": validation_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY STOP] patience={args.patience}")
                break

    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "split.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name in (
        "BEST_V27_TRANSITION_DIFFUSION_CKPT.txt",
        "BEST_V31_TRANSITION_CKPT.txt",
    ):
        (output / name).write_text(str(best_path), encoding="utf-8")
    print(f"[SAVED] {best_path}")


if __name__ == "__main__":
    main()
