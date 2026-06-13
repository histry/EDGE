#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train EDGE V32 continuous contact-aware INR and latent diffusion.

Historical filename is retained for scheduler compatibility.

Stages:
  1. autoencoder:
     deterministic variable-length encoder + continuous C2 SO(3) INR;
  2. contact_finetune:
     freeze encoder/condition/diffusion and fine-tune the INR on real targets
     with differentiable foot contact, skating, height and penetration losses;
  3. diffusion:
     standardise deterministic latents and train conditional latent diffusion.
     A limited decoded-contact loss is back-propagated through the frozen INR.

The main model can use real intra-event windows even when full source gaps are
unavailable. Synthetic adjacent samples remain an explicitly controlled weak
prior and are never counted as real supervision.
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
from tools.v34_boundary_dynamics import (
    boundary_state_from_training_batch,
)

from tools.v32_contact_inr import (
    V32ContactINRSystem,
    V32INRConfig,
    linear_beta_schedule,
)
from tools.v32_contact_losses import (
    contact_loss_total,
    differentiable_contact_losses,
    masked_per_sample,
    weighted_mean,
)


REAL_KINDS = {
    "intra_event_real",
    "source_gap_real",
    "source_boundary_mask_real",
}


class TransitionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str | Path,
        include_synthetic: bool,
        real_only: bool = False,
    ) -> None:
        data = np.load(path, allow_pickle=True)
        kinds = np.asarray(
            data["sample_kind"]
            if "sample_kind" in data.files
            else ["unknown"] * len(data["target"]),
            dtype=object,
        )
        real_target = np.asarray(
            data["real_target"]
            if "real_target" in data.files
            else [str(x) in REAL_KINDS for x in kinds],
            dtype=np.bool_,
        )
        keep = np.ones((len(kinds),), dtype=np.bool_)
        if real_only:
            keep = real_target & np.asarray(
                [str(x) in REAL_KINDS for x in kinds], dtype=np.bool_
            )
        elif not include_synthetic:
            keep = real_target
        indices = np.flatnonzero(keep)
        if len(indices) == 0:
            raise RuntimeError(
                "No eligible transition samples. Build intra-event/real "
                "boundary samples or enable synthetic data only for an ablation."
            )

        def array(name: str, default=None):
            if name in data.files:
                return np.asarray(data[name])[indices]
            if default is None:
                raise KeyError(name)
            return np.asarray(default)[indices]

        total = len(kinds)
        self.target = array("target").astype(np.float32)
        self.mask = array("mask").astype(np.float32)
        self.start = array("start").astype(np.float32)
        self.end = array("end").astype(np.float32)
        self.music = array(
            "music", np.zeros((total, 12), np.float32)
        ).astype(np.float32)
        self.length = array("length").astype(np.float32)
        self.sample_weight = array(
            "sample_weight", np.ones((total,), np.float32)
        ).astype(np.float32)
        if "contact_confidence" not in data.files:
            raise RuntimeError(
                "V33 training requires event-level contact_confidence. "
                "Rebuild the dataset with build_v33_event_contact_cache.py "
                "and the V33 transition dataset builder."
            )
        self.contact_confidence = array("contact_confidence").astype(np.float32)
        self.start_contact_confidence = array(
            "start_contact_confidence", np.ones((total, 4), np.float32)
        ).astype(np.float32)
        self.end_contact_confidence = array(
            "end_contact_confidence", np.ones((total, 4), np.float32)
        ).astype(np.float32)
        self.sample_kind = kinds[indices]
        self.real_target = real_target[indices]
        self.start_group = array(
            "start_group",
            np.asarray([f"sample:{i}" for i in range(total)], object),
        ).astype(object)
        self.end_group = array(
            "end_group",
            np.asarray([f"sample:{i}" for i in range(total)], object),
        ).astype(object)
        self.meta = (
            json.loads(str(data["meta"].item()))
            if "meta" in data.files else {}
        )
        contact_pipeline = self.meta.get("contact_pipeline", {})
        if contact_pipeline.get("level") != "complete_event_before_window_sampling":
            raise RuntimeError(
                "Refusing non-event-level contact dataset: "
                f"contact_pipeline={contact_pipeline}"
            )
        if not bool(contact_pipeline.get("synchronised_slicing", False)):
            raise RuntimeError("Contact labels were not synchronously sliced")
        self.original_indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int) -> Dict[str, object]:
        length = max(1, int(self.length[index]))
        first = self.target[index, 0]
        last = self.target[index, length - 1]
        return {
            "target": self.target[index],
            "mask": self.mask[index],
            "start": self.start[index],
            "end": self.end[index],
            "music": self.music[index],
            "length": self.length[index],
            "sample_weight": self.sample_weight[index],
            "contact_confidence": self.contact_confidence[index],
            "start_contact_confidence": self.start_contact_confidence[index],
            "end_contact_confidence": self.end_contact_confidence[index],
            "start_velocity": (first - self.start[index]).astype(np.float32),
            "end_velocity": (self.end[index] - last).astype(np.float32),
            "real_target": float(self.real_target[index]),
            "index": int(index),
        }


def source_disjoint_split(
    dataset: TransitionDataset,
    validation_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    groups = sorted(
        set(str(x) for x in dataset.start_group)
        | set(str(x) for x in dataset.end_group)
    )
    if len(groups) < 2:
        raise RuntimeError(
            "At least two source groups are required for source-disjoint validation"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    validation_count = max(
        1, min(len(groups) - 1, int(round(len(groups) * validation_ratio)))
    )
    validation_groups = set(groups[:validation_count])
    training_groups = set(groups[validation_count:])
    train, validation, cross = [], [], []
    for index, (left, right) in enumerate(
        zip(dataset.start_group, dataset.end_group)
    ):
        left, right = str(left), str(right)
        if left in validation_groups and right in validation_groups:
            validation.append(index)
        elif left in training_groups and right in training_groups:
            train.append(index)
        else:
            cross.append(index)
    if not train or not validation:
        raise RuntimeError(
            "Unable to form non-empty source-disjoint train/validation sets"
        )
    return (
        np.asarray(train, np.int64),
        np.asarray(validation, np.int64),
        {
            "num_groups": len(groups),
            "train_groups": len(training_groups),
            "validation_groups": sorted(validation_groups),
            "train_samples": len(train),
            "validation_samples": len(validation),
            "cross_group_dropped": len(cross),
        },
    )


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


def _coordinates(length: torch.Tensor, maximum: int) -> torch.Tensor:
    positions = torch.arange(
        1, maximum + 1, device=length.device, dtype=length.dtype
    ).reshape(1, -1)
    return (
        positions / (length.reshape(-1, 1) + 1.0)
    ).clamp(0.0, 1.0)[..., None]


def _spectral_loss(
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    lengths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spectral, high_excess = [], []
    for batch_index in range(predicted_positions.shape[0]):
        length = int(lengths[batch_index].item())
        if length < 8:
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
            torch.log1p(pred_spectrum), torch.log1p(target_spectrum)
        ))
        high_start = max(1, int(0.55 * pred_spectrum.shape[0]))
        pred_high = pred_spectrum[high_start:].square().mean()
        target_high = target_spectrum[high_start:].square().mean()
        high_excess.append(F.relu(pred_high - 1.05 * target_high))
    if not spectral:
        zero = predicted_positions.new_tensor(0.0)
        return zero, zero
    return torch.stack(spectral).mean(), torch.stack(high_excess).mean()


def latent_regularisation(latent: torch.Tensor) -> Dict[str, torch.Tensor]:
    """VICReg-style deterministic latent regularisation."""
    centered = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    if len(latent) > 1:
        covariance = centered.transpose(0, 1) @ centered / float(len(latent) - 1)
        off_diagonal = covariance - torch.diag(torch.diag(covariance))
        covariance_loss = off_diagonal.square().sum() / float(latent.shape[1])
    else:
        covariance_loss = latent.new_tensor(0.0)
    mean_loss = latent.mean(dim=0).square().mean()
    return {
        "latent_variance": variance,
        "latent_covariance": covariance_loss,
        "latent_mean": mean_loss,
    }


def reconstruction_losses(
    system: V32ContactINRSystem,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    weights: Dict[str, float],
    detach_latent: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    target = batch["target"].to(device).float()
    mask = batch["mask"].to(device).float()
    start = batch["start"].to(device).float()
    end = batch["end"].to(device).float()
    music = batch["music"].to(device).float()
    length = batch["length"].to(device).float().reshape(-1, 1)
    sample_weight = batch["sample_weight"].to(device).float().reshape(-1)
    contact_confidence = batch["contact_confidence"].to(device).float()
    real_target = batch["real_target"].to(device).float().reshape(-1)
    sample_weight = sample_weight * (1.0 + 0.50 * real_target)
    start_velocity = batch["start_velocity"].to(device).float()
    end_velocity = batch["end_velocity"].to(device).float()

    condition = system.condition(
        start, end, start_velocity, end_velocity, music, length
    )
    latent = system.encode(target, mask, condition)
    if detach_latent:
        latent = latent.detach()
        condition = condition.detach()
    coordinate = _coordinates(length, target.shape[1])
    boundary_state = boundary_state_from_training_batch(
        start, target, end, length
    )
    predicted, aux = system.decode(
        latent,
        start,
        end,
        start_velocity,
        end_velocity,
        condition,
        coordinate,
        length,
        return_aux=True,
        boundary_state=boundary_state,
    )
    predicted = project_motion_rotations_torch(predicted)
    target = project_motion_rotations_torch(target)

    losses: Dict[str, torch.Tensor] = {}
    rotation = geodesic_rotation_error_torch(predicted, target).square()
    losses["rotation"] = weighted_mean(
        masked_per_sample(rotation, mask), sample_weight
    )
    losses["root"] = weighted_mean(
        masked_per_sample(
            (predicted[..., ROOT] - target[..., ROOT]).square(), mask
        ),
        sample_weight,
    )

    predicted_positions = motion_to_joint_positions_torch(predicted)
    target_positions = motion_to_joint_positions_torch(target)
    losses["fk"] = weighted_mean(
        masked_per_sample(
            (predicted_positions - target_positions).square(), mask
        ),
        sample_weight,
    )
    for name, order in (
        ("velocity", 1),
        ("acceleration", 2),
        ("jerk", 3),
    ):
        if target.shape[1] <= order:
            losses[name] = target.new_tensor(0.0)
            continue
        diff_mask = _difference_mask(mask, order)
        losses[name] = weighted_mean(
            masked_per_sample(
                F.smooth_l1_loss(
                    _difference(predicted_positions, order),
                    _difference(target_positions, order),
                    reduction="none",
                ),
                diff_mask,
            ),
            sample_weight,
        )

    predicted_angular = angular_velocity_torch(predicted)
    target_angular = angular_velocity_torch(target)
    losses["angular_velocity"] = weighted_mean(
        masked_per_sample(
            F.smooth_l1_loss(
                predicted_angular, target_angular, reduction="none"
            ),
            _difference_mask(mask, 1),
        ),
        sample_weight,
    )

    lengths = mask.sum(dim=1).long().clamp_min(1)
    indices = torch.arange(len(target), device=device)
    last = lengths - 1
    predicted_first = predicted[:, 0]
    predicted_last = predicted[indices, last]
    target_first = target[:, 0]
    target_last = target[indices, last]
    endpoint_pose = (
        F.smooth_l1_loss(
            predicted_first, target_first, reduction="none"
        ).mean(dim=-1)
        + F.smooth_l1_loss(
            predicted_last, target_last, reduction="none"
        ).mean(dim=-1)
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
    losses["endpoint"] = weighted_mean(
        endpoint_pose + endpoint_velocity, sample_weight
    )

    # V34 explicitly supervises the first/last acceleration and jerk of the
    # decoded interval in FK space.  This complements the septic base and keeps
    # the learned residual from re-introducing a cross-boundary impulse.
    start_positions = motion_to_joint_positions_torch(start)
    end_positions = motion_to_joint_positions_torch(end)
    endpoint_acceleration_rows = []
    endpoint_jerk_rows = []
    for batch_index in range(len(target)):
        count = int(lengths[batch_index].item())
        predicted_sequence = torch.cat([
            start_positions[batch_index : batch_index + 1],
            predicted_positions[batch_index, :count],
            end_positions[batch_index : batch_index + 1],
        ], dim=0)
        target_sequence = torch.cat([
            start_positions[batch_index : batch_index + 1],
            target_positions[batch_index, :count],
            end_positions[batch_index : batch_index + 1],
        ], dim=0)
        if len(predicted_sequence) >= 3:
            predicted_acceleration = torch.stack([
                predicted_sequence[2] - 2.0 * predicted_sequence[1]
                + predicted_sequence[0],
                predicted_sequence[-1] - 2.0 * predicted_sequence[-2]
                + predicted_sequence[-3],
            ])
            target_acceleration = torch.stack([
                target_sequence[2] - 2.0 * target_sequence[1]
                + target_sequence[0],
                target_sequence[-1] - 2.0 * target_sequence[-2]
                + target_sequence[-3],
            ])
            endpoint_acceleration_rows.append(F.smooth_l1_loss(
                predicted_acceleration,
                target_acceleration,
                reduction="mean",
            ))
        else:
            endpoint_acceleration_rows.append(target.new_tensor(0.0))
        if len(predicted_sequence) >= 4:
            predicted_jerk = torch.stack([
                predicted_sequence[3] - 3.0 * predicted_sequence[2]
                + 3.0 * predicted_sequence[1] - predicted_sequence[0],
                predicted_sequence[-1] - 3.0 * predicted_sequence[-2]
                + 3.0 * predicted_sequence[-3] - predicted_sequence[-4],
            ])
            target_jerk = torch.stack([
                target_sequence[3] - 3.0 * target_sequence[2]
                + 3.0 * target_sequence[1] - target_sequence[0],
                target_sequence[-1] - 3.0 * target_sequence[-2]
                + 3.0 * target_sequence[-3] - target_sequence[-4],
            ])
            endpoint_jerk_rows.append(F.smooth_l1_loss(
                predicted_jerk,
                target_jerk,
                reduction="mean",
            ))
        else:
            endpoint_jerk_rows.append(target.new_tensor(0.0))
    losses["endpoint_acceleration"] = weighted_mean(
        torch.stack(endpoint_acceleration_rows), sample_weight
    )
    losses["endpoint_jerk"] = weighted_mean(
        torch.stack(endpoint_jerk_rows), sample_weight
    )

    spectral, high_frequency = _spectral_loss(
        predicted_positions, target_positions, lengths
    )
    losses["spectral"] = spectral
    losses["high_frequency"] = high_frequency

    contact = differentiable_contact_losses(
        predicted,
        target,
        mask,
        aux["contact_logits"],
        sample_weight,
        contact_confidence=contact_confidence,
        fps=float(weights.get("fps", 30.0)),
        penetration_tolerance=float(
            weights.get("penetration_tolerance", 0.008)
        ),
        swing_clearance=float(weights.get("swing_clearance", 0.025)),
    )
    losses.update(contact)
    losses.update(latent_regularisation(latent))

    total = target.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total, losses, latent


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_reconstruction_epoch(
    system: V32ContactINRSystem,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    weights: Dict[str, float],
    amp: bool,
    grad_clip: float,
    detach_latent: bool = False,
) -> Dict[str, float]:
    training = optimizer is not None
    system.train(training)
    totals: Dict[str, List[float]] = {}
    scaler = make_grad_scaler(amp and training and device.type == "cuda")
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
                total, losses, _ = reconstruction_losses(
                    system,
                    batch,
                    device,
                    weights,
                    detach_latent=detach_latent,
                )
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in system.parameters() if p.requires_grad],
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            values = {
                **{name: float(value.detach().cpu())
                   for name, value in losses.items()},
                "total": float(total.detach().cpu()),
            }
            for name, value in values.items():
                totals.setdefault(name, []).append(value)
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in totals.items()
    }


@torch.no_grad()
def encode_subset(
    system: V32ContactINRSystem,
    dataset: TransitionDataset,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    workers: int,
) -> Dict[str, torch.Tensor]:
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    system.eval()
    rows: Dict[str, List[torch.Tensor]] = {
        key: [] for key in (
            "latent", "condition", "weight", "target", "mask",
            "start", "end", "music", "length",
            "start_velocity", "end_velocity", "contact_confidence",
        )
    }
    for batch in loader:
        target = batch["target"].to(device).float()
        mask = batch["mask"].to(device).float()
        start = batch["start"].to(device).float()
        end = batch["end"].to(device).float()
        music = batch["music"].to(device).float()
        length = batch["length"].to(device).float().reshape(-1, 1)
        start_velocity = batch["start_velocity"].to(device).float()
        end_velocity = batch["end_velocity"].to(device).float()
        condition = system.condition(
            start, end, start_velocity, end_velocity, music, length
        )
        latent = system.encode(target, mask, condition)
        rows["latent"].append(latent.cpu())
        rows["condition"].append(condition.cpu())
        rows["weight"].append(batch["sample_weight"].reshape(-1).float())
        for key in (
            "target", "mask", "start", "end", "music", "length",
            "start_velocity", "end_velocity", "contact_confidence",
        ):
            value = batch[key]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            rows[key].append(value.cpu().float())
    return {key: torch.cat(value, dim=0) for key, value in rows.items()}


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Dict[str, torch.Tensor]) -> None:
        self.rows = {key: value.float() for key, value in rows.items()}

    def __len__(self) -> int:
        return len(self.rows["latent"])

    def __getitem__(self, index: int):
        return {key: value[index] for key, value in self.rows.items()}


def decoded_diffusion_loss(
    system: V32ContactINRSystem,
    predicted_normalised: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    limit: int,
    device: torch.device,
    weights: Dict[str, float],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    count = min(int(limit), len(predicted_normalised))
    if count <= 0:
        zero = predicted_normalised.new_tensor(0.0)
        return zero, {}
    latent = (
        predicted_normalised[:count] * latent_std + latent_mean
    )
    start = batch["start"][:count].to(device)
    end = batch["end"][:count].to(device)
    start_velocity = batch["start_velocity"][:count].to(device)
    end_velocity = batch["end_velocity"][:count].to(device)
    condition = batch["condition"][:count].to(device)
    length = batch["length"][:count].to(device).reshape(-1, 1)
    target = batch["target"][:count].to(device)
    mask = batch["mask"][:count].to(device)
    sample_weight = batch["weight"][:count].to(device)
    contact_confidence = batch["contact_confidence"][:count].to(device)
    coordinate = _coordinates(length, target.shape[1])
    boundary_state = boundary_state_from_training_batch(
        start, target, end, length
    )
    predicted, aux = system.decode(
        latent,
        start,
        end,
        start_velocity,
        end_velocity,
        condition,
        coordinate,
        length,
        return_aux=True,
        boundary_state=boundary_state,
    )
    target = project_motion_rotations_torch(target)
    predicted = project_motion_rotations_torch(predicted)
    rotation = weighted_mean(
        masked_per_sample(
            geodesic_rotation_error_torch(predicted, target).square(),
            mask,
        ),
        sample_weight,
    )
    pred_pos = motion_to_joint_positions_torch(predicted)
    target_pos = motion_to_joint_positions_torch(target)
    fk = weighted_mean(
        masked_per_sample((pred_pos - target_pos).square(), mask),
        sample_weight,
    )
    contact = differentiable_contact_losses(
        predicted,
        target,
        mask,
        aux["contact_logits"],
        sample_weight,
        contact_confidence=contact_confidence,
        fps=float(weights.get("fps", 30.0)),
    )
    contact_total = contact_loss_total(contact, weights)
    total = (
        float(weights.get("decoded_rotation", 0.40)) * rotation
        + float(weights.get("decoded_fk", 0.60)) * fk
        + float(weights.get("decoded_contact", 1.00)) * contact_total
    )
    return total, {
        "decoded_rotation": rotation,
        "decoded_fk": fk,
        "decoded_contact_total": contact_total,
        **{f"decoded_{k}": v for k, v in contact.items()},
    }


def latent_diffusion_epoch(
    system: V32ContactINRSystem,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None,
    ema: EMA | None,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    diffusion_steps: int,
    condition_dropout: float,
    decoded_weight: float,
    decoded_batch_limit: int,
    decoded_weights: Dict[str, float],
    device: torch.device,
    amp: bool,
    grad_clip: float,
    seed: int | None = None,
) -> Dict[str, float]:
    training = optimizer is not None
    system.diffusion.train(training)
    scaler = make_grad_scaler(amp and training and device.type == "cuda")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    _, _, alpha_bar = linear_beta_schedule(diffusion_steps, device)
    totals: Dict[str, List[float]] = {}
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            latent = batch["latent"].to(device)
            condition = batch["condition"].to(device)
            weight = batch["weight"].to(device).reshape(-1).clamp_min(1e-4)
            normalised = (latent - latent_mean) / latent_std
            timestep = torch.randint(
                0, diffusion_steps, (len(latent),),
                device=device, generator=generator,
            )
            noise = torch.randn(
                normalised.shape,
                device=device,
                dtype=normalised.dtype,
                generator=generator,
            )
            alpha = alpha_bar[timestep].reshape(-1, 1)
            noisy = (
                torch.sqrt(alpha) * normalised
                + torch.sqrt(1.0 - alpha) * noise
            )
            if training and condition_dropout > 0.0:
                keep = (
                    torch.rand((len(condition), 1), device=device)
                    >= condition_dropout
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
                predicted_noise = system.diffusion(
                    noisy, time, model_condition
                )
                noise_per = (
                    predicted_noise - noise
                ).square().mean(dim=-1)
                noise_loss = (
                    noise_per * weight
                ).sum() / weight.sum()
                predicted_x0 = (
                    noisy - torch.sqrt(1.0 - alpha) * predicted_noise
                ) / torch.sqrt(alpha).clamp_min(1e-6)
                x0_per = F.smooth_l1_loss(
                    predicted_x0, normalised, reduction="none"
                ).mean(dim=-1)
                x0_loss = (
                    x0_per * weight
                ).sum() / weight.sum()
                decoded, decoded_metrics = decoded_diffusion_loss(
                    system,
                    predicted_x0,
                    batch,
                    latent_mean,
                    latent_std,
                    decoded_batch_limit,
                    device,
                    decoded_weights,
                )
                total = noise_loss + 0.10 * x0_loss + float(
                    decoded_weight
                ) * decoded
            if training:
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    system.diffusion.parameters(), grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                if ema is not None:
                    ema.update(system.diffusion)
            row = {
                "noise": noise_loss,
                "x0": x0_loss,
                "decoded": decoded,
                "total": total,
                **decoded_metrics,
            }
            for name, value in row.items():
                totals.setdefault(name, []).append(
                    float(value.detach().cpu())
                )
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in totals.items()
    }


def save_autoencoder_checkpoint(
    path: Path,
    system: V32ContactINRSystem,
    config: V32INRConfig,
    dataset: TransitionDataset,
    split: Dict[str, object],
    epoch: int,
    val_metrics: Dict[str, float],
    stage: str,
) -> None:
    torch.save({
        "system": system.state_dict(),
        "config": {
            "architecture": "v34_continuous_c3_contact_inr_autoencoder",
            "model": asdict(config),
            "stage": stage,
        },
        "dataset_meta": dataset.meta,
        "split": split,
        "epoch": epoch,
        "best_val_loss": val_metrics.get("total"),
        "val_metrics": val_metrics,
    }, path)


def load_system_checkpoint(
    path: str | Path,
    system: V32ContactINRSystem,
    device: torch.device,
) -> Dict[str, object]:
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )
    state = checkpoint.get("system", checkpoint.get("model"))
    if state is None:
        raise RuntimeError(f"No system state in {path}")
    system.load_state_dict(state, strict=False)
    return checkpoint


def train_reconstruction_stage(
    stage_name: str,
    system: V32ContactINRSystem,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Dict[str, float],
    epochs: int,
    patience: int,
    amp: bool,
    grad_clip: float,
    checkpoint_path: Path,
    config: V32INRConfig,
    dataset: TransitionDataset,
    split: Dict[str, object],
    detach_latent: bool,
) -> List[Dict[str, object]]:
    history: List[Dict[str, object]] = []
    best, bad = float("inf"), 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=optimizer.param_groups[0]["lr"] * 0.08
    )
    for epoch in range(1, epochs + 1):
        train = run_reconstruction_epoch(
            system, train_loader, optimizer, device,
            weights, amp, grad_clip, detach_latent,
        )
        val = run_reconstruction_epoch(
            system, val_loader, None, device,
            weights, False, grad_clip, detach_latent,
        )
        scheduler.step()
        row = {
            "stage": stage_name,
            "epoch": epoch,
            "train": train,
            "val": val,
            "val_loss": val["total"],
        }
        history.append(row)
        print(
            f"[{stage_name}] epoch={epoch:04d} "
            f"train={train['total']:.6f} val={val['total']:.6f} "
            f"contact={val.get('contact_bce',0):.6f} "
            f"skate={val.get('contact_skate',0):.6f} "
            f"penetration={val.get('foot_penetration',0):.6f}",
            flush=True,
        )
        if val["total"] < best:
            best, bad = val["total"], 0
            save_autoencoder_checkpoint(
                checkpoint_path, system, config, dataset,
                split, epoch, val, stage_name,
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"[EARLY STOP] {stage_name} patience={patience}")
                break
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--stage",
        choices=["all", "autoencoder", "contact_finetune", "diffusion"],
        default="all",
    )
    parser.add_argument("--autoencoder_ckpt", default="")
    parser.add_argument("--include_synthetic", type=int, default=1)
    parser.add_argument("--ae_epochs", type=int, default=220)
    parser.add_argument("--contact_epochs", type=int, default=90)
    parser.add_argument("--diffusion_epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--contact_batch_size", type=int, default=40)
    parser.add_argument("--latent_batch_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--contact_lr", type=float, default=4e-5)
    parser.add_argument("--diffusion_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--condition_dim", type=int, default=256)
    parser.add_argument("--encoder_hidden", type=int, default=320)
    parser.add_argument("--inr_hidden", type=int, default=320)
    parser.add_argument("--inr_layers", type=int, default=6)
    parser.add_argument("--fourier_bands", type=int, default=5)
    parser.add_argument("--diffusion_hidden", type=int, default=512)
    parser.add_argument("--diffusion_blocks", type=int, default=6)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--rotation_residual_scale", type=float, default=0.16)
    parser.add_argument("--root_y_residual_scale", type=float, default=0.045)
    parser.add_argument("--contact_logit_scale", type=float, default=3.0)
    parser.add_argument("--condition_dropout", type=float, default=0.10)
    parser.add_argument("--decoded_weight", type=float, default=0.08)
    parser.add_argument("--decoded_batch_limit", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--ae_patience", type=int, default=55)
    parser.add_argument("--contact_patience", type=int, default=30)
    parser.add_argument("--diffusion_patience", type=int, default=70)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--fps", type=float, default=30.0)

    # Reconstruction and dynamics.
    parser.add_argument("--w_rotation", type=float, default=0.35)
    parser.add_argument("--w_root", type=float, default=0.20)
    parser.add_argument("--w_fk", type=float, default=0.42)
    parser.add_argument("--w_velocity", type=float, default=0.50)
    parser.add_argument("--w_acceleration", type=float, default=0.26)
    parser.add_argument("--w_jerk", type=float, default=0.10)
    parser.add_argument("--w_angular_velocity", type=float, default=0.22)
    parser.add_argument("--w_endpoint", type=float, default=0.90)
    parser.add_argument("--w_endpoint_acceleration", type=float, default=0.45)
    parser.add_argument("--w_endpoint_jerk", type=float, default=0.20)
    parser.add_argument("--w_spectral", type=float, default=0.18)
    parser.add_argument("--w_high_frequency", type=float, default=0.10)
    parser.add_argument("--w_latent_variance", type=float, default=0.04)
    parser.add_argument("--w_latent_covariance", type=float, default=0.01)
    parser.add_argument("--w_latent_mean", type=float, default=0.002)

    # Differentiable contact.
    parser.add_argument("--w_contact_bce", type=float, default=0.35)
    parser.add_argument("--w_contact_skate", type=float, default=1.20)
    parser.add_argument("--w_contact_height", type=float, default=0.45)
    parser.add_argument("--w_foot_penetration", type=float, default=0.80)
    parser.add_argument("--w_swing_clearance", type=float, default=0.12)
    parser.add_argument("--w_contact_temporal", type=float, default=0.08)
    parser.add_argument("--w_contact_binary", type=float, default=0.02)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output = Path(args.out_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TransitionDataset(
        args.data, include_synthetic=bool(args.include_synthetic)
    )
    train_indices, val_indices, split = source_disjoint_split(
        dataset, args.val_ratio, args.seed
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = V32INRConfig(
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
        diffusion_steps=args.diffusion_steps,
        residual_rotation_scale=args.rotation_residual_scale,
        residual_root_y_scale=args.root_y_residual_scale,
        contact_logit_scale=args.contact_logit_scale,
        max_len=int(dataset.target.shape[1]),
        dropout=args.dropout,
    )
    system = V32ContactINRSystem(config).to(device)

    weights = {
        "rotation": args.w_rotation,
        "root": args.w_root,
        "fk": args.w_fk,
        "velocity": args.w_velocity,
        "acceleration": args.w_acceleration,
        "jerk": args.w_jerk,
        "angular_velocity": args.w_angular_velocity,
        "endpoint": args.w_endpoint,
        "endpoint_acceleration": args.w_endpoint_acceleration,
        "endpoint_jerk": args.w_endpoint_jerk,
        "spectral": args.w_spectral,
        "high_frequency": args.w_high_frequency,
        "latent_variance": args.w_latent_variance,
        "latent_covariance": args.w_latent_covariance,
        "latent_mean": args.w_latent_mean,
        "contact_bce": args.w_contact_bce,
        "contact_skate": args.w_contact_skate,
        "contact_height": args.w_contact_height,
        "foot_penetration": args.w_foot_penetration,
        "swing_clearance": args.w_swing_clearance,
        "contact_temporal": args.w_contact_temporal,
        "contact_binary": args.w_contact_binary,
        "contact_confidence_mean": 0.0,
        "fps": args.fps,
        "decoded_rotation": 0.40,
        "decoded_fk": 0.60,
        "decoded_contact": 1.00,
    }

    history: List[Dict[str, object]] = []
    ae_path = checkpoint_dir / "autoencoder_best.pt"
    contact_path = checkpoint_dir / "contact_finetuned_best.pt"

    if args.stage in {"all", "autoencoder"}:
        optimizer = torch.optim.AdamW(
            system.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        history.extend(train_reconstruction_stage(
            "autoencoder",
            system,
            train_loader,
            val_loader,
            optimizer,
            device,
            weights,
            args.ae_epochs,
            args.ae_patience,
            bool(args.amp),
            args.grad_clip,
            ae_path,
            config,
            dataset,
            split,
            detach_latent=False,
        ))
        load_system_checkpoint(ae_path, system, device)

    elif args.autoencoder_ckpt:
        load_system_checkpoint(args.autoencoder_ckpt, system, device)
    else:
        raise RuntimeError(
            "--autoencoder_ckpt is required when skipping autoencoder stage"
        )

    if args.stage in {"all", "contact_finetune"}:
        real_dataset = TransitionDataset(
            args.data, include_synthetic=False, real_only=True
        )
        real_train, real_val, real_split = source_disjoint_split(
            real_dataset, args.val_ratio, args.seed + 17
        )
        contact_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(real_dataset, real_train.tolist()),
            batch_size=args.contact_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        contact_val_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(real_dataset, real_val.tolist()),
            batch_size=args.contact_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for parameter in system.parameters():
            parameter.requires_grad_(False)
        for parameter in system.inr.parameters():
            parameter.requires_grad_(True)

        contact_weights = dict(weights)
        contact_weights.update({
            "rotation": args.w_rotation * 0.45,
            "root": args.w_root * 0.60,
            "fk": args.w_fk * 0.65,
            "velocity": args.w_velocity * 0.75,
            "acceleration": args.w_acceleration * 0.85,
            "jerk": args.w_jerk * 1.25,
            "endpoint": args.w_endpoint * 1.20,
            "endpoint_acceleration": args.w_endpoint_acceleration * 1.50,
            "endpoint_jerk": args.w_endpoint_jerk * 1.75,
            "contact_bce": args.w_contact_bce * 1.50,
            "contact_skate": args.w_contact_skate * 1.80,
            "contact_height": args.w_contact_height * 1.50,
            "foot_penetration": args.w_foot_penetration * 1.60,
            "swing_clearance": args.w_swing_clearance * 1.20,
            "latent_variance": 0.0,
            "latent_covariance": 0.0,
            "latent_mean": 0.0,
        })
        optimizer = torch.optim.AdamW(
            system.inr.parameters(),
            lr=args.contact_lr,
            weight_decay=args.weight_decay,
        )
        history.extend(train_reconstruction_stage(
            "contact_finetune",
            system,
            contact_train_loader,
            contact_val_loader,
            optimizer,
            device,
            contact_weights,
            args.contact_epochs,
            args.contact_patience,
            bool(args.amp),
            args.grad_clip,
            contact_path,
            config,
            real_dataset,
            real_split,
            detach_latent=True,
        ))
        load_system_checkpoint(contact_path, system, device)
        for parameter in system.parameters():
            parameter.requires_grad_(True)

    if args.stage == "contact_finetune":
        (output / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "BEST_V32_CONTACT_FINETUNE_CKPT.txt").write_text(
            str(contact_path), encoding="utf-8"
        )
        return

    # Freeze encoder/condition/INR for latent diffusion.
    for module in (
        system.condition_encoder, system.encoder, system.inr
    ):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    train_rows = encode_subset(
        system, dataset, train_indices,
        args.batch_size, device, args.num_workers,
    )
    val_rows = encode_subset(
        system, dataset, val_indices,
        args.batch_size, device, args.num_workers,
    )
    latent_mean = train_rows["latent"].mean(dim=0).to(device)
    latent_std = train_rows["latent"].std(dim=0).clamp_min(1e-4).to(device)

    train_loader_latent = torch.utils.data.DataLoader(
        LatentDataset(train_rows),
        batch_size=args.latent_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader_latent = torch.utils.data.DataLoader(
        LatentDataset(val_rows),
        batch_size=args.latent_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        system.diffusion.parameters(),
        lr=args.diffusion_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.diffusion_epochs, 1),
        eta_min=args.diffusion_lr * 0.08,
    )
    ema = EMA(system.diffusion, args.ema_decay)
    best, bad = float("inf"), 0
    final_path = checkpoint_dir / "best.pt"

    for epoch in range(1, args.diffusion_epochs + 1):
        train = latent_diffusion_epoch(
            system,
            train_loader_latent,
            optimizer,
            ema,
            latent_mean,
            latent_std,
            args.diffusion_steps,
            args.condition_dropout,
            args.decoded_weight,
            args.decoded_batch_limit,
            weights,
            device,
            bool(args.amp),
            args.grad_clip,
        )
        val = latent_diffusion_epoch(
            system,
            val_loader_latent,
            None,
            None,
            latent_mean,
            latent_std,
            args.diffusion_steps,
            0.0,
            args.decoded_weight,
            args.decoded_batch_limit,
            weights,
            device,
            False,
            args.grad_clip,
            seed=args.seed + 9127,
        )
        scheduler.step()
        row = {
            "stage": "diffusion",
            "epoch": epoch,
            "train": train,
            "val": val,
            "val_loss": val["total"],
        }
        history.append(row)
        print(
            f"[diffusion] epoch={epoch:04d} "
            f"train={train['total']:.6f} val={val['total']:.6f} "
            f"decoded={val.get('decoded',0):.6f} "
            f"contact={val.get('decoded_contact_total',0):.6f}",
            flush=True,
        )
        if val["total"] < best:
            best, bad = val["total"], 0
            torch.save({
                "system": system.state_dict(),
                "ema_diffusion": ema.state_dict(),
                "config": {
                    "architecture":
                        "v34_continuous_c3_contact_inr_latent_diffusion",
                    "model": asdict(config),
                    "diffusion_steps": args.diffusion_steps,
                    "contact_weights": weights,
                    "include_synthetic": bool(args.include_synthetic),
                },
                "latent_mean": latent_mean.detach().cpu(),
                "latent_std": latent_std.detach().cpu(),
                "dataset_meta": dataset.meta,
                "split": split,
                "epoch": epoch,
                "best_val_loss": best,
                "val_metrics": val,
            }, final_path)
        else:
            bad += 1
            if bad >= args.diffusion_patience:
                print(
                    f"[EARLY STOP] diffusion patience={args.diffusion_patience}"
                )
                break

    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for name in (
        "BEST_V27_TRANSITION_DIFFUSION_CKPT.txt",
        "BEST_V32_CONTACT_INR_CKPT.txt",
    ):
        (output / name).write_text(str(final_path), encoding="utf-8")
    print(f"[SAVED] {final_path}")


if __name__ == "__main__":
    main()
