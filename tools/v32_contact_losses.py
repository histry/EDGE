#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differentiable contact and foot-physics losses for EDGE V32."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from tools.v29_motion_geometry import (
    CONTACT,
    motion_to_joint_positions_torch,
)

# Contact channel order is aligned with the established evaluator:
# left ankle, right ankle, left toe, right toe.
FOOT_JOINTS = (7, 8, 10, 11)


def _expand_mask(mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    result = mask
    while result.ndim < values.ndim:
        result = result.unsqueeze(-1)
    return result.expand_as(values).to(values.dtype)


def masked_per_sample(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    expanded = _expand_mask(mask, values)
    dimensions = tuple(range(1, values.ndim))
    return (
        (values * expanded).sum(dim=dimensions)
        / expanded.sum(dim=dimensions).clamp_min(1.0)
    )


def weighted_mean(
    values: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    w = weights.reshape(-1).clamp_min(1e-4)
    return (values * w).sum() / w.sum().clamp_min(1e-4)


@torch.no_grad()
def estimate_ground_height(
    target_motion: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    positions = motion_to_joint_positions_torch(target_motion)
    foot_y = positions[..., FOOT_JOINTS, 1]
    target_contact = target_motion[..., CONTACT].clamp(0.0, 1.0)
    valid = mask[..., None]
    weight = target_contact * valid
    denominator = weight.sum(dim=(1, 2))
    contact_ground = (
        (foot_y * weight).sum(dim=(1, 2))
        / denominator.clamp_min(1.0)
    )
    masked = foot_y.masked_fill(valid < 0.5, float("inf"))
    fallback = masked.flatten(1).amin(dim=1)
    fallback = torch.where(
        torch.isfinite(fallback), fallback, torch.zeros_like(fallback)
    )
    return torch.where(
        denominator > 0.5, contact_ground, fallback
    ).reshape(-1, 1, 1)


def differentiable_contact_losses(
    predicted_motion: torch.Tensor,
    target_motion: torch.Tensor,
    mask: torch.Tensor,
    contact_logits: torch.Tensor,
    sample_weight: torch.Tensor,
    fps: float = 30.0,
    penetration_tolerance: float = 0.008,
    swing_clearance: float = 0.025,
) -> Dict[str, torch.Tensor]:
    """Compute contact-supervised and geometry-coupled losses.

    Gradients from skate/height/penetration flow through FK into local
    rotations and root height. Contact probabilities remain differentiable.
    """
    target_contact = target_motion[..., CONTACT].clamp(0.0, 1.0)
    contact_prob = torch.sigmoid(contact_logits)
    bce = F.binary_cross_entropy_with_logits(
        contact_logits, target_contact, reduction="none"
    )
    losses: Dict[str, torch.Tensor] = {
        "contact_bce": weighted_mean(
            masked_per_sample(bce, mask), sample_weight
        )
    }

    positions = motion_to_joint_positions_torch(predicted_motion)
    feet = positions[..., FOOT_JOINTS, :]
    ground = estimate_ground_height(target_motion, mask)
    foot_y = feet[..., 1]
    horizontal = feet[..., (0, 2)]

    if feet.shape[1] > 1:
        velocity = (
            horizontal[:, 1:] - horizontal[:, :-1]
        ) * float(fps)
        speed = torch.linalg.norm(velocity, dim=-1)
        pair_contact = 0.5 * (
            contact_prob[:, 1:] + contact_prob[:, :-1]
        )
        pair_mask = mask[:, 1:] * mask[:, :-1]
        skate = pair_contact * F.smooth_l1_loss(
            speed, torch.zeros_like(speed), reduction="none"
        )
        losses["contact_skate"] = weighted_mean(
            masked_per_sample(skate, pair_mask), sample_weight
        )

        stable_target = 1.0 - torch.abs(
            target_contact[:, 1:] - target_contact[:, :-1]
        )
        contact_delta = torch.abs(
            contact_prob[:, 1:] - contact_prob[:, :-1]
        ) * stable_target
        losses["contact_temporal"] = weighted_mean(
            masked_per_sample(contact_delta, pair_mask), sample_weight
        )
    else:
        zero = predicted_motion.new_tensor(0.0)
        losses["contact_skate"] = zero
        losses["contact_temporal"] = zero

    contact_height = contact_prob * F.smooth_l1_loss(
        foot_y, ground.expand_as(foot_y), reduction="none"
    )
    losses["contact_height"] = weighted_mean(
        masked_per_sample(contact_height, mask), sample_weight
    )

    penetration = F.relu(
        ground - foot_y - float(penetration_tolerance)
    ).square()
    losses["foot_penetration"] = weighted_mean(
        masked_per_sample(penetration, mask), sample_weight
    )

    swing_target = (1.0 - target_contact)
    clearance = F.relu(
        ground + float(swing_clearance) - foot_y
    ).square() * swing_target
    losses["swing_clearance"] = weighted_mean(
        masked_per_sample(clearance, mask), sample_weight
    )

    binary_regularizer = (
        contact_prob * (1.0 - contact_prob)
    )
    losses["contact_binary"] = weighted_mean(
        masked_per_sample(binary_regularizer, mask), sample_weight
    )
    return losses


def contact_loss_total(
    losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    if not losses:
        raise ValueError("No contact losses")
    first = next(iter(losses.values()))
    total = first.new_tensor(0.0)
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    return total


def contact_metrics_from_motion(
    motion: torch.Tensor,
    mask: torch.Tensor,
    fps: float = 30.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    positions = motion_to_joint_positions_torch(motion)
    feet = positions[..., FOOT_JOINTS, :]
    if feet.shape[1] <= 1:
        zero = motion.new_zeros((motion.shape[0],))
        return zero, zero
    speed = torch.linalg.norm(
        (feet[:, 1:] - feet[:, :-1]) * float(fps), dim=-1
    )
    contact = motion[..., CONTACT].clamp(0.0, 1.0)
    pair_contact = 0.5 * (contact[:, 1:] + contact[:, :-1])
    pair_mask = mask[:, 1:] * mask[:, :-1]
    slip = masked_per_sample(speed * pair_contact, pair_mask)
    contact_rate = masked_per_sample(contact, mask)
    return slip, contact_rate
