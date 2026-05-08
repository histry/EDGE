"""Differentiable contact / foot-lock losses for EDGE 151-D motion.

This file is intentionally independent from the main diffusion implementation.
It is used by ``differentiable_contact_loss_patch.py`` to replace/augment the
existing foot sliding loss at training time.

Representation:
    [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotations

Main objective:
    L_contact = mean_t,foot C_t * || v_foot_world(t) ||^2

where C_t can come from:
1. target contact channels, if available;
2. FK-derived target foot height + speed mask;
3. predicted contact channels as a fallback.

Environment switches:
    EDGE_DCL_USE_FK_CONTACT_LABELS=0|1
    EDGE_DCL_CONTACT_THRESHOLD=0.5
    EDGE_DCL_HEIGHT_THRESHOLD=0.035
    EDGE_DCL_SPEED_THRESHOLD=0.08
    EDGE_DCL_HORIZONTAL_ONLY=1
    EDGE_DCL_MIN_CONTACT_RATIO=0.002
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

FOOT_JOINTS = [7, 8, 10, 11]
CONTACT_SLICE = slice(0, 4)
ROOT_SLICE = slice(4, 7)
ROT_SLICE = slice(7, 151)
_TRUE = {"1", "true", "yes", "y", "on"}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def maybe_unnormalize(normalizer, x: torch.Tensor) -> torch.Tensor:
    if normalizer is None:
        return x
    out = normalizer.unnormalize(x)
    if not torch.is_tensor(out):
        out = torch.as_tensor(out, device=x.device, dtype=x.dtype)
    return out.to(device=x.device, dtype=x.dtype)


def contact_mask_from_channels(
    motion_physical: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return [B,T,4] boolean contact mask from 151D contact channels."""
    if motion_physical.shape[-1] != 151:
        return torch.zeros(
            (*motion_physical.shape[:2], 4),
            dtype=torch.bool,
            device=motion_physical.device,
        )
    return motion_physical[..., CONTACT_SLICE].clamp(0.0, 1.0) > float(threshold)


def contact_mask_from_fk(
    joints: torch.Tensor,
    fps: float = 30.0,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.08,
    min_contact_ratio: float = 0.002,
) -> torch.Tensor:
    """Derive [B,T,4] contacts from foot height and horizontal speed.

    This is useful when dataset contact channels are missing or noisy.
    It is height-first: if speed gating removes nearly all contacts, it falls
    back to height-only so sliding low feet are still treated as contacts.
    """
    feet = joints[:, :, FOOT_JOINTS, :]  # [B,T,4,3]
    heights = feet[..., 1]
    floor = torch.quantile(heights.reshape(heights.shape[0], -1), 0.02, dim=1)
    low = heights <= floor[:, None, None] + float(height_threshold)

    horiz_speed = torch.zeros_like(heights)
    if feet.shape[1] > 1:
        horiz_speed[:, 1:] = (
            torch.linalg.norm(feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1)
            * float(fps)
        )
        horiz_speed[:, 0] = horiz_speed[:, 1]

    slow = horiz_speed <= float(speed_threshold)
    strict = low & slow
    ratio = strict.float().mean()
    if bool((ratio < float(min_contact_ratio)).detach().cpu().item()):
        return low
    return strict


def foot_velocity_from_joints(
    joints: torch.Tensor,
    horizontal_only: bool = True,
) -> torch.Tensor:
    """Return [B,T-1,4] foot velocity squared."""
    feet = joints[:, :, FOOT_JOINTS, :]
    delta = feet[:, 1:] - feet[:, :-1]
    if horizontal_only:
        delta = delta[..., [0, 2]]
    return delta.pow(2).sum(dim=-1)


def differentiable_contact_velocity_loss(
    pred_physical: torch.Tensor,
    target_physical: Optional[torch.Tensor],
    pred_joints: torch.Tensor,
    target_joints: Optional[torch.Tensor] = None,
    fps: float = 30.0,
    contact_threshold: float = 0.5,
    use_fk_contact_labels: bool = False,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.08,
    horizontal_only: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute differentiable contact velocity loss for predicted motion.

    The gradient flows through ``pred_joints`` into predicted rotations/root.
    The contact mask is treated as a target gate and detached.
    """
    zero = pred_physical.new_tensor(0.0)
    if pred_physical.ndim != 3 or pred_physical.shape[-1] != 151:
        return zero, {"dcl_valid": 0.0, "dcl_reason": "not_151d"}

    if use_fk_contact_labels and target_joints is not None:
        contacts = contact_mask_from_fk(
            target_joints.detach(),
            fps=fps,
            height_threshold=height_threshold,
            speed_threshold=speed_threshold,
        )
        contact_source = "target_fk_height_speed"
    elif target_physical is not None:
        contacts = contact_mask_from_channels(
            target_physical.detach(),
            threshold=contact_threshold,
        )
        contact_source = "target_contact_channels"
    else:
        contacts = contact_mask_from_channels(
            pred_physical.detach(),
            threshold=contact_threshold,
        )
        contact_source = "pred_contact_channels"

    if contacts.shape[1] < 2:
        return zero, {"dcl_valid": 0.0, "dcl_reason": "too_short"}

    # Require contact at both adjacent frames to define a locked-foot interval.
    contact_pairs = contacts[:, 1:] & contacts[:, :-1]  # [B,T-1,4]
    if not bool(contact_pairs.any().detach().cpu().item()):
        return zero, {
            "dcl_valid": 0.0,
            "dcl_reason": "no_contact_pairs",
            "contact_source": contact_source,
            "contact_ratio": float(contacts.float().mean().detach().cpu().item()),
        }

    vel_sq = foot_velocity_from_joints(pred_joints, horizontal_only=horizontal_only)
    loss = vel_sq[contact_pairs].mean()
    debug = {
        "dcl_valid": 1.0,
        "contact_source": contact_source,
        "contact_ratio": float(contacts.float().mean().detach().cpu().item()),
        "contact_pair_ratio": float(contact_pairs.float().mean().detach().cpu().item()),
        "foot_vel_sq_mean": float(vel_sq.detach().mean().cpu().item()),
        "horizontal_only": float(bool(horizontal_only)),
    }
    return loss, debug
