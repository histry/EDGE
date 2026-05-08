"""Differentiable contact / foot-lock losses for EDGE 151-D motion.

V11.1 logic-gap fix
-------------------
The previous DCL path could blindly trust target contact channels. In the current
Dunhuang processed data those channels can be very dense / almost always-on,
which turns DCL into an over-strong "lock feet all the time" loss.

This version adds explicit contact-source control and an `auto` mode:

    EDGE_DCL_CONTACT_SOURCE=auto|target_channels|target_fk|pred_fk_height|pred_fk_height_speed|hybrid

Recommended first formal DCL run:
    EDGE_DCL_CONTACT_SOURCE=auto
    EDGE_DCL_MAX_TARGET_CONTACT_RATIO=0.85
    EDGE_DCL_FALLBACK_CONTACT_SOURCE=pred_fk_height

If target contact channels are too dense, auto mode falls back to a height-based
FK contact mask. This closes the "94-100% foot slide / all-contact" logic gap
without requiring perfectly annotated contact labels.

Representation:
    [0:4] contacts, [4:7] root xyz, [7:151] 24 joints * 6D rotations
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

FOOT_JOINTS = [7, 8, 10, 11]
CONTACT_SLICE = slice(0, 4)
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


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip()


def contact_mask_from_channels(motion_physical: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Return [B,T,4] boolean contact mask from 151D contact channels."""
    if motion_physical is None or motion_physical.ndim != 3 or motion_physical.shape[-1] != 151:
        raise ValueError("contact_mask_from_channels expects [B,T,151]")
    return motion_physical[..., CONTACT_SLICE].clamp(0.0, 1.0) > float(threshold)


def _foot_heights_and_speed(joints: torch.Tensor, fps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    feet = joints[:, :, FOOT_JOINTS, :]  # [B,T,4,3]
    heights = feet[..., 1]
    horiz_speed = torch.zeros_like(heights)
    if feet.shape[1] > 1:
        horiz_speed[:, 1:] = (
            torch.linalg.norm(feet[:, 1:, :, [0, 2]] - feet[:, :-1, :, [0, 2]], dim=-1)
            * float(fps)
        )
        horiz_speed[:, 0] = horiz_speed[:, 1]
    return heights, horiz_speed


def contact_mask_from_fk(
    joints: torch.Tensor,
    fps: float = 30.0,
    height_threshold: float = 0.035,
    speed_threshold: float = 0.08,
    mode: str = "height",
    min_contact_ratio: float = 0.002,
) -> torch.Tensor:
    """Derive [B,T,4] contacts from foot height/speed.

    mode:
      - height: low foot = contact; catches sliding low feet.
      - height_speed: low and slow = contact; stricter but may miss sliding.
      - speed: slow feet regardless of height; diagnostic only.
    """
    if joints is None or joints.ndim != 4:
        raise ValueError("contact_mask_from_fk expects joints [B,T,J,3]")

    heights, horiz_speed = _foot_heights_and_speed(joints, fps=fps)
    floor = torch.quantile(heights.reshape(heights.shape[0], -1), 0.02, dim=1)
    low = heights <= floor[:, None, None] + float(height_threshold)
    slow = horiz_speed <= float(speed_threshold)

    mode = str(mode or "height").lower()
    if mode in {"height_speed", "height+speed", "strict"}:
        strict = low & slow
        if bool((strict.float().mean() < float(min_contact_ratio)).detach().cpu().item()):
            return low.detach()
        return strict.detach()
    if mode in {"speed", "slow"}:
        return slow.detach()
    return low.detach()


def foot_velocity_from_joints(joints: torch.Tensor, horizontal_only: bool = True) -> torch.Tensor:
    """Return [B,T-1,4] foot velocity squared."""
    feet = joints[:, :, FOOT_JOINTS, :]
    delta = feet[:, 1:] - feet[:, :-1]
    if horizontal_only:
        delta = delta[..., [0, 2]]
    return delta.pow(2).sum(dim=-1)


def _select_contact_mask(
    pred_physical: torch.Tensor,
    target_physical: Optional[torch.Tensor],
    pred_joints: torch.Tensor,
    target_joints: Optional[torch.Tensor],
    fps: float,
    contact_threshold: float,
    height_threshold: float,
    speed_threshold: float,
    contact_source: str,
) -> Tuple[torch.Tensor, str, Dict[str, float]]:
    contact_source = str(contact_source or "auto").lower()
    max_target_ratio = env_float("EDGE_DCL_MAX_TARGET_CONTACT_RATIO", 0.85)
    fallback_source = env_str("EDGE_DCL_FALLBACK_CONTACT_SOURCE", "pred_fk_height").lower()

    def from_source(source: str) -> Tuple[torch.Tensor, str]:
        source = str(source or "target_channels").lower()
        if source in {"target_channels", "channels", "target_contact_channels"}:
            if target_physical is None:
                return contact_mask_from_channels(pred_physical.detach(), contact_threshold), "pred_contact_channels"
            return contact_mask_from_channels(target_physical.detach(), contact_threshold), "target_contact_channels"
        if source in {"pred_channels", "pred_contact_channels"}:
            return contact_mask_from_channels(pred_physical.detach(), contact_threshold), "pred_contact_channels"
        if source in {"target_fk", "target_fk_height"}:
            if target_joints is None:
                return from_source("target_channels")
            return contact_mask_from_fk(
                target_joints.detach(), fps=fps, height_threshold=height_threshold,
                speed_threshold=speed_threshold, mode="height"
            ), "target_fk_height"
        if source in {"target_fk_height_speed", "target_fk_strict"}:
            if target_joints is None:
                return from_source("target_channels")
            return contact_mask_from_fk(
                target_joints.detach(), fps=fps, height_threshold=height_threshold,
                speed_threshold=speed_threshold, mode="height_speed"
            ), "target_fk_height_speed"
        if source in {"pred_fk", "pred_fk_height"}:
            return contact_mask_from_fk(
                pred_joints.detach(), fps=fps, height_threshold=height_threshold,
                speed_threshold=speed_threshold, mode="height"
            ), "pred_fk_height"
        if source in {"pred_fk_height_speed", "pred_fk_strict"}:
            return contact_mask_from_fk(
                pred_joints.detach(), fps=fps, height_threshold=height_threshold,
                speed_threshold=speed_threshold, mode="height_speed"
            ), "pred_fk_height_speed"
        if source in {"hybrid", "hybrid_min"}:
            ch, _ = from_source("target_channels")
            fk, _ = from_source("pred_fk_height")
            return (ch & fk).detach(), "hybrid_target_channel_and_pred_fk_height"
        raise ValueError(f"Unknown EDGE_DCL_CONTACT_SOURCE={source}")

    if contact_source == "auto":
        if target_physical is not None:
            ch = contact_mask_from_channels(target_physical.detach(), contact_threshold)
            ratio = float(ch.float().mean().detach().cpu().item())
            if ratio <= max_target_ratio:
                return ch, "target_contact_channels", {
                    "auto_target_contact_ratio": ratio,
                    "auto_fallback": 0.0,
                }
            fallback, name = from_source(fallback_source)
            return fallback, f"auto_fallback_{name}", {
                "auto_target_contact_ratio": ratio,
                "auto_fallback": 1.0,
                "auto_max_target_ratio": float(max_target_ratio),
            }
        fallback, name = from_source(fallback_source)
        return fallback, f"auto_fallback_{name}", {"auto_fallback": 1.0}

    mask, name = from_source(contact_source)
    return mask, name, {"auto_fallback": 0.0}


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
    contact_source: str = "auto",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    zero = pred_physical.new_tensor(0.0)
    if pred_physical.ndim != 3 or pred_physical.shape[-1] != 151:
        return zero, {"dcl_valid": 0.0, "dcl_reason": "not_151d"}

    if use_fk_contact_labels and contact_source in {"", "auto", "target_channels"}:
        contact_source = "target_fk"

    try:
        contacts, resolved_source, extra = _select_contact_mask(
            pred_physical=pred_physical,
            target_physical=target_physical,
            pred_joints=pred_joints,
            target_joints=target_joints,
            fps=fps,
            contact_threshold=contact_threshold,
            height_threshold=height_threshold,
            speed_threshold=speed_threshold,
            contact_source=contact_source,
        )
    except Exception as exc:
        return zero, {"dcl_valid": 0.0, "dcl_reason": f"contact_mask_failed:{exc}"}

    if contacts.shape[1] < 2:
        return zero, {"dcl_valid": 0.0, "dcl_reason": "too_short", "contact_source": resolved_source}

    contact_pairs = contacts[:, 1:] & contacts[:, :-1]
    contact_ratio = float(contacts.float().mean().detach().cpu().item())
    contact_pair_ratio = float(contact_pairs.float().mean().detach().cpu().item())

    min_pair_ratio = env_float("EDGE_DCL_MIN_CONTACT_PAIR_RATIO", 0.001)
    if not bool(contact_pairs.any().detach().cpu().item()) or contact_pair_ratio < min_pair_ratio:
        debug = {
            "dcl_valid": 0.0,
            "dcl_reason": "no_contact_pairs",
            "contact_source": resolved_source,
            "contact_ratio": contact_ratio,
            "contact_pair_ratio": contact_pair_ratio,
        }
        debug.update(extra)
        return zero, debug

    vel_sq = foot_velocity_from_joints(pred_joints, horizontal_only=horizontal_only)
    loss = vel_sq[contact_pairs].mean()

    dense_scale = 1.0
    dense_ratio = env_float("EDGE_DCL_DENSE_CONTACT_DOWNSCALE_RATIO", 0.0)
    dense_weight = env_float("EDGE_DCL_DENSE_CONTACT_DOWNSCALE_WEIGHT", 0.5)
    if dense_ratio > 0 and contact_ratio > dense_ratio:
        dense_scale = float(dense_weight)
        loss = loss * dense_scale

    debug = {
        "dcl_valid": 1.0,
        "contact_source": resolved_source,
        "contact_ratio": contact_ratio,
        "contact_pair_ratio": contact_pair_ratio,
        "foot_vel_sq_mean": float(vel_sq.detach().mean().cpu().item()),
        "horizontal_only": float(bool(horizontal_only)),
        "dense_scale": float(dense_scale),
    }
    debug.update(extra)
    return loss, debug
