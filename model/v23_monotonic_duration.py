#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V23 monotonic duration-and-time-warp network.

The network never predicts a pose residual.  It predicts:
1. a monotonic source-time map tau for a fixed-length motion window; and
2. the number of output frames that the detected turn should occupy.

The final motion is produced only by SO(3)-aware temporal resampling, which
preserves the original motion manifold much better than frame-wise pose edits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

CONTACT = slice(0, 4)
ROOT = slice(4, 7)
ROT = slice(7, 151)
ROOT_ROT6D = slice(7, 13)


class FiLMBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.cond = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.cond(cond).chunk(4, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + gamma1[..., None]) + beta1[..., None]
        h = self.conv1(F.silu(h))
        h = self.dropout(h)
        h = self.norm2(h)
        h = h * (1.0 + gamma2[..., None]) + beta2[..., None]
        h = self.conv2(F.silu(h))
        return x + h


def root_yaw_velocity_dps(motion: torch.Tensor, fps: float = 30.0) -> torch.Tensor:
    """Signed global-heading yaw velocity, consistent with runtime evaluation."""
    root = rotation_6d_to_matrix(motion[..., ROOT_ROT6D])
    yaw = torch.atan2(root[..., 0, 2], root[..., 2, 2])
    delta = yaw[:, 1:] - yaw[:, :-1]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return delta * float(fps) * (180.0 / np.pi)


def warp_motion_so3(motion: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Resample [B,T,151] motion using monotonic normalized source positions tau."""
    if motion.ndim != 3 or motion.shape[-1] != 151:
        raise ValueError(f"motion must be [B,T,151], got {tuple(motion.shape)}")
    if tau.ndim != 2 or tau.shape[:2] != motion.shape[:2]:
        raise ValueError(f"tau must be [B,T], got {tuple(tau.shape)}")

    b, t, _ = motion.shape
    pos = torch.clamp(tau, 0.0, 1.0) * float(t - 1)
    lower = torch.floor(pos).long().clamp(0, t - 1)
    upper = (lower + 1).clamp(0, t - 1)
    alpha = (pos - lower.to(pos.dtype)).clamp(0.0, 1.0)

    batch = torch.arange(b, device=motion.device)[:, None]
    lower_motion = motion[batch, lower]
    upper_motion = motion[batch, upper]

    nearest = torch.where(alpha < 0.5, lower, upper)
    contacts = motion[batch, nearest, CONTACT]
    root = torch.lerp(lower_motion[..., ROOT], upper_motion[..., ROOT], alpha[..., None])

    rot_all = rotation_6d_to_matrix(motion[..., ROT].reshape(b, t, 24, 6))
    r0 = rot_all[batch, lower]
    r1 = rot_all[batch, upper]
    relative = torch.matmul(r0.transpose(-1, -2), r1)
    axis_angle = matrix_to_axis_angle(relative)
    delta = axis_angle_to_matrix(axis_angle * alpha[..., None, None])
    rotation = torch.matmul(r0, delta)
    rot6d = matrix_to_rotation_6d(rotation).reshape(b, t, 144)
    return torch.cat([contacts, root, rot6d], dim=-1)


def soft_turn_duration_ratio(
    tau: torch.Tensor,
    turn_start: torch.Tensor,
    turn_end: torch.Tensor,
    temperature: float = 0.012,
) -> torch.Tensor:
    """Differentiable number of output frames mapped into the input turn interval."""
    if turn_start.ndim == 1:
        turn_start = turn_start[:, None]
    if turn_end.ndim == 1:
        turn_end = turn_end[:, None]
    left = torch.sigmoid((tau - turn_start) / float(temperature))
    right = torch.sigmoid((turn_end - tau) / float(temperature))
    return (left * right).mean(dim=1)


class V23MonotonicDurationNet(nn.Module):
    """Predict a monotonic time map and explicit target turn duration."""

    def __init__(
        self,
        motion_dim: int = 151,
        condition_dim: int = 17,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)

        # motion + first velocity + mask + time + signed yaw velocity
        input_dim = motion_dim * 2 + 3
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        dilations = (1, 2, 4, 8, 16, 32, 16, 8, 4, 2)
        self.blocks = nn.ModuleList(
            [FiLMBlock1D(hidden_dim, hidden_dim, d, dropout) for d in dilations]
        )
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, hidden_dim)
        self.increment_head = nn.Conv1d(hidden_dim, 1, 1)
        nn.init.zeros_(self.increment_head.weight)
        nn.init.zeros_(self.increment_head.bias)

        self.duration_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.duration_head[-1].weight)
        nn.init.constant_(self.duration_head[-1].bias, -1.38629436)  # sigmoid ~= 0.20

    def forward(
        self,
        motion: torch.Tensor,
        edit_mask: torch.Tensor,
        condition: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if motion.ndim != 3 or motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion must be [B,T,{self.motion_dim}], got {tuple(motion.shape)}")
        if edit_mask.ndim == 3 and edit_mask.shape[-1] == 1:
            edit_mask = edit_mask[..., 0]
        if edit_mask.ndim != 2:
            raise ValueError(f"edit_mask must be [B,T], got {tuple(edit_mask.shape)}")
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(
                f"condition must be [B,{self.condition_dim}], got {tuple(condition.shape)}"
            )

        b, t, _ = motion.shape
        velocity = torch.zeros_like(motion)
        if t > 1:
            velocity[:, 1:] = motion[:, 1:] - motion[:, :-1]
            velocity[:, 0] = velocity[:, 1]
        yaw = root_yaw_velocity_dps(motion)
        yaw = torch.cat([yaw[:, :1], yaw], dim=1) / 600.0
        yaw = torch.clamp(yaw, -2.0, 2.0)[..., None]
        time = torch.linspace(0.0, 1.0, t, device=motion.device, dtype=motion.dtype)
        time = time[None, :, None].expand(b, -1, -1)
        mask = edit_mask[..., None].to(motion.dtype)
        x = torch.cat([motion, velocity, mask, time, yaw], dim=-1).transpose(1, 2)

        h = self.input_projection(x)
        cond = self.condition_projection(condition)
        for block in self.blocks:
            h = block(h, cond)
        h = F.silu(self.output_norm(h))

        # Zero logits -> equal increments -> exact identity map at initialization.
        logits = self.increment_head(h).squeeze(1)[:, :-1]
        increments = torch.exp(torch.clamp(logits, -6.0, 6.0)) + 1e-5
        cumulative = torch.cumsum(increments, dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        tau = torch.cat(
            [torch.zeros((b, 1), device=motion.device, dtype=motion.dtype), cumulative],
            dim=1,
        )

        mask_weight = edit_mask.to(h.dtype).clamp_min(0.0)
        pooled = (h.transpose(1, 2) * mask_weight[..., None]).sum(dim=1)
        pooled = pooled / mask_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        duration_ratio = torch.sigmoid(self.duration_head(torch.cat([pooled, cond], dim=-1))).squeeze(-1)

        return {
            "tau": tau,
            "increments": increments,
            "duration_ratio": duration_ratio,
        }


def load_v23_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = V23MonotonicDurationNet(
        motion_dim=int(config.get("motion_dim", 151)),
        condition_dim=int(config.get("condition_dim", 17)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return {
        "model": model,
        "config": config,
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", float("inf")),
    }
