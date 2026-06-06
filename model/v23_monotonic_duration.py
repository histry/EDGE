#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V23-v2 natural-duration and monotonic time-warp network.

The network never predicts a pose residual. It predicts:
1. a bounded natural event duration;
2. a monotonic source-time map tau; and
3. an edit probability used by runtime safety gating.

The time map is parameterized by positive increments, so monotonicity and exact
endpoints are guaranteed by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
    root = rotation_6d_to_matrix(motion[..., ROOT_ROT6D])
    yaw = torch.atan2(root[..., 0, 2], root[..., 2, 2])
    delta = yaw[:, 1:] - yaw[:, :-1]
    delta = torch.atan2(torch.sin(delta), torch.cos(delta))
    return delta * float(fps) * (180.0 / np.pi)


def warp_motion_so3(motion: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Resample [B,T,151] motion using normalized source positions tau."""
    if motion.ndim != 3 or motion.shape[-1] != 151:
        raise ValueError(f"motion must be [B,T,151], got {tuple(motion.shape)}")
    if tau.ndim != 2 or tau.shape[:2] != motion.shape[:2]:
        raise ValueError(f"tau must be [B,T], got {tuple(tau.shape)}")

    batch_size, time_steps, _ = motion.shape
    positions = torch.clamp(tau, 0.0, 1.0) * float(time_steps - 1)
    lower = torch.floor(positions).long().clamp(0, time_steps - 1)
    upper = (lower + 1).clamp(0, time_steps - 1)
    alpha = (positions - lower.to(positions.dtype)).clamp(0.0, 1.0)
    batch = torch.arange(batch_size, device=motion.device)[:, None]

    lower_motion = motion[batch, lower]
    upper_motion = motion[batch, upper]
    nearest = torch.where(alpha < 0.5, lower, upper)
    contacts = motion[batch, nearest, CONTACT]
    root = torch.lerp(lower_motion[..., ROOT], upper_motion[..., ROOT], alpha[..., None])

    rotations = rotation_6d_to_matrix(motion[..., ROT].reshape(batch_size, time_steps, 24, 6))
    r0 = rotations[batch, lower]
    r1 = rotations[batch, upper]
    relative = torch.matmul(r0.transpose(-1, -2), r1)
    axis_angle = matrix_to_axis_angle(relative)
    delta = axis_angle_to_matrix(axis_angle * alpha[..., None, None])
    rotation = torch.matmul(r0, delta)
    rot6d = matrix_to_rotation_6d(rotation).reshape(batch_size, time_steps, 144)
    return torch.cat([contacts, root, rot6d], dim=-1)


def soft_turn_duration_ratio(
    tau: torch.Tensor,
    turn_start: torch.Tensor,
    turn_end: torch.Tensor,
    temperature: float = 0.012,
) -> torch.Tensor:
    if turn_start.ndim == 1:
        turn_start = turn_start[:, None]
    if turn_end.ndim == 1:
        turn_end = turn_end[:, None]
    left = torch.sigmoid((tau - turn_start) / float(temperature))
    right = torch.sigmoid((turn_end - tau) / float(temperature))
    return (left * right).mean(dim=1)


class V23MonotonicDurationNet(nn.Module):
    """Predict natural duration, edit confidence and a monotonic time map."""

    def __init__(
        self,
        motion_dim: int = 151,
        condition_dim: int = 17,
        hidden_dim: int = 256,
        dropout: float = 0.10,
        duration_min_frames: float = 8.0,
        duration_max_frames: float = 56.0,
        window_len: int = 72,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.duration_min_frames = float(duration_min_frames)
        self.duration_max_frames = float(duration_max_frames)
        self.window_len = int(window_len)
        if not (0.0 < self.duration_min_frames < self.duration_max_frames <= self.window_len):
            raise ValueError("Invalid duration range")

        # observed motion + first velocity + edit mask + normalized time + signed yaw
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
            [FiLMBlock1D(hidden_dim, hidden_dim, dilation, dropout) for dilation in dilations]
        )
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, hidden_dim)

        pooled_dim = hidden_dim * 2
        self.duration_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.edit_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.duration_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.increment_head = nn.Conv1d(hidden_dim, 1, 1)

        # Start from identity tau and the middle of the valid duration range.
        nn.init.zeros_(self.increment_head.weight)
        nn.init.zeros_(self.increment_head.bias)
        nn.init.zeros_(self.duration_head[-1].weight)
        nn.init.zeros_(self.duration_head[-1].bias)
        nn.init.zeros_(self.edit_head[-1].weight)
        nn.init.zeros_(self.edit_head[-1].bias)

    def _bounded_duration(self, duration_logit: torch.Tensor, time_steps: int) -> torch.Tensor:
        minimum = self.duration_min_frames / float(time_steps)
        maximum = self.duration_max_frames / float(time_steps)
        return minimum + (maximum - minimum) * torch.sigmoid(duration_logit)

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
            raise ValueError(f"condition must be [B,{self.condition_dim}], got {tuple(condition.shape)}")

        batch_size, time_steps, _ = motion.shape
        velocity = torch.zeros_like(motion)
        if time_steps > 1:
            velocity[:, 1:] = motion[:, 1:] - motion[:, :-1]
            velocity[:, 0] = velocity[:, 1]
        yaw = root_yaw_velocity_dps(motion)
        yaw = torch.cat([yaw[:, :1], yaw], dim=1) / 600.0
        yaw = torch.clamp(yaw, -2.0, 2.0)[..., None]
        time = torch.linspace(0.0, 1.0, time_steps, device=motion.device, dtype=motion.dtype)
        time = time[None, :, None].expand(batch_size, -1, -1)
        mask = edit_mask[..., None].to(motion.dtype)
        x = torch.cat([motion, velocity, mask, time, yaw], dim=-1).transpose(1, 2)

        hidden = self.input_projection(x)
        cond = self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden, cond)
        hidden = F.silu(self.output_norm(hidden))

        mask_weight = edit_mask.to(hidden.dtype).clamp_min(0.0)
        pooled = (hidden.transpose(1, 2) * (0.10 + mask_weight)[..., None]).sum(dim=1)
        pooled = pooled / (0.10 + mask_weight).sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_cond = torch.cat([pooled, cond], dim=-1)

        duration_logit = self.duration_head(pooled_cond).squeeze(-1)
        duration_ratio = self._bounded_duration(duration_logit, time_steps)
        edit_logit = self.edit_head(pooled_cond).squeeze(-1)
        edit_probability = torch.sigmoid(edit_logit)

        duration_feature = self.duration_embedding(duration_ratio[:, None])
        hidden_for_tau = hidden + duration_feature[..., None]
        raw_logits = self.increment_head(hidden_for_tau).squeeze(1)[:, :-1]

        # Restrict non-uniform timing primarily to the event and its context.
        interval_gate = torch.maximum(edit_mask[:, :-1], edit_mask[:, 1:]).to(raw_logits.dtype)
        logits = raw_logits * (0.08 + 0.92 * interval_gate)
        increments = F.softplus(torch.clamp(logits, -10.0, 10.0)) + 1e-5
        cumulative = torch.cumsum(increments, dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        tau = torch.cat(
            [torch.zeros((batch_size, 1), device=motion.device, dtype=motion.dtype), cumulative],
            dim=1,
        )

        return {
            "tau": tau,
            "increments": increments,
            "duration_ratio": duration_ratio,
            "duration_frames": duration_ratio * float(time_steps),
            "duration_logit": duration_logit,
            "edit_logit": edit_logit,
            "edit_probability": edit_probability,
        }


def load_v23_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    model = V23MonotonicDurationNet(
        motion_dim=int(config.get("motion_dim", 151)),
        condition_dim=int(config.get("condition_dim", 17)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        dropout=float(config.get("dropout", 0.10)),
        duration_min_frames=float(config.get("duration_min_frames", 8.0)),
        duration_max_frames=float(config.get("duration_max_frames", 56.0)),
        window_len=int(config.get("window_len", 72)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return {
        "model": model,
        "config": config,
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "selection_score": checkpoint.get("selection_score", float("inf")),
        "split": checkpoint.get("split", {}),
    }
