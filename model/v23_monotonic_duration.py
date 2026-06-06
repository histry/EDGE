#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V23-v2.3 bin-residual duration and two-stage monotonic time-warp model.

Stage 1 predicts:
- a duration bin;
- an intra-bin residual;
- whether temporal correction is necessary.

Stage 2 freezes the duration branch and learns a monotonic source-time map tau,
conditioned on a teacher-forced/predicted natural duration.  The model never
predicts pose residuals.  Final motion is produced only by SO(3)-aware temporal
resampling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

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
    def __init__(self, channels: int, cond_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, channels * 4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.condition(condition).chunk(4, dim=-1)
        hidden = self.norm1(x)
        hidden = hidden * (1.0 + gamma1[..., None]) + beta1[..., None]
        hidden = self.dropout(self.conv1(F.silu(hidden)))
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + gamma2[..., None]) + beta2[..., None]
        hidden = self.conv2(F.silu(hidden))
        return x + hidden


class TemporalFiLMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        dropout: float,
        dilations: Sequence[int],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, 1)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [FiLMBlock1D(hidden_dim, hidden_dim, int(d), dropout) for d in dilations]
        )
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, hidden_dim)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(features)
        condition_hidden = self.condition_projection(condition)
        for block in self.blocks:
            hidden = block(hidden, condition_hidden)
        return F.silu(self.output_norm(hidden)), condition_hidden


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
    temperature: float = 0.010,
) -> torch.Tensor:
    if turn_start.ndim == 1:
        turn_start = turn_start[:, None]
    if turn_end.ndim == 1:
        turn_end = turn_end[:, None]
    left = torch.sigmoid((tau - turn_start) / float(temperature))
    right = torch.sigmoid((turn_end - tau) / float(temperature))
    return (left * right).mean(dim=1)


def _validate_duration_edges(edges: Iterable[float], window_len: int) -> list[float]:
    values = [float(value) for value in edges]
    if len(values) < 3:
        raise ValueError("duration_edges must contain at least three boundaries")
    if any(right <= left for left, right in zip(values[:-1], values[1:])):
        raise ValueError(f"duration_edges must be strictly increasing: {values}")
    if values[0] < 1.0 or values[-1] - 1.0 > float(window_len):
        raise ValueError(f"duration_edges outside window: {values}, window={window_len}")
    return values


class V23MonotonicDurationNet(nn.Module):
    """Decoupled duration and monotonic time-warp network."""

    def __init__(
        self,
        motion_dim: int = 151,
        condition_dim: int = 17,
        hidden_dim: int = 128,
        dropout: float = 0.18,
        duration_edges: Sequence[float] = (12, 24, 37, 50, 63, 76, 89),
        window_len: int = 120,
        duration_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        tau_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.window_len = int(window_len)
        edges = _validate_duration_edges(duration_edges, self.window_len)
        self.register_buffer("duration_edges", torch.tensor(edges, dtype=torch.float32), persistent=True)
        self.num_duration_bins = len(edges) - 1

        input_dim = motion_dim * 2 + 3
        self.duration_encoder = TemporalFiLMEncoder(
            input_dim, condition_dim, hidden_dim, dropout, duration_dilations
        )
        self.tau_encoder = TemporalFiLMEncoder(
            input_dim, condition_dim, hidden_dim, dropout, tau_dilations
        )

        pooled_dim = hidden_dim * 4
        self.duration_bin_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_duration_bins),
        )
        self.duration_residual_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_duration_bins),
        )
        self.edit_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, max(32, hidden_dim // 2)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )

        self.duration_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.tau_increment_head = nn.Conv1d(hidden_dim, 1, 1)

        nn.init.zeros_(self.duration_bin_head[-1].weight)
        nn.init.zeros_(self.duration_bin_head[-1].bias)
        nn.init.zeros_(self.duration_residual_head[-1].weight)
        nn.init.zeros_(self.duration_residual_head[-1].bias)
        nn.init.zeros_(self.edit_head[-1].weight)
        nn.init.zeros_(self.edit_head[-1].bias)
        nn.init.zeros_(self.tau_increment_head.weight)
        nn.init.zeros_(self.tau_increment_head.bias)

    @property
    def duration_min_frames(self) -> float:
        return float(self.duration_edges[0].item())

    @property
    def duration_max_frames(self) -> float:
        return float(self.duration_edges[-1].item() - 1.0)

    def _motion_features(self, motion: torch.Tensor, edit_mask: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = motion.shape
        velocity = torch.zeros_like(motion)
        if time_steps > 1:
            velocity[:, 1:] = motion[:, 1:] - motion[:, :-1]
            velocity[:, 0] = velocity[:, 1]
        yaw = root_yaw_velocity_dps(motion)
        yaw = torch.cat([yaw[:, :1], yaw], dim=1) / 300.0
        yaw = torch.clamp(yaw, -3.0, 3.0)[..., None]
        time = torch.linspace(0.0, 1.0, time_steps, device=motion.device, dtype=motion.dtype)
        time = time[None, :, None].expand(batch_size, -1, -1)
        mask = edit_mask[..., None].to(motion.dtype)
        return torch.cat([motion, velocity, mask, time, yaw], dim=-1).transpose(1, 2)

    def _pool_duration(
        self,
        hidden: torch.Tensor,
        condition_hidden: torch.Tensor,
        edit_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence = hidden.transpose(1, 2)
        mask_weight = edit_mask.to(sequence.dtype).clamp(0.0, 1.0)
        masked = (sequence * (0.05 + mask_weight)[..., None]).sum(dim=1)
        masked = masked / (0.05 + mask_weight).sum(dim=1, keepdim=True).clamp_min(1.0)
        global_mean = sequence.mean(dim=1)
        global_max = sequence.amax(dim=1)
        return torch.cat([masked, global_mean, global_max, condition_hidden], dim=-1)

    def _duration_candidates(self, residual_logits: torch.Tensor) -> torch.Tensor:
        lower = self.duration_edges[:-1].to(residual_logits.dtype)[None]
        upper = (self.duration_edges[1:] - 1.0).to(residual_logits.dtype)[None]
        residual = torch.sigmoid(residual_logits)
        return lower + residual * (upper - lower).clamp_min(1.0)

    def predict_duration(
        self,
        motion: torch.Tensor,
        edit_mask: torch.Tensor,
        condition: torch.Tensor,
        use_hard_duration: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        if motion.ndim != 3 or motion.shape[-1] != self.motion_dim:
            raise ValueError(f"motion must be [B,T,{self.motion_dim}], got {tuple(motion.shape)}")
        if edit_mask.ndim == 3 and edit_mask.shape[-1] == 1:
            edit_mask = edit_mask[..., 0]
        if edit_mask.ndim != 2:
            raise ValueError(f"edit_mask must be [B,T], got {tuple(edit_mask.shape)}")
        if condition.ndim != 2 or condition.shape[-1] != self.condition_dim:
            raise ValueError(f"condition must be [B,{self.condition_dim}], got {tuple(condition.shape)}")

        features = self._motion_features(motion, edit_mask)
        hidden, condition_hidden = self.duration_encoder(features, condition)
        pooled = self._pool_duration(hidden, condition_hidden, edit_mask)
        bin_logits = self.duration_bin_head(pooled)
        residual_logits = self.duration_residual_head(pooled)
        candidates = self._duration_candidates(residual_logits)
        probabilities = torch.softmax(bin_logits, dim=-1)
        soft_duration = (probabilities * candidates).sum(dim=-1)
        hard_bin = torch.argmax(bin_logits, dim=-1)
        hard_duration = candidates.gather(1, hard_bin[:, None]).squeeze(1)
        confidence = probabilities.gather(1, hard_bin[:, None]).squeeze(1)
        if use_hard_duration is None:
            use_hard_duration = not self.training
        duration_frames = hard_duration if use_hard_duration else soft_duration
        edit_logit = self.edit_head(pooled).squeeze(-1)
        return {
            "duration_bin_logits": bin_logits,
            "duration_bin_probabilities": probabilities,
            "duration_residual_logits": residual_logits,
            "duration_candidates": candidates,
            "duration_soft_frames": soft_duration,
            "duration_hard_frames": hard_duration,
            "duration_frames": duration_frames,
            "duration_bin_index": hard_bin,
            "duration_bin_confidence": confidence,
            "edit_logit": edit_logit,
            "edit_probability": torch.sigmoid(edit_logit),
        }

    def predict_tau(
        self,
        motion: torch.Tensor,
        edit_mask: torch.Tensor,
        condition: torch.Tensor,
        duration_frames: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = self._motion_features(motion, edit_mask)
        hidden, condition_hidden = self.tau_encoder(features, condition)
        normalized_duration = (duration_frames / float(motion.shape[1])).clamp(0.0, 1.0)
        duration_feature = self.duration_embedding(normalized_duration[:, None])
        hidden = hidden + duration_feature[..., None] + 0.15 * condition_hidden[..., None]
        raw_logits = self.tau_increment_head(hidden).squeeze(1)[:, :-1]
        interval_gate = torch.maximum(edit_mask[:, :-1], edit_mask[:, 1:]).to(raw_logits.dtype)
        logits = raw_logits * (0.05 + 0.95 * interval_gate)
        increments = F.softplus(torch.clamp(logits, -10.0, 10.0)) + 1e-5
        cumulative = torch.cumsum(increments, dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-8)
        tau = torch.cat(
            [
                torch.zeros((motion.shape[0], 1), device=motion.device, dtype=motion.dtype),
                cumulative,
            ],
            dim=1,
        )
        return {"tau": tau, "increments": increments, "duration_for_tau": duration_frames}

    def forward(
        self,
        motion: torch.Tensor,
        edit_mask: torch.Tensor,
        condition: torch.Tensor,
        duration_override_frames: torch.Tensor | None = None,
        use_hard_duration: bool | None = None,
    ) -> Dict[str, torch.Tensor]:
        duration_output = self.predict_duration(
            motion, edit_mask, condition, use_hard_duration=use_hard_duration
        )
        duration_for_tau = (
            duration_override_frames
            if duration_override_frames is not None
            else duration_output["duration_frames"].detach()
        )
        tau_output = self.predict_tau(motion, edit_mask, condition, duration_for_tau)
        return {**duration_output, **tau_output}

    def set_train_stage(self, stage: str) -> None:
        stage = str(stage).lower()
        if stage not in {"duration", "timewarp", "joint"}:
            raise ValueError(f"Unsupported stage: {stage}")
        for parameter in self.parameters():
            parameter.requires_grad = stage == "joint"
        if stage == "duration":
            modules = [
                self.duration_encoder,
                self.duration_bin_head,
                self.duration_residual_head,
                self.edit_head,
            ]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
        elif stage == "timewarp":
            modules = [self.tau_encoder, self.duration_embedding, self.tau_increment_head]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True


def load_v23_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    duration_edges = config.get("duration_edges", [12, 24, 37, 50, 63, 76, 89])
    model = V23MonotonicDurationNet(
        motion_dim=int(config.get("motion_dim", 151)),
        condition_dim=int(config.get("condition_dim", 17)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.18)),
        duration_edges=duration_edges,
        window_len=int(config.get("window_len", 120)),
        duration_dilations=config.get("duration_dilations", [1, 2, 4, 8, 16, 32]),
        tau_dilations=config.get("tau_dilations", [1, 2, 4, 8, 16, 32]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return {
        "model": model,
        "config": config,
        "epoch": checkpoint.get("epoch", -1),
        "val_loss": checkpoint.get("val_loss", float("inf")),
        "selection_score": checkpoint.get("selection_score", float("inf")),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "stage": checkpoint.get("stage", config.get("stage", "unknown")),
    }
