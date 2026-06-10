#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trainable hyperbolic music-motion alignment for EDGE V30.

The original V27 path projects CLAP through a fixed random matrix and only uses
it to perturb handcrafted 12D query fields.  V30 instead learns music and
multi-part motion encoders in a shared Poincare ball.  Retrieval supervision can
come from explicit phrase-event pairs, the existing V21 router triplets, or a
mixture of both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GeometricAlignmentConfig:
    rule_dim: int = 12
    clap_dim: int = 512
    motion_raw_dim: int = 12
    motion_mmr_dim: int = 64
    hidden_dim: int = 256
    embed_dim: int = 32
    dropout: float = 0.10
    curvature: float = 1.0
    minimum_radius: float = 0.08
    maximum_radius: float = 0.92


def _project_ball(x: torch.Tensor, curvature: float) -> torch.Tensor:
    c = max(float(curvature), 1e-6)
    maximum = (1.0 - 1e-5) / math.sqrt(c)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(1e-8)
    return x * torch.minimum(torch.ones_like(norm), maximum / norm)


def expmap0(tangent: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    c = max(float(curvature), 1e-6)
    sqrt_c = math.sqrt(c)
    norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp_min(1e-8)
    result = torch.tanh(sqrt_c * norm) * tangent / (sqrt_c * norm)
    return _project_ball(result, c)


def logmap0(point: torch.Tensor, curvature: float = 1.0) -> torch.Tensor:
    c = max(float(curvature), 1e-6)
    sqrt_c = math.sqrt(c)
    point = _project_ball(point, c)
    norm = torch.linalg.vector_norm(point, dim=-1, keepdim=True).clamp_min(1e-8)
    scaled = (sqrt_c * norm).clamp(max=1.0 - 1e-5)
    return torch.atanh(scaled) * point / (sqrt_c * norm)


def poincare_distance_pairwise(
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Pairwise distance matrix between [N,D] and [M,D]."""
    c = max(float(curvature), 1e-6)
    sqrt_c = math.sqrt(c)
    left = _project_ball(left, c)
    right = _project_ball(right, c)
    left_norm = left.square().sum(dim=-1, keepdim=True)
    right_norm = right.square().sum(dim=-1).reshape(1, -1)
    difference = (left[:, None] - right[None]).square().sum(dim=-1)
    denominator = (
        (1.0 - c * left_norm)
        * (1.0 - c * right_norm)
    ).clamp_min(1e-7)
    argument = 1.0 + 2.0 * c * difference / denominator
    return torch.acosh(argument.clamp_min(1.0 + 1e-6)) / sqrt_c


def poincare_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    return poincare_distance_pairwise(left, right, curvature).diagonal()


class FeatureTower(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MusicHyperbolicEncoder(torch.nn.Module):
    def __init__(self, config: GeometricAlignmentConfig) -> None:
        super().__init__()
        self.config = config
        self.rule = FeatureTower(
            config.rule_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.clap = FeatureTower(
            max(config.clap_dim, 1),
            config.hidden_dim,
            config.hidden_dim,
            config.dropout,
        )
        self.fusion = FeatureTower(
            config.hidden_dim * 2 + 1,
            config.hidden_dim,
            config.embed_dim + 1,
            config.dropout,
        )

    def forward(
        self,
        rule: torch.Tensor,
        clap: torch.Tensor,
        clap_valid: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if rule.shape[-1] < self.config.rule_dim:
            rule = F.pad(rule, (0, self.config.rule_dim - rule.shape[-1]))
        rule = rule[..., : self.config.rule_dim]
        if self.config.clap_dim <= 0:
            clap = rule.new_zeros((len(rule), 1))
            valid = rule.new_zeros((len(rule), 1))
        else:
            if clap.shape[-1] < self.config.clap_dim:
                clap = F.pad(clap, (0, self.config.clap_dim - clap.shape[-1]))
            clap = clap[..., : self.config.clap_dim]
            valid = (
                clap_valid.reshape(-1, 1).to(rule.dtype)
                if clap_valid is not None
                else (torch.linalg.vector_norm(clap, dim=-1, keepdim=True) > 1e-6).to(rule.dtype)
            )
        rule_hidden = self.rule(rule)
        clap_hidden = self.clap(clap) * valid
        output = self.fusion(torch.cat([rule_hidden, clap_hidden, valid], dim=-1))
        direction = F.normalize(output[..., :-1], dim=-1)
        radius_ratio = self.config.minimum_radius + (
            self.config.maximum_radius - self.config.minimum_radius
        ) * torch.sigmoid(output[..., -1:])
        tangent_norm = torch.atanh(radius_ratio.clamp(max=0.98)) / math.sqrt(
            max(self.config.curvature, 1e-6)
        )
        tangent = direction * tangent_norm
        return expmap0(tangent, self.config.curvature), tangent, radius_ratio.squeeze(-1)


class MotionHyperbolicEncoder(torch.nn.Module):
    """Multi-part motion encoder: body, centre/support and gesture/detail."""

    def __init__(self, config: GeometricAlignmentConfig) -> None:
        super().__init__()
        self.config = config
        raw_dim = config.motion_raw_dim
        body_dim = min(raw_dim, 9)
        centre_dim = raw_dim
        self.body = FeatureTower(
            body_dim, config.hidden_dim, config.embed_dim, config.dropout
        )
        self.centre = FeatureTower(
            centre_dim, config.hidden_dim, config.embed_dim, config.dropout
        )
        self.gesture = FeatureTower(
            max(config.motion_mmr_dim, 1),
            config.hidden_dim,
            config.embed_dim,
            config.dropout,
        )
        self.part_weight = torch.nn.Sequential(
            torch.nn.LayerNorm(raw_dim + max(config.motion_mmr_dim, 1)),
            torch.nn.Linear(raw_dim + max(config.motion_mmr_dim, 1), config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, 3),
        )
        self.radius = torch.nn.Sequential(
            torch.nn.LayerNorm(raw_dim),
            torch.nn.Linear(raw_dim, config.hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        raw: torch.Tensor,
        mmr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if raw.shape[-1] < self.config.motion_raw_dim:
            raw = F.pad(raw, (0, self.config.motion_raw_dim - raw.shape[-1]))
        raw = raw[..., : self.config.motion_raw_dim]
        mmr_dim = max(self.config.motion_mmr_dim, 1)
        if mmr.shape[-1] < mmr_dim:
            mmr = F.pad(mmr, (0, mmr_dim - mmr.shape[-1]))
        mmr = mmr[..., :mmr_dim]
        body_input = raw[..., : min(raw.shape[-1], 9)]
        body = F.normalize(self.body(body_input), dim=-1)
        centre = F.normalize(self.centre(raw), dim=-1)
        gesture = F.normalize(self.gesture(mmr), dim=-1)
        weights = torch.softmax(self.part_weight(torch.cat([raw, mmr], dim=-1)), dim=-1)
        # Tangent-space weighted aggregation is stable, differentiable and
        # preserves the interpretable body/centre/gesture decomposition.
        direction = F.normalize(
            weights[..., 0:1] * body
            + weights[..., 1:2] * centre
            + weights[..., 2:3] * gesture,
            dim=-1,
        )
        radius_ratio = self.config.minimum_radius + (
            self.config.maximum_radius - self.config.minimum_radius
        ) * torch.sigmoid(self.radius(raw))
        tangent_norm = torch.atanh(radius_ratio.clamp(max=0.98)) / math.sqrt(
            max(self.config.curvature, 1e-6)
        )
        tangent = direction * tangent_norm
        return (
            expmap0(tangent, self.config.curvature),
            tangent,
            radius_ratio.squeeze(-1),
            weights,
        )


class V30GeometricAligner(torch.nn.Module):
    def __init__(self, config: GeometricAlignmentConfig) -> None:
        super().__init__()
        self.config = config
        self.music = MusicHyperbolicEncoder(config)
        self.motion = MotionHyperbolicEncoder(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(8.0)))

    def encode_music(
        self,
        rule: torch.Tensor,
        clap: torch.Tensor,
        clap_valid: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.music(rule, clap, clap_valid)

    def encode_motion(
        self,
        raw: torch.Tensor,
        mmr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.motion(raw, mmr)

    def similarity(self, music: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return -scale * poincare_distance_pairwise(
            music, motion, self.config.curvature
        )


def geometric_alignment_loss(
    model: V30GeometricAligner,
    music_rule: torch.Tensor,
    music_clap: torch.Tensor,
    clap_valid: torch.Tensor,
    positive_raw: torch.Tensor,
    positive_mmr: torch.Tensor,
    negative_raw: torch.Tensor,
    negative_mmr: torch.Tensor,
    hierarchy_label: torch.Tensor,
    target_radius: torch.Tensor,
    positive_id: torch.Tensor | None = None,
    preference_margin: float = 0.20,
    hierarchy_weight: float = 0.20,
    radius_weight: float = 0.12,
    preference_weight: float = 0.55,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    music_embed, _, music_radius = model.encode_music(
        music_rule, music_clap, clap_valid
    )
    positive_embed, _, motion_radius, part_weights = model.encode_motion(
        positive_raw, positive_mmr
    )
    negative_embed, _, _, _ = model.encode_motion(negative_raw, negative_mmr)

    logits = model.similarity(music_embed, positive_embed)
    if positive_id is None:
        labels = torch.arange(len(logits), device=logits.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )
    else:
        identity = positive_id.reshape(-1)
        positive_mask = identity[:, None] == identity[None, :]
        def multi_positive_nce(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            log_denominator = torch.logsumexp(values, dim=1)
            numerator = values.masked_fill(~mask, -1e4)
            log_numerator = torch.logsumexp(numerator, dim=1)
            return (log_denominator - log_numerator).mean()
        contrastive = 0.5 * (
            multi_positive_nce(logits, positive_mask)
            + multi_positive_nce(logits.transpose(0, 1), positive_mask.transpose(0, 1))
        )
    positive_distance = poincare_distance(
        music_embed, positive_embed, model.config.curvature
    )
    negative_distance = poincare_distance(
        music_embed, negative_embed, model.config.curvature
    )
    preference = F.softplus(
        positive_distance - negative_distance + float(preference_margin)
    ).mean()

    hierarchy_distance = poincare_distance_pairwise(
        positive_embed, positive_embed, model.config.curvature
    )
    same = hierarchy_label[:, None] == hierarchy_label[None, :]
    eye = torch.eye(len(same), device=same.device, dtype=torch.bool)
    positive_mask = same & (~eye)
    negative_mask = (~same) & (~eye)
    hierarchy = positive_embed.new_tensor(0.0)
    if positive_mask.any() and negative_mask.any():
        positive_mean = hierarchy_distance[positive_mask].mean()
        negative_mean = hierarchy_distance[negative_mask].mean()
        hierarchy = F.softplus(positive_mean - negative_mean + 0.10)

    target_radius = target_radius.reshape(-1).clamp(
        model.config.minimum_radius, model.config.maximum_radius
    )
    radius = F.mse_loss(motion_radius, target_radius)
    # Music radius should follow the paired motion's hierarchy specificity.
    radius = radius + 0.5 * F.mse_loss(music_radius, target_radius)
    balance = ((part_weights.mean(dim=0) - 1.0 / 3.0) ** 2).mean()
    total = (
        contrastive
        + preference_weight * preference
        + hierarchy_weight * hierarchy
        + radius_weight * radius
        + 0.02 * balance
    )
    return total, {
        "contrastive": contrastive,
        "preference": preference,
        "hierarchy": hierarchy,
        "radius": radius,
        "part_balance": balance,
        "positive_distance": positive_distance.mean(),
        "negative_distance": negative_distance.mean(),
    }


def load_geometric_aligner(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Tuple[V30GeometricAligner, Dict[str, Any]]:
    checkpoint_path = Path(str(path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    config_values = dict(checkpoint.get("config", {}))
    fields = GeometricAlignmentConfig.__dataclass_fields__
    config = GeometricAlignmentConfig(**{
        key: config_values[key] for key in fields if key in config_values
    })
    model = V30GeometricAligner(config).to(device)
    state = checkpoint.get("ema_model", checkpoint.get("model"))
    if state is None:
        raise RuntimeError("Alignment checkpoint has no model state")
    model.load_state_dict(state)
    model.eval()
    return model, {
        "path": str(checkpoint_path),
        "config": config_values,
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
    }


def encode_music_numpy(
    model: V30GeometricAligner,
    rule: np.ndarray,
    clap: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    device = torch.device(device)
    rule_tensor = torch.from_numpy(np.asarray(rule, np.float32)).to(device)
    clap_tensor = torch.from_numpy(np.asarray(clap, np.float32)).to(device)
    if rule_tensor.ndim == 1:
        rule_tensor = rule_tensor[None]
    if clap_tensor.ndim == 1:
        clap_tensor = clap_tensor[None]
    valid = (torch.linalg.vector_norm(clap_tensor, dim=-1) > 1e-6).float()
    with torch.no_grad():
        embedding, _, _ = model.encode_music(rule_tensor, clap_tensor, valid)
    return embedding.cpu().numpy().astype(np.float32)


def encode_motion_numpy(
    model: V30GeometricAligner,
    raw: np.ndarray,
    mmr: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    device = torch.device(device)
    raw_tensor = torch.from_numpy(np.asarray(raw, np.float32)).to(device)
    mmr_tensor = torch.from_numpy(np.asarray(mmr, np.float32)).to(device)
    if raw_tensor.ndim == 1:
        raw_tensor = raw_tensor[None]
    if mmr_tensor.ndim == 1:
        mmr_tensor = mmr_tensor[None]
    with torch.no_grad():
        embedding, _, _, _ = model.encode_motion(raw_tensor, mmr_tensor)
    return embedding.cpu().numpy().astype(np.float32)
