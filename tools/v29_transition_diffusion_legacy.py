#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V29 temporal conditional diffusion for safe Dunhuang transitions.

This file intentionally keeps the historical module name so existing EDGE
scheduler imports remain valid.  The old frame-independent MLP is replaced by
a dilated temporal network with global self-attention, correct skipped-step
DDIM inference, SO(3) projection, endpoint-safe blend envelopes, and optional
local manifold filtering.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tools.v29_motion_geometry import (
    CONTACT,
    MOTION_DIM,
    ROOT_X,
    ROOT_Z,
    ROT,
    make_so3_transition,
    project_motion_rotations_torch,
    temporal_so3_filter_np,
    transition_blend_envelope,
)


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.reshape(-1, 1)
        half = self.dim // 2
        frequency = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(10000.0), half,
                device=x.device, dtype=x.dtype,
            )
        )
        phase = x / frequency.reshape(1, -1)
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class FiLMTemporalBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = torch.nn.GroupNorm(groups, channels)
        self.conv1 = torch.nn.Conv1d(
            channels, channels, kernel_size=3,
            padding=int(dilation), dilation=int(dilation),
        )
        self.norm2 = torch.nn.GroupNorm(groups, channels)
        self.conv2 = torch.nn.Conv1d(
            channels, channels, kernel_size=3, padding=1,
        )
        self.cond = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(cond_dim, channels * 4),
        )
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.cond(condition).chunk(4, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + gamma1[..., None]) + beta1[..., None]
        h = self.dropout(self.conv1(F.silu(h)))
        h = self.norm2(h)
        h = h * (1.0 + gamma2[..., None]) + beta2[..., None]
        h = self.conv2(F.silu(h))
        if mask is not None:
            h = h * mask[:, None, :]
        return x + h


class TemporalTransitionDenoiser(torch.nn.Module):
    """Dilated temporal denoiser with full-sequence attention."""

    def __init__(
        self,
        motion_dim: int = MOTION_DIM,
        music_dim: int = 12,
        hidden_dim: int = 384,
        num_blocks: int = 10,
        num_heads: int = 8,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        time_dim = 96
        self.time_embedding = torch.nn.Sequential(
            SinusoidalEmbedding(time_dim),
            torch.nn.Linear(time_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        cond_input = 4 * self.motion_dim + self.music_dim + 1
        self.condition = torch.nn.Sequential(
            torch.nn.LayerNorm(cond_input),
            torch.nn.Linear(cond_input, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        # position, sin(pi p), cos(pi p), valid mask
        self.input_projection = torch.nn.Conv1d(
            self.motion_dim + 4, hidden_dim, kernel_size=1
        )
        dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8]
        self.blocks = torch.nn.ModuleList(
            [
                FiLMTemporalBlock(
                    hidden_dim,
                    hidden_dim,
                    dilations[i % len(dilations)],
                    dropout,
                )
                for i in range(self.num_blocks)
            ]
        )
        heads = max(1, min(int(num_heads), hidden_dim // 32))
        while hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.attn_norm = torch.nn.LayerNorm(hidden_dim)
        self.attention = torch.nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        groups = 8 if hidden_dim % 8 == 0 else 1
        self.output = torch.nn.Sequential(
            torch.nn.GroupNorm(groups, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden_dim, self.motion_dim, 1),
        )
        torch.nn.init.normal_(self.output[-1].weight, mean=0.0, std=1e-3)
        torch.nn.init.zeros_(self.output[-1].bias)

    def forward(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        music: torch.Tensor,
        length_norm: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_velocity: torch.Tensor | None = None,
        end_velocity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, k, d = noisy.shape
        if d != self.motion_dim:
            raise ValueError(f"Expected motion dim {self.motion_dim}, got {d}")
        if music.shape[-1] > self.music_dim:
            music = music[..., : self.music_dim]
        elif music.shape[-1] < self.music_dim:
            music = F.pad(music, (0, self.music_dim - music.shape[-1]))
        if start_velocity is None:
            start_velocity = torch.zeros_like(start)
        if end_velocity is None:
            end_velocity = torch.zeros_like(end)
        if mask is None:
            mask = torch.ones((b, k), device=noisy.device, dtype=noisy.dtype)
        else:
            mask = mask.to(noisy.dtype)
        if pos.shape[-1] != 1:
            pos = pos[..., :1]

        position_features = torch.cat(
            [
                pos,
                torch.sin(math.pi * pos),
                torch.cos(math.pi * pos),
                mask[..., None],
            ],
            dim=-1,
        )
        x = torch.cat([noisy, position_features], dim=-1).transpose(1, 2)
        x = self.input_projection(x)

        cond_raw = torch.cat(
            [
                start,
                end,
                start_velocity,
                end_velocity,
                music,
                length_norm,
            ],
            dim=-1,
        )
        condition = self.condition(cond_raw) + self.time_embedding(t)

        midpoint = max(1, len(self.blocks) // 2)
        for index, block in enumerate(self.blocks):
            x = block(x, condition, mask)
            if index == midpoint - 1:
                seq = x.transpose(1, 2)
                normed = self.attn_norm(seq)
                attended = self.attention(
                    normed,
                    normed,
                    normed,
                    key_padding_mask=~(mask > 0.5),
                    need_weights=False,
                )[0]
                seq = seq + attended
                x = seq.transpose(1, 2)
        result = self.output(x).transpose(1, 2)
        return result * mask[..., None]


# Backward-compatible symbol used by the existing training script imports.
TransitionDenoiser = TemporalTransitionDenoiser


def _linear_beta_schedule(
    steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1e-4, 0.02, int(steps), device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def _selected_timesteps(
    train_steps: int,
    infer_steps: int,
    start_index: int,
    device: torch.device,
) -> torch.Tensor:
    count = max(2, min(int(infer_steps), int(start_index) + 1))
    indices = torch.linspace(
        int(start_index), 0, count, device=device
    ).round().long()
    return torch.unique_consecutive(indices)


def load_transition_diffusion(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> Dict[str, Any] | None:
    if not path:
        return None
    ckpt_path = Path(str(path))
    if not ckpt_path.is_file():
        raise RuntimeError(f"Transition diffusion checkpoint not found: {ckpt_path}")
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    architecture = str(config.get("architecture", "legacy_frame_mlp"))
    if architecture != "v29_temporal_dilated_attention":
        raise RuntimeError(
            "The selected transition checkpoint uses the old frame-independent "
            f"architecture ({architecture}). Rebuild the V29 dataset and retrain "
            "with train_v27_transition_diffusion.py before enabling V29 diffusion."
        )
    model = TemporalTransitionDenoiser(
        motion_dim=int(config.get("motion_dim", MOTION_DIM)),
        music_dim=int(config.get("music_dim", 12)),
        hidden_dim=int(config.get("hidden_dim", 384)),
        num_blocks=int(config.get("num_blocks", 10)),
        num_heads=int(config.get("num_heads", 8)),
        dropout=float(config.get("dropout", 0.08)),
    ).to(device)
    state = ckpt.get("ema_model", ckpt.get("model"))
    if state is None:
        raise RuntimeError(f"Checkpoint has no model state: {ckpt_path}")
    model.load_state_dict(state)
    model.eval()
    return {
        "model": model,
        "config": config,
        "path": str(ckpt_path),
        "best_val_loss": ckpt.get("best_val_loss"),
        "epoch": ckpt.get("epoch"),
    }


def sample_transition_diffusion(
    bundle: Dict[str, Any] | None,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    blend: float = 0.18,
    steps: int = 32,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample a local transition with correct skipped-step DDIM updates.

    Additional V29 runtime controls are read from environment variables:
      V29_TRANSITION_NOISE_STRENGTH      default 0.55
      V29_TRANSITION_BLEND_POWER         default 2.0
      V29_TRANSITION_FILTER_WINDOW       default 5
      V29_TRANSITION_FILTER_STRENGTH     default 0.20
      V29_TRANSITION_PRESERVE_CONTACTS   default 1
    """
    k = int(length)
    start_np = np.asarray(start_frame, dtype=np.float32).reshape(-1)
    end_np = np.asarray(end_frame, dtype=np.float32).reshape(-1)
    if k <= 0:
        return np.zeros((0, len(start_np)), dtype=np.float32), {
            "enabled": False,
            "reason": "zero_length",
        }

    if rough is None or len(rough) != k:
        rough_np = make_so3_transition(start_np[None], end_np[None], k)
    else:
        rough_np = np.asarray(rough, dtype=np.float32).copy()

    if bundle is None:
        return rough_np, {"enabled": False, "reason": "no_checkpoint"}

    device = torch.device(device)
    model: TemporalTransitionDenoiser = bundle["model"]
    config = bundle.get("config", {})
    train_steps = int(config.get("diffusion_steps", 100))
    motion_dim = int(config.get("motion_dim", len(start_np)))
    music_dim = int(config.get("music_dim", 12))

    noise_strength = float(
        np.clip(float(os.getenv("V29_TRANSITION_NOISE_STRENGTH", "0.55")), 0.05, 1.0)
    )
    blend_power = float(os.getenv("V29_TRANSITION_BLEND_POWER", "2.0"))
    filter_window = int(os.getenv("V29_TRANSITION_FILTER_WINDOW", "5"))
    filter_strength = float(
        np.clip(float(os.getenv("V29_TRANSITION_FILTER_STRENGTH", "0.20")), 0.0, 1.0)
    )
    preserve_contacts = os.getenv("V29_TRANSITION_PRESERVE_CONTACTS", "1").lower() in {
        "1", "true", "yes", "on",
    }

    start = torch.from_numpy(start_np[:motion_dim]).to(device).reshape(1, motion_dim)
    end = torch.from_numpy(end_np[:motion_dim]).to(device).reshape(1, motion_dim)
    music = torch.from_numpy(
        np.asarray(music_query, dtype=np.float32).reshape(-1)[:music_dim]
    ).to(device).reshape(1, -1)
    if music.shape[-1] < music_dim:
        music = F.pad(music, (0, music_dim - music.shape[-1]))

    rough_t = torch.from_numpy(rough_np[:, :motion_dim]).to(device).reshape(
        1, k, motion_dim
    )
    start_velocity = rough_t[:, 0] - start
    end_velocity = end - rough_t[:, -1]
    length_norm = torch.tensor(
        [[min(k / float(max(int(config.get("max_len", 120)), 1)), 1.0)]],
        device=device,
        dtype=torch.float32,
    )
    pos = torch.linspace(
        1.0 / (k + 1), k / (k + 1), k, device=device
    ).reshape(1, k, 1)
    mask = torch.ones((1, k), device=device, dtype=torch.float32)

    _, _, alpha_bar = _linear_beta_schedule(train_steps, device)
    start_index = int(round(noise_strength * (train_steps - 1)))
    indices = _selected_timesteps(
        train_steps, int(steps), start_index=start_index, device=device
    )

    generator = torch.Generator(device=device)
    seed = int(os.getenv("V29_TRANSITION_SEED", "20260610"))
    generator.manual_seed(seed + k)
    noise = torch.randn(
        rough_t.shape,
        device=device,
        dtype=rough_t.dtype,
        generator=generator,
    )
    first_ab = alpha_bar[indices[0]]
    x = torch.sqrt(first_ab) * rough_t + torch.sqrt(1.0 - first_ab) * noise

    with torch.no_grad():
        for index_pos, idx in enumerate(indices):
            t_value = torch.full(
                (1,),
                float(idx.item()) / max(train_steps - 1, 1),
                device=device,
            )
            eps = model(
                x,
                t_value,
                start,
                end,
                music,
                length_norm,
                pos,
                mask=mask,
                start_velocity=start_velocity,
                end_velocity=end_velocity,
            )
            ab_t = alpha_bar[idx]
            x0 = (
                x - torch.sqrt(1.0 - ab_t) * eps
            ) / torch.sqrt(ab_t).clamp_min(1e-6)
            x0 = project_motion_rotations_torch(x0)
            x0[..., CONTACT] = x0[..., CONTACT].clamp(0.0, 1.0)
            x0[..., ROOT_X] = 0.0
            x0[..., ROOT_Z] = 0.0

            if index_pos + 1 < len(indices):
                prev_idx = indices[index_pos + 1]
                ab_prev = alpha_bar[prev_idx]
                # Deterministic DDIM path for the selected, possibly skipped timestep.
                x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1.0 - ab_prev) * eps
            else:
                x = x0

    generated = x[0].cpu().numpy().astype(np.float32)
    if generated.shape[1] < len(start_np):
        generated = np.pad(
            generated,
            ((0, 0), (0, len(start_np) - generated.shape[1])),
            mode="constant",
        )

    envelope = transition_blend_envelope(k, blend_power)[:, None]
    effective_blend = float(np.clip(blend, 0.0, 1.0)) * envelope
    result = (
        effective_blend * generated[:, : len(start_np)]
        + (1.0 - effective_blend) * rough_np
    ).astype(np.float32)

    # Contacts remain scheduler-controlled until a dedicated contact decoder is
    # trained; this avoids hallucinated one-frame foot switches.
    if preserve_contacts:
        result[:, CONTACT] = rough_np[:, CONTACT]
    result[:, ROOT_X] = 0.0
    result[:, ROOT_Z] = 0.0
    result = temporal_so3_filter_np(
        result,
        window=filter_window,
        strength=filter_strength,
        preserve_contacts=preserve_contacts,
    )

    return result.astype(np.float32), {
        "enabled": True,
        "architecture": "v29_temporal_dilated_attention",
        "checkpoint": str(bundle.get("path", "")),
        "steps": int(len(indices)),
        "train_steps": int(train_steps),
        "start_timestep": int(start_index),
        "noise_strength": float(noise_strength),
        "blend": float(blend),
        "blend_power": float(blend_power),
        "filter_window": int(filter_window),
        "filter_strength": float(filter_strength),
        "preserve_contacts": bool(preserve_contacts),
    }
