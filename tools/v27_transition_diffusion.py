#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight conditional diffusion in-betweening for V27/V28 transitions.

The scheduler still retrieves real Dunhuang events as the semantic backbone.
This model is intentionally local: it only redraws transition-budget frames
between two selected events, so it addresses "advanced concatenation" without
allowing a generator to overwrite the cultural motion vocabulary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    from tools.v21_common import ROOT_X, ROOT_Z
except Exception:
    ROOT_X = 4
    ROOT_Z = 6


class TransitionDenoiser(torch.nn.Module):
    def __init__(self, motion_dim: int = 151, music_dim: int = 12, hidden_dim: int = 384) -> None:
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.music_dim = int(music_dim)
        cond_dim = 2 * self.motion_dim + self.music_dim + 3
        self.time = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.motion_dim + hidden_dim * 2 + 1, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, self.motion_dim),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
        music: torch.Tensor,
        length_norm: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        b, k, _ = noisy.shape
        if music.shape[-1] != self.music_dim:
            if music.shape[-1] > self.music_dim:
                music = music[..., : self.music_dim]
            else:
                music = torch.nn.functional.pad(music, (0, self.music_dim - music.shape[-1]))
        cond = torch.cat([start, end, music, length_norm, t.reshape(b, 1), torch.ones_like(length_norm)], dim=-1)
        cond_emb = self.cond(cond).reshape(b, 1, -1).expand(-1, k, -1)
        t_emb = self.time(t.reshape(b, 1)).reshape(b, 1, -1).expand(-1, k, -1)
        x = torch.cat([noisy, cond_emb, t_emb, pos], dim=-1)
        return self.net(x)


def _linear_beta_schedule(steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta = torch.linspace(1e-4, 0.02, int(steps), device=device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def load_transition_diffusion(path: str | Path, device: torch.device | str = "cpu") -> Dict[str, Any] | None:
    if not path:
        return None
    ckpt_path = Path(str(path))
    if not ckpt_path.is_file():
        raise RuntimeError(f"Transition diffusion checkpoint not found: {ckpt_path}")
    device = torch.device(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = TransitionDenoiser(
        motion_dim=int(config.get("motion_dim", 151)),
        music_dim=int(config.get("music_dim", 12)),
        hidden_dim=int(config.get("hidden_dim", 384)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return {"model": model, "config": config, "path": str(ckpt_path)}


def sample_transition_diffusion(
    bundle: Dict[str, Any] | None,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    length: int,
    music_query: np.ndarray,
    rough: np.ndarray | None = None,
    device: torch.device | str = "cpu",
    blend: float = 0.45,
    steps: int = 12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    k = int(length)
    if k <= 0:
        return np.zeros((0, len(start_frame)), dtype=np.float32), {"enabled": False, "reason": "zero_length"}
    if bundle is None:
        if rough is not None:
            return np.asarray(rough, dtype=np.float32), {"enabled": False, "reason": "no_checkpoint"}
        return np.linspace(start_frame, end_frame, k + 2, dtype=np.float32)[1:-1], {"enabled": False, "reason": "no_checkpoint"}

    device = torch.device(device)
    model: TransitionDenoiser = bundle["model"]
    config = bundle.get("config", {})
    train_steps = int(config.get("diffusion_steps", 64))
    infer_steps = max(2, min(int(steps), train_steps))
    beta, alpha, alpha_bar = _linear_beta_schedule(train_steps, device)

    motion_dim = int(config.get("motion_dim", len(start_frame)))
    music_dim = int(config.get("music_dim", 12))
    start = torch.from_numpy(np.asarray(start_frame, dtype=np.float32)[:motion_dim]).to(device).reshape(1, motion_dim)
    end = torch.from_numpy(np.asarray(end_frame, dtype=np.float32)[:motion_dim]).to(device).reshape(1, motion_dim)
    music = torch.from_numpy(np.asarray(music_query, dtype=np.float32).reshape(-1)[:music_dim]).to(device).reshape(1, -1)
    if music.shape[-1] < music_dim:
        music = torch.nn.functional.pad(music, (0, music_dim - music.shape[-1]))
    length_norm = torch.tensor([[min(k / 120.0, 1.0)]], dtype=torch.float32, device=device)
    pos = torch.linspace(1.0 / (k + 1), k / (k + 1), k, device=device).reshape(1, k, 1)

    if rough is not None and len(rough) == k:
        x = torch.from_numpy(np.asarray(rough, dtype=np.float32)[:, :motion_dim]).to(device).reshape(1, k, motion_dim)
        x = x + 0.15 * torch.randn_like(x)
    else:
        x = torch.randn((1, k, motion_dim), device=device)
    indices = torch.linspace(train_steps - 1, 0, infer_steps, device=device).long()
    with torch.no_grad():
        for idx in indices:
            t_value = torch.full((1,), float(idx.item()) / max(train_steps - 1, 1), device=device)
            eps = model(x, t_value, start, end, music, length_norm, pos)
            a = alpha[idx]
            ab = alpha_bar[idx]
            x0 = (x - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab).clamp_min(1e-6)
            if idx.item() > 0:
                prev_ab = alpha_bar[idx - 1]
                x = torch.sqrt(prev_ab) * x0 + torch.sqrt(1.0 - prev_ab) * eps
            else:
                x = x0
    generated = x[0].detach().cpu().numpy().astype(np.float32)

    # Anchor generated transition to endpoints and blend with the physically
    # safe rough path.  The model removes interpolation artifacts, not the
    # scheduler's safety constraints.
    endpoint_line = np.linspace(start_frame[:motion_dim], end_frame[:motion_dim], k + 2, dtype=np.float32)[1:-1]
    generated = 0.88 * generated + 0.12 * endpoint_line
    if rough is not None and len(rough) == k:
        blend = float(np.clip(blend, 0.0, 1.0))
        generated = blend * generated + (1.0 - blend) * np.asarray(rough, dtype=np.float32)[:, :motion_dim]
    if generated.shape[1] < len(start_frame):
        pad = np.zeros((k, len(start_frame) - generated.shape[1]), dtype=np.float32)
        generated = np.concatenate([generated, pad], axis=1)
    generated[:, ROOT_X] = 0.0
    generated[:, ROOT_Z] = 0.0
    return generated[:, : len(start_frame)].astype(np.float32), {
        "enabled": True,
        "checkpoint": str(bundle.get("path", "")),
        "steps": int(infer_steps),
        "blend": float(blend),
    }
