#!/usr/bin/env python3
"""A small learnable turn-aware event refiner for EDGE 151D motions.

This is deliberately decoupled from the heavy diffusion backbone. It learns a
residual from base motion + trajectory event features to a pseudo target
produced by event-aligned compositor. Root X/Z is preserved by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn


@dataclass
class TurnEventRefinerConfig:
    motion_dim: int = 151
    event_dim: int = 10
    hidden_dim: int = 256
    layers: int = 4
    dropout: float = 0.05
    preserve_root_xz: bool = True


class TurnAwareEventRefiner(nn.Module):
    def __init__(self, cfg: TurnEventRefinerConfig = TurnEventRefinerConfig()):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.motion_dim + cfg.event_dim
        blocks = [nn.Conv1d(in_dim, cfg.hidden_dim, kernel_size=1), nn.GELU()]
        for _ in range(max(1, cfg.layers)):
            blocks += [
                nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            ]
        blocks.append(nn.Conv1d(cfg.hidden_dim, cfg.motion_dim, kernel_size=1))
        self.net = nn.Sequential(*blocks)
        # zero-init final residual for safe start
        if isinstance(self.net[-1], nn.Conv1d):
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, motion: torch.Tensor, event: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """motion [B,T,151], event [B,T,D]."""
        x = torch.cat([motion, event], dim=-1).transpose(1, 2)
        residual = self.net(x).transpose(1, 2)
        if self.cfg.preserve_root_xz:
            residual = residual.clone()
            residual[..., 4] = 0.0
            residual[..., 6] = 0.0
        return motion + residual, residual


def save_checkpoint(path: str, model: TurnAwareEventRefiner, cfg: TurnEventRefinerConfig, meta: Dict) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__, "meta": meta}, path)


def load_checkpoint(path: str, map_location="cpu") -> TurnAwareEventRefiner:
    obj = torch.load(path, map_location=map_location)
    cfg = TurnEventRefinerConfig(**obj.get("config", {}))
    model = TurnAwareEventRefiner(cfg)
    model.load_state_dict(obj["state_dict"], strict=True)
    return model
