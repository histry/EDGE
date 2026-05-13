#!/usr/bin/env python3
"""Turn-aware Event Refiner v2.

This module implements the second-stage trainable correction proposed after the
first single-target refiner showed over-smoothing:

A. multi pseudo-target training
B. event-weighted body-part losses
C. small residual correction on top of the no-train turn-aware anchor motion

The model is deliberately lightweight and checkpoint-friendly. It does not touch
EDGE.py/model.py. It learns:
    refined = anchor_no_train_turn_event + bounded_delta(anchor, base, event_features)
while preserving root X/Z.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

from turn_aware_event_utils import (
    CONTACT_SLICE,
    LOWER_ROT_INDEX,
    TORSO_ROT_INDEX,
    UPPER_ROT_INDEX,
    ROOT_X_IDX,
    ROOT_Z_IDX,
)


def _as_index_array(idx) -> np.ndarray:
    if isinstance(idx, slice):
        return np.arange(idx.start or 0, idx.stop, idx.step or 1, dtype=np.int64)
    return np.asarray(idx, dtype=np.int64).reshape(-1)


CONTACT_INDEX = _as_index_array(CONTACT_SLICE)
LOWER_INDEX = _as_index_array(LOWER_ROT_INDEX)
TORSO_INDEX = _as_index_array(TORSO_ROT_INDEX)
UPPER_INDEX = _as_index_array(UPPER_ROT_INDEX)
ROOT_XZ_INDEX = np.asarray([ROOT_X_IDX, ROOT_Z_IDX], dtype=np.int64)


@dataclass
class RefinerConfig:
    motion_dim: int = 151
    event_dim: int = 11
    hidden: int = 256
    depth: int = 3
    dropout: float = 0.05
    max_delta: float = 0.18
    preserve_root_xz: bool = True
    contact_clip: bool = True

    # Loss weights
    base_mse_weight: float = 1.0
    contact_weight: float = 5.0
    lower_weight: float = 3.0
    torso_weight: float = 4.0
    upper_weight: float = 4.0
    root_preserve_weight: float = 20.0
    smooth_delta_weight: float = 0.15
    anchor_outside_event_weight: float = 0.25
    activity_keep_weight: float = 2.0
    activity_keep_ratio: float = 0.70
    activity_max_ratio: float = 1.25


class TurnAwareEventRefiner(nn.Module):
    """Small bounded residual model.

    Input per frame:
      anchor motion, base motion, anchor-base delta, event feature vector
    Output:
      bounded residual delta added to anchor.
    """

    def __init__(self, cfg: RefinerConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.motion_dim * 3 + cfg.event_dim
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(1, cfg.depth)):
            layers += [nn.Linear(d, cfg.hidden), nn.SiLU(), nn.Dropout(cfg.dropout)]
            d = cfg.hidden
        layers.append(nn.Linear(d, cfg.motion_dim))
        self.net = nn.Sequential(*layers)

        # Start as identity/refiner-off. This preserves no-train anchor at init.
        last = self.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        root_mask = torch.ones(cfg.motion_dim, dtype=torch.float32)
        if cfg.preserve_root_xz:
            root_mask[ROOT_X_IDX] = 0.0
            root_mask[ROOT_Z_IDX] = 0.0
        self.register_buffer("delta_mask", root_mask.view(1, 1, cfg.motion_dim))

    def forward(self, base: torch.Tensor, anchor: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        if base.ndim == 2:
            base = base.unsqueeze(0)
        if anchor.ndim == 2:
            anchor = anchor.unsqueeze(0)
        if event.ndim == 2:
            event = event.unsqueeze(0)
        x = torch.cat([anchor, base, anchor - base, event], dim=-1)
        delta = torch.tanh(self.net(x)) * float(self.cfg.max_delta)
        delta = delta * self.delta_mask.to(delta.device)
        out = anchor + delta
        if self.cfg.contact_clip:
            contact_idx = torch.as_tensor(CONTACT_INDEX, dtype=torch.long, device=out.device)
            out_contacts = out.index_select(-1, contact_idx).clamp(0.0, 1.0)
            out = out.clone()
            out.index_copy_(-1, contact_idx, out_contacts)
        return out


def frame_energy_torch(x: torch.Tensor, indices: Iterable[int]) -> torch.Tensor:
    idx = torch.as_tensor(list(indices), dtype=torch.long, device=x.device)
    if x.shape[1] <= 1:
        return torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    y = x.index_select(-1, idx)
    e = torch.zeros((x.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
    e[:, 1:] = torch.sqrt(torch.mean((y[:, 1:] - y[:, :-1]) ** 2, dim=-1) + 1e-8)
    return e


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, idx: np.ndarray, w: torch.Tensor) -> torch.Tensor:
    ii = torch.as_tensor(idx, dtype=torch.long, device=pred.device)
    diff = pred.index_select(-1, ii) - target.index_select(-1, ii)
    # diff: [B,T,D], w: [B,T]
    return torch.mean((diff ** 2) * w.unsqueeze(-1))


def _ensure_btd(x: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize motion/event tensors to [B,T,D].

    Training usually feeds base/anchor/target/event as [T,D], while model.forward
    returns pred as [1,T,D].  The previous v2 loss only unsqueezed tensors when
    pred was 2-D, which left target/event as [T,D] and caused frame-energy shapes
    like [150,151] to be compared with [1,150].
    """
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} must be [T,D] or [B,T,D], got {tuple(x.shape)}")


def _match_batch(x: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if x.shape[0] == batch:
        return x
    if x.shape[0] == 1:
        return x.expand(batch, -1, -1)
    raise ValueError(f"{name} batch={x.shape[0]} cannot match batch={batch}")


def refiner_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    anchor: torch.Tensor,
    event: torch.Tensor,
    cfg: RefinerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Event-weighted body-part loss.

    event feature columns are defined in turn_aware_event_utils.event_feature_matrix:
      6 turn_gate, 7 support_gate, 8 expressive_gate, 10 speed_gate.
    """
    pred = _ensure_btd(pred, "pred")
    target = _ensure_btd(target, "target")
    base = _ensure_btd(base, "base")
    anchor = _ensure_btd(anchor, "anchor")
    event = _ensure_btd(event, "event")

    B = pred.shape[0]
    target = _match_batch(target, B, "target")
    base = _match_batch(base, B, "base")
    anchor = _match_batch(anchor, B, "anchor")
    event = _match_batch(event, B, "event")

    # Align sequence length defensively. This keeps loss robust if a motion or
    # event feature file has a slightly different T.
    T = min(pred.shape[1], target.shape[1], base.shape[1], anchor.shape[1], event.shape[1])
    pred = pred[:, :T]
    target = target[:, :T]
    base = base[:, :T]
    anchor = anchor[:, :T]
    event = event[:, :T]

    turn_gate = event[..., 6].clamp(0, 1)
    support_gate = event[..., 7].clamp(0, 1)
    expressive_gate = event[..., 8].clamp(0, 1)
    speed_gate = event[..., 10].clamp(0, 1)
    event_gate = torch.maximum(torch.maximum(turn_gate, support_gate), expressive_gate)

    contact_w = 1.0 + cfg.contact_weight * support_gate
    lower_w = 1.0 + cfg.lower_weight * (support_gate + 0.5 * speed_gate)
    torso_w = 1.0 + cfg.torso_weight * (expressive_gate + 0.5 * turn_gate)
    upper_w = 1.0 + cfg.upper_weight * (expressive_gate + 0.5 * turn_gate)

    loss_contact = weighted_mse(pred, target, CONTACT_INDEX, contact_w)
    loss_lower = weighted_mse(pred, target, LOWER_INDEX, lower_w)
    loss_torso = weighted_mse(pred, target, TORSO_INDEX, torso_w)
    loss_upper = weighted_mse(pred, target, UPPER_INDEX, upper_w)

    # Preserve root X/Z from anchor/no-train result.
    loss_root = weighted_mse(pred, anchor, ROOT_XZ_INDEX, torch.ones_like(event_gate))

    # Keep outside-event frames close to anchor so we learn timing correction, not full reconstruction.
    outside = (1.0 - event_gate).clamp(0, 1)
    loss_anchor_outside = torch.mean(((pred - anchor) ** 2) * outside.unsqueeze(-1))

    # Smooth only the learned delta, not the motion itself.
    delta = pred - anchor
    if delta.shape[1] > 1:
        loss_smooth = torch.mean((delta[:, 1:] - delta[:, :-1]) ** 2)
    else:
        loss_smooth = torch.zeros((), dtype=pred.dtype, device=pred.device)

    # Activity keep margin: do not let refiner erase event-local target energy.
    lower_p = frame_energy_torch(pred, LOWER_INDEX)
    lower_t = frame_energy_torch(target, LOWER_INDEX).detach()
    torso_p = frame_energy_torch(pred, TORSO_INDEX)
    torso_t = frame_energy_torch(target, TORSO_INDEX).detach()
    upper_p = frame_energy_torch(pred, UPPER_INDEX)
    upper_t = frame_energy_torch(target, UPPER_INDEX).detach()

    lower_event = torch.maximum(support_gate, speed_gate)
    expr_event = torch.maximum(expressive_gate, turn_gate)

    def margin_keep(p: torch.Tensor, t: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        low = torch.relu(cfg.activity_keep_ratio * t - p) ** 2
        high = torch.relu(p - cfg.activity_max_ratio * (t + 1e-6)) ** 2
        return torch.mean((low + 0.25 * high) * gate)

    loss_activity = (
        margin_keep(lower_p, lower_t, lower_event)
        + margin_keep(torso_p, torso_t, expr_event)
        + margin_keep(upper_p, upper_t, expr_event)
    )

    loss = (
        cfg.base_mse_weight * (loss_contact + loss_lower + loss_torso + loss_upper)
        + cfg.root_preserve_weight * loss_root
        + cfg.anchor_outside_event_weight * loss_anchor_outside
        + cfg.smooth_delta_weight * loss_smooth
        + cfg.activity_keep_weight * loss_activity
    )

    parts = {
        "loss": float(loss.detach().cpu()),
        "contact": float(loss_contact.detach().cpu()),
        "lower": float(loss_lower.detach().cpu()),
        "torso": float(loss_torso.detach().cpu()),
        "upper": float(loss_upper.detach().cpu()),
        "root": float(loss_root.detach().cpu()),
        "anchor_outside": float(loss_anchor_outside.detach().cpu()),
        "smooth": float(loss_smooth.detach().cpu()),
        "activity": float(loss_activity.detach().cpu()),
    }
    return loss, parts


def make_checkpoint(model: TurnAwareEventRefiner, cfg: RefinerConfig, feature_names: List[str], extra: Dict | None = None) -> Dict:
    return {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "feature_names": feature_names,
        "extra": extra or {},
    }


def load_refiner(path: str, map_location: str | torch.device = "cpu") -> Tuple[TurnAwareEventRefiner, RefinerConfig, List[str], Dict]:
    ckpt = torch.load(path, map_location=map_location)
    cfg = RefinerConfig(**ckpt["config"])
    model = TurnAwareEventRefiner(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, cfg, list(ckpt.get("feature_names", [])), dict(ckpt.get("extra", {}))
