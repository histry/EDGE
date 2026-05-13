"""Model-internal turn-aware event conditioning patch for EDGE.

This runtime patch injects trajectory-derived turn/support/expression event
features into the DanceDecoder.  It is designed to be checkpoint-friendly and
fully environment-gated.

Main modes
----------
1) Native trainable trajectory-token adapter
   EDGE_TURN_EVENT_MODEL_ADAPTER=1
   EDGE_TURN_EVENT_TRAJ_TOKEN=1
   Adds a zero-init event projection residual into trajectory tokens.  When
   used during normal train.py training, only the new adapter can be trained
   with EDGE_TURN_EVENT_FREEZE_BACKBONE=1.

2) Optional output adapter ckpt for pseudo-target distillation
   EDGE_TURN_EVENT_OUTPUT_ADAPTER=1
   EDGE_TURN_EVENT_ADAPTER_CKPT=runs/turn_event_internal_adapter/...
   Loads a small event-conditioned output residual module.  Root X/Z are
   preserved by default.

The patch computes event features internally from cond["trajectory"], so it
requires no change to dataset files or generate_controlled.py.
"""
from __future__ import annotations

import os
import weakref
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from turn_aware_event_utils import EVENT_DIM, env_bool, env_float, env_int, event_feature_matrix_torch

CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_START = 7
ROT_DIM = 6
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def _rot_indices(joints):
    out = []
    for j in joints:
        start = ROT_START + ROT_DIM * int(j)
        out.extend(range(start, start + ROT_DIM))
    return out


def _ensure_btd(x, batch_size, seq_len, device, dtype, name: str):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"{name} must be [T,D] or [B,T,D], got {tuple(x.shape)}")
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1)
    if x.shape[0] != batch_size:
        raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {x.shape[0]}")
    if x.shape[1] != seq_len:
        x = F.interpolate(x.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
    return x


def make_motion_feature_gate(nfeats: int, device, dtype, prefix: str = "EDGE_TURN_EVENT") -> torch.Tensor:
    gate = torch.ones((nfeats,), device=device, dtype=dtype) * float(env_float(f"{prefix}_GATE_DEFAULT", 0.25))
    if nfeats >= 4:
        gate[CONTACT_SLICE] = float(env_float(f"{prefix}_GATE_CONTACTS", 0.0))
    if nfeats > ROOT_X_IDX:
        gate[ROOT_X_IDX] = float(env_float(f"{prefix}_GATE_ROOT_XZ", 0.0))
    if nfeats > ROOT_Y_IDX:
        gate[ROOT_Y_IDX] = float(env_float(f"{prefix}_GATE_ROOT_Y", 0.0))
    if nfeats > ROOT_Z_IDX:
        gate[ROOT_Z_IDX] = float(env_float(f"{prefix}_GATE_ROOT_XZ", 0.0))
    pelvis_idx = [i for i in _rot_indices([0]) if i < nfeats]
    if pelvis_idx:
        gate[pelvis_idx] = float(env_float(f"{prefix}_GATE_PELVIS_ROT", 0.10))
    for idxs, value in [
        (_rot_indices(LOWER_JOINTS), float(env_float(f"{prefix}_GATE_LOWER", 0.45))),
        (_rot_indices(TORSO_JOINTS), float(env_float(f"{prefix}_GATE_TORSO", 0.75))),
        (_rot_indices(UPPER_JOINTS), float(env_float(f"{prefix}_GATE_UPPER", 0.75))),
    ]:
        idxs = [i for i in idxs if i < nfeats]
        if idxs:
            gate[idxs] = value
    return gate.clamp(0.0, 1.0)


class TurnEventTrajectoryProjection(nn.Module):
    """Residual wrapper that injects event features into trajectory tokens."""

    def __init__(self, base: nn.Module, owner, latent_dim: int, event_dim: int = EVENT_DIM):
        super().__init__()
        self.base = base
        self.owner_ref = weakref.ref(owner)
        self.event_dim = int(event_dim)
        self.event_projection = nn.Sequential(
            nn.Linear(self.event_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # Zero init gate means old checkpoints start unchanged.
        self.event_gate = nn.Parameter(torch.tensor(float(env_float("EDGE_TURN_EVENT_INIT_GATE", 0.0))))

    def _pad_or_trim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        out = self.base(traj_features)
        owner = self.owner_ref()
        if owner is None or not env_bool("EDGE_TURN_EVENT_TRAJ_TOKEN", True):
            return out
        event = getattr(owner, "_edge_last_turn_event_features", None)
        if event is None:
            return out
        event = event.to(device=out.device, dtype=out.dtype)
        if event.shape[1] != out.shape[1]:
            event = F.interpolate(event.transpose(1, 2), size=out.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        event = self._pad_or_trim(event, self.event_dim)
        scale = float(env_float("EDGE_TURN_EVENT_TRAJ_SCALE", 1.0))
        return out + scale * torch.tanh(self.event_gate).to(out.dtype) * self.event_projection(event)


class TurnEventOutputAdapter(nn.Module):
    """Small output-space residual module used for optional adapter distillation.

    It is the same module trained by tools/train_turn_event_internal_adapter.py.
    When loaded into DanceDecoder it operates inside the decoder forward pass.
    """

    def __init__(self, nfeats: int = 151, event_dim: int = EVENT_DIM, hidden: int = 256, max_delta: float = 0.12):
        super().__init__()
        self.nfeats = int(nfeats)
        self.event_dim = int(event_dim)
        self.hidden = int(hidden)
        self.max_delta = float(max_delta)
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(self.nfeats),
            nn.Linear(self.nfeats, hidden),
            nn.SiLU(),
        )
        self.event_proj = nn.Sequential(
            nn.LayerNorm(self.event_dim),
            nn.Linear(self.event_dim, hidden),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.nfeats),
        )
        # Starts as identity if used without checkpoint.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _pad_or_trim_event(self, event: torch.Tensor) -> torch.Tensor:
        if event.shape[-1] == self.event_dim:
            return event
        if event.shape[-1] > self.event_dim:
            return event[..., : self.event_dim]
        pad = torch.zeros(*event.shape[:-1], self.event_dim - event.shape[-1], device=event.device, dtype=event.dtype)
        return torch.cat([event, pad], dim=-1)

    def forward(self, motion: torch.Tensor, event: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        if event.ndim == 2:
            event = event.unsqueeze(0)
        if event.shape[0] == 1 and motion.shape[0] > 1:
            event = event.expand(motion.shape[0], -1, -1)
        if event.shape[1] != motion.shape[1]:
            event = F.interpolate(event.transpose(1, 2), size=motion.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        event = self._pad_or_trim_event(event).to(device=motion.device, dtype=motion.dtype)
        h = torch.cat([self.motion_proj(motion), self.event_proj(event)], dim=-1)
        delta = torch.tanh(self.net(h)) * self.max_delta
        if gate is not None:
            delta = delta * gate.view(1, 1, -1).to(device=motion.device, dtype=motion.dtype)
        return motion + delta


def _load_output_adapter_if_needed(owner, verbose: bool = False) -> None:
    if getattr(owner, "_edge_turn_event_output_adapter_loaded", False):
        return
    ckpt = os.environ.get("EDGE_TURN_EVENT_ADAPTER_CKPT", "").strip()
    if not ckpt:
        owner._edge_turn_event_output_adapter_loaded = True
        return
    path = Path(ckpt)
    if not path.exists():
        print(f"⚠️ EDGE_TURN_EVENT_ADAPTER_CKPT not found: {path}")
        owner._edge_turn_event_output_adapter_loaded = True
        return
    payload = torch.load(str(path), map_location="cpu")
    state = payload.get("state_dict", payload)
    max_delta = float(payload.get("max_delta", env_float("EDGE_TURN_EVENT_OUTPUT_MAX_DELTA", 0.12))) if isinstance(payload, dict) else env_float("EDGE_TURN_EVENT_OUTPUT_MAX_DELTA", 0.12)
    hidden = int(payload.get("hidden", env_int("EDGE_TURN_EVENT_OUTPUT_HIDDEN", 256))) if isinstance(payload, dict) else env_int("EDGE_TURN_EVENT_OUTPUT_HIDDEN", 256)
    adapter = TurnEventOutputAdapter(
        nfeats=int(getattr(owner, "nfeats", 151)),
        event_dim=env_int("EDGE_TURN_EVENT_DIM", EVENT_DIM),
        hidden=hidden,
        max_delta=max_delta,
    )
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    adapter.to(device=next(owner.parameters()).device)
    owner.turn_event_output_adapter = adapter
    owner._edge_turn_event_output_adapter_loaded = True
    if verbose or env_bool("EDGE_TURN_EVENT_VERBOSE", False):
        print(f"✅ Loaded turn-event output adapter: {path} missing={len(missing)} unexpected={len(unexpected)}")


def _freeze_backbone_except_turn_event(module: nn.Module) -> None:
    if not env_bool("EDGE_TURN_EVENT_FREEZE_BACKBONE", False):
        return
    for name, param in module.named_parameters():
        trainable = ("turn_event" in name) or ("event_projection" in name and "trajectory_projection" in name)
        param.requires_grad = bool(trainable)


def install_turn_event_model_adapter_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print(f"⚠️ turn-event model adapter patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_turn_event_model_adapter_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs
    original_forward = DanceDecoder.forward

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Keep kwargs compatibility. Users can pass enable_turn_event_model_adapter,
        # but environment is the primary switch.
        enable_kw = bool(kwargs.pop("enable_turn_event_model_adapter", False))
        original_init(self, *args, **kwargs)
        enable = enable_kw or env_bool("EDGE_TURN_EVENT_MODEL_ADAPTER", False)
        self.enable_turn_event_model_adapter = bool(enable)
        self._edge_last_turn_event_features = None
        if enable:
            latent_dim = int(getattr(self.null_trajectory_embed, "shape", [1, 1, 256])[-1])
            event_dim = env_int("EDGE_TURN_EVENT_DIM", EVENT_DIM)
            if not isinstance(getattr(self, "trajectory_projection", None), TurnEventTrajectoryProjection):
                self.trajectory_projection = TurnEventTrajectoryProjection(
                    self.trajectory_projection,
                    owner=self,
                    latent_dim=latent_dim,
                    event_dim=event_dim,
                )
            if env_bool("EDGE_TURN_EVENT_OUTPUT_ADAPTER", False):
                self.turn_event_output_adapter = TurnEventOutputAdapter(
                    nfeats=int(getattr(self, "nfeats", 151)),
                    event_dim=event_dim,
                    hidden=env_int("EDGE_TURN_EVENT_OUTPUT_HIDDEN", 256),
                    max_delta=env_float("EDGE_TURN_EVENT_OUTPUT_MAX_DELTA", 0.12),
                )
            _freeze_backbone_except_turn_event(self)
            if verbose:
                trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
                print(
                    "✅ Turn-aware event model adapter initialized: "
                    f"traj_token={env_bool('EDGE_TURN_EVENT_TRAJ_TOKEN', True)}, "
                    f"output_adapter={env_bool('EDGE_TURN_EVENT_OUTPUT_ADAPTER', False)}, "
                    f"freeze_backbone={env_bool('EDGE_TURN_EVENT_FREEZE_BACKBONE', False)}, "
                    f"trainable_params={trainable}"
                )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        self._edge_last_turn_event_features = None
        if getattr(self, "enable_turn_event_model_adapter", False):
            traj = cond_embed.get("trajectory", None) if isinstance(cond_embed, dict) else None
            event = cond_embed.get("turn_event", None) if isinstance(cond_embed, dict) else None
            if event is not None:
                event = _ensure_btd(event, batch_size, seq_len, device, dtype, "turn_event")
                self._edge_last_turn_event_features = event[..., : env_int("EDGE_TURN_EVENT_DIM", EVENT_DIM)]
            elif traj is not None:
                traj_btd = _ensure_btd(traj, batch_size, seq_len, device, dtype, "trajectory")[..., :2]
                count = env_int("EDGE_TURN_EVENT_COUNT", 5)
                self._edge_last_turn_event_features = event_feature_matrix_torch(
                    traj_btd,
                    count=count,
                    support_lag=env_int("EDGE_TURN_SUPPORT_LAG", 8),
                    expressive_lag=env_int("EDGE_TURN_EXPR_LAG", 4),
                    min_gap=env_int("EDGE_TURN_MIN_GAP", 18),
                    gate_sigma=env_float("EDGE_TURN_GATE_SIGMA", 5.0),
                ).to(device=device, dtype=dtype)
        return original_prepare(self, cond_embed, batch_size, seq_len, device, dtype)

    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        out = original_forward(self, *args, **kwargs)
        if not getattr(self, "enable_turn_event_model_adapter", False):
            return out
        if not env_bool("EDGE_TURN_EVENT_OUTPUT_ADAPTER", False):
            return out
        event = getattr(self, "_edge_last_turn_event_features", None)
        if event is None:
            return out
        _load_output_adapter_if_needed(self, verbose=not getattr(self, "_edge_turn_event_load_logged", False))
        self._edge_turn_event_load_logged = True
        adapter = getattr(self, "turn_event_output_adapter", None)
        if adapter is None:
            return out
        gate = make_motion_feature_gate(out.shape[-1], out.device, out.dtype, prefix="EDGE_TURN_EVENT")
        refined = adapter(out, event.to(device=out.device, dtype=out.dtype), gate=gate)
        if env_bool("EDGE_TURN_EVENT_PRESERVE_ROOT_XZ", True) and refined.shape[-1] > ROOT_Z_IDX:
            refined = refined.clone()
            refined[..., ROOT_X_IDX] = out[..., ROOT_X_IDX]
            refined[..., ROOT_Z_IDX] = out[..., ROOT_Z_IDX]
        return refined

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder.forward = patched_forward
    DanceDecoder._edge_turn_event_model_adapter_installed = True

    if verbose:
        print(
            "✅ Installed turn-aware event model adapter patch: "
            f"enabled={env_bool('EDGE_TURN_EVENT_MODEL_ADAPTER', False)}, "
            f"traj_token={env_bool('EDGE_TURN_EVENT_TRAJ_TOKEN', True)}, "
            f"output_adapter={env_bool('EDGE_TURN_EVENT_OUTPUT_ADAPTER', False)}"
        )
    return True


def install():
    return install_turn_event_model_adapter_patch(verbose=True)
