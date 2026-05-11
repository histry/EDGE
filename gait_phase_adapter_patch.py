"""Runtime model/diffusion patch for gait/footstep phase conditioning.

Enable training/inference with:
  EDGE_GAIT_PHASE_COND=1

The patch is checkpoint-friendly:
- It wraps DanceDecoder.trajectory_projection instead of changing the base
  constructor signature.
- Existing checkpoints can load with the repository's checkpoint adaptation
  because new gait_phase_projection parameters are newly initialized.
- With train_stage=adapter, trajectory_projection remains trainable in the
  existing EDGE stage-freezing logic, so the gait branch trains automatically.

Expected condition:
  cond['gait_phase'] = [B,T,6]
    0 phase_sin
    1 phase_cos
    2 left_contact_prior
    3 right_contact_prior
    4 move_gate
    5 speed_norm
"""
from __future__ import annotations

import os
import weakref
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


class GaitAwareTrajectoryProjection(nn.Module):
    def __init__(self, base: nn.Module, owner, phase_dim: int, latent_dim: int):
        super().__init__()
        self.base = base
        self.owner_ref = weakref.ref(owner)
        self.phase_dim = int(phase_dim)
        self.gait_phase_projection = nn.Sequential(
            nn.Linear(self.phase_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.gait_phase_gate = nn.Parameter(torch.tensor(float(_env_float("EDGE_GAIT_PHASE_INIT_GATE", 0.0))))
        self.gait_phase_dropout = nn.Dropout(float(_env_float("EDGE_GAIT_PHASE_DROP_PROB", 0.10)))

    def forward(self, traj_features):
        out = self.base(traj_features)
        if not _env_bool("EDGE_GAIT_PHASE_COND", False):
            return out
        owner = self.owner_ref()
        phase = getattr(owner, "_edge_last_gait_phase_cond", None) if owner is not None else None
        if phase is None:
            return out
        phase = phase.to(device=out.device, dtype=out.dtype)
        if phase.ndim == 2:
            phase = phase[None]
        if phase.shape[1] != out.shape[1]:
            phase = F.interpolate(phase.transpose(1, 2), size=out.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        if phase.shape[-1] < self.phase_dim:
            pad = torch.zeros(*phase.shape[:-1], self.phase_dim - phase.shape[-1], device=phase.device, dtype=phase.dtype)
            phase = torch.cat([phase, pad], dim=-1)
        phase = phase[..., : self.phase_dim]
        phase_tokens = self.gait_phase_projection(phase)
        gate = torch.tanh(self.gait_phase_gate).to(device=out.device, dtype=out.dtype)
        return out + gate * self.gait_phase_dropout(phase_tokens)


def _resize_phase(phase, batch_size, seq_len, device, dtype, phase_dim):
    if phase is None:
        return None
    phase = phase.to(device=device, dtype=dtype)
    if phase.ndim == 2:
        phase = phase[None].expand(batch_size, -1, -1)
    if phase.ndim != 3:
        raise ValueError(f"gait_phase must be [B,T,D] or [T,D], got {tuple(phase.shape)}")
    if phase.shape[0] == 1 and batch_size > 1:
        phase = phase.expand(batch_size, -1, -1)
    if phase.shape[0] != batch_size:
        raise ValueError(f"gait_phase batch mismatch: expected {batch_size}, got {phase.shape[0]}")
    if phase.shape[1] != seq_len:
        phase = F.interpolate(phase.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
    if phase.shape[-1] < phase_dim:
        pad = torch.zeros(*phase.shape[:-1], phase_dim - phase.shape[-1], device=device, dtype=dtype)
        phase = torch.cat([phase, pad], dim=-1)
    return phase[..., :phase_dim]


def install_gait_phase_adapter_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
        from model.diffusion import GaussianDiffusion, maybe_unnormalize
    except Exception as exc:
        if verbose:
            print(f"⚠️ Gait phase adapter patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_gait_phase_adapter_patch_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.edge_gait_phase_dim = _env_int("EDGE_GAIT_PHASE_DIM", 6)
        latent_dim = int(getattr(self.null_trajectory_embed, "shape", [1, 1, 512])[-1])
        if not isinstance(getattr(self, "trajectory_projection", None), GaitAwareTrajectoryProjection):
            self.trajectory_projection = GaitAwareTrajectoryProjection(
                self.trajectory_projection,
                owner=self,
                phase_dim=self.edge_gait_phase_dim,
                latent_dim=latent_dim,
            )
        self.edge_gait_phase_enabled = True
        if verbose:
            print(
                "🦶 Gait phase adapter branch initialized: "
                f"enabled={_env_bool('EDGE_GAIT_PHASE_COND', False)}, dim={self.edge_gait_phase_dim}"
            )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        phase = cond_embed.get("gait_phase", None) if isinstance(cond_embed, dict) else None
        if _env_bool("EDGE_GAIT_PHASE_COND", False):
            phase_dim = int(getattr(self, "edge_gait_phase_dim", _env_int("EDGE_GAIT_PHASE_DIM", 6)))
            self._edge_last_gait_phase_cond = _resize_phase(phase, batch_size, seq_len, device, dtype, phase_dim)
        else:
            self._edge_last_gait_phase_cond = None
        return original_prepare(self, cond_embed, batch_size, seq_len, device, dtype)

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder._edge_gait_phase_adapter_patch_installed = True

    # Store latest condition on diffusion so _contact_loss can see gait priors.
    original_diff_forward = GaussianDiffusion.forward

    @wraps(original_diff_forward)
    def patched_diff_forward(self, *args, **kwargs):
        cond = None
        if len(args) >= 2:
            cond = args[1]
        elif "cond" in kwargs:
            cond = kwargs["cond"]
        elif "cond_embed" in kwargs:
            cond = kwargs["cond_embed"]
        self._edge_last_training_cond = cond
        return original_diff_forward(self, *args, **kwargs)

    GaussianDiffusion.forward = patched_diff_forward

    original_contact_loss = GaussianDiffusion._contact_loss

    @wraps(original_contact_loss)
    def patched_contact_loss(self, model_motion_x0, target_motion_x0):
        base = original_contact_loss(self, model_motion_x0, target_motion_x0)
        if not _env_bool("EDGE_GAIT_CONTACT_LOSS", _env_bool("EDGE_GAIT_PHASE_COND", False)):
            return base
        cond = getattr(self, "_edge_last_training_cond", None)
        if not isinstance(cond, dict) or cond.get("gait_phase", None) is None:
            return base
        weight = _env_float("EDGE_GAIT_CONTACT_LOSS_WEIGHT", 0.60)
        if weight <= 0:
            return base
        try:
            phase = cond["gait_phase"].to(device=model_motion_x0.device, dtype=model_motion_x0.dtype)
            if phase.shape[1] != model_motion_x0.shape[1]:
                phase = F.interpolate(phase.transpose(1, 2), size=model_motion_x0.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            left_prior = phase[..., 2].clamp(0.0, 1.0)
            right_prior = phase[..., 3].clamp(0.0, 1.0)
            move_gate = phase[..., 4].clamp(0.0, 1.0) if phase.shape[-1] > 4 else torch.ones_like(left_prior)

            physical = maybe_unnormalize(self.normalizer, model_motion_x0) if getattr(self, "normalizer", None) is not None else model_motion_x0
            pred_contacts = physical[..., 0:4].clamp(0.0, 1.0)
            pred_left = pred_contacts[..., 0:2].mean(dim=-1)
            pred_right = pred_contacts[..., 2:4].mean(dim=-1)
            denom = move_gate.sum().clamp_min(1.0)
            gait_loss = (((pred_left - left_prior) ** 2 + (pred_right - right_prior) ** 2) * move_gate).sum() / denom
            return base + float(weight) * gait_loss
        except Exception as exc:
            if _env_bool("EDGE_GAIT_PHASE_STRICT", False):
                raise
            if _env_bool("EDGE_GAIT_PHASE_VERBOSE", False):
                print(f"⚠️ gait contact loss skipped: {exc}")
            return base

    GaussianDiffusion._contact_loss = patched_contact_loss

    if verbose:
        print(
            "✅ Installed gait phase adapter patch: "
            f"cond={_env_bool('EDGE_GAIT_PHASE_COND', False)}, "
            f"contact_loss={_env_bool('EDGE_GAIT_CONTACT_LOSS', _env_bool('EDGE_GAIT_PHASE_COND', False))}"
        )
    return True


def install():
    return install_gait_phase_adapter_patch(verbose=True)
