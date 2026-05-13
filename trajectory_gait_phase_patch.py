"""Runtime patch: gait-phase residual branch for trajectory projection.

Purpose
-------
This patch fixes inference/training structure alignment for checkpoints that
were trained with gait/phase-aware trajectory projection keys such as:

    trajectory_projection.gait_phase_gate
    trajectory_projection.base.*

It must be installed after trajectory_enhancement_patch, because it wraps the
already-enhanced trajectory_projection as:

    GaitPhaseTrajectoryProjection(
        base=EdgeAdvancedTrajectoryProjection(...)
    )

The wrapper is checkpoint-friendly:
- If EDGE_GAIT_PHASE_COND=0, it still wraps only when checkpoint compatibility
  requires it, but the residual branch can stay near-zero.
- If checkpoint contains only gait_phase_gate, it will load that key.
- If checkpoint contains no gait projection weights, the additional projection
  starts as zero-initialized and does not disturb old checkpoints.

Enable:
    EDGE_GAIT_PHASE_COND=1
    EDGE_GAIT_PHASE_DIM=6

Optional:
    EDGE_GAIT_PHASE_INIT_GATE=0.0
    EDGE_GAIT_PHASE_FREQ_BASE=1.0
    EDGE_GAIT_PHASE_SPEED_GAIN=2.0
"""

from __future__ import annotations

import math
import os
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    xmin = x.amin(dim=1, keepdim=True)
    xmax = x.amax(dim=1, keepdim=True)
    return (x - xmin) / (xmax - xmin).clamp_min(1e-8)


def _build_gait_phase_from_traj(
    trajectory: torch.Tensor,
    out_dim: int = 6,
    freq_base: float = 1.0,
    speed_gain: float = 2.0,
) -> torch.Tensor:
    """Build a pseudo gait/support phase from X/Z trajectory.

    trajectory: [B,T,2]
    returns: [B,T,out_dim]

    Feature layout for default dim=6:
      sin(phase), cos(phase),
      sin(phase+pi), cos(phase+pi),
      normalized_speed,
      contact_prior
    """
    traj = trajectory[..., :2].float()
    B, T, _ = traj.shape
    device = traj.device
    dtype = traj.dtype

    if T <= 1:
        return torch.zeros((B, T, out_dim), device=device, dtype=dtype)

    vel = torch.zeros_like(traj)
    vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
    vel[:, 0] = vel[:, 1]

    speed = torch.linalg.norm(vel, dim=-1)
    speed_n = _normalize_01(speed)

    # Speed-controlled phase. Faster trajectory advances gait phase faster.
    step = float(freq_base) + float(speed_gain) * speed_n
    phase = torch.cumsum(step, dim=1)
    phase = 2.0 * math.pi * phase / phase[:, -1:].clamp_min(1e-8)

    left_sin = torch.sin(phase)
    left_cos = torch.cos(phase)
    right_sin = torch.sin(phase + math.pi)
    right_cos = torch.cos(phase + math.pi)

    contact_prior = (0.5 + 0.5 * torch.cos(phase)).clamp(0.0, 1.0)

    feats = torch.stack(
        [left_sin, left_cos, right_sin, right_cos, speed_n, contact_prior],
        dim=-1,
    ).to(dtype=dtype)

    if out_dim == feats.shape[-1]:
        return feats
    if out_dim < feats.shape[-1]:
        return feats[..., :out_dim]

    pad = torch.zeros((B, T, out_dim - feats.shape[-1]), device=device, dtype=dtype)
    return torch.cat([feats, pad], dim=-1)


class GaitPhaseTrajectoryProjection(nn.Module):
    """Residual gait phase wrapper around trajectory_projection.

    Key design:
      self.base -> previous trajectory_projection module
      self.gait_phase_gate -> key seen in current checkpoint logs
    """

    def __init__(self, base: nn.Module, owner, latent_dim: int, gait_dim: int = 6):
        super().__init__()
        self.base = base
        self.owner_ref = lambda: owner
        self.gait_dim = int(gait_dim)
        self.gait_phase_gate = nn.Parameter(
            torch.tensor(float(_env_float("EDGE_GAIT_PHASE_INIT_GATE", 0.0)))
        )
        self.gait_phase_projection = nn.Sequential(
            nn.Linear(self.gait_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        # Safe start: extra branch initially has no effect unless loaded/trained.
        nn.init.zeros_(self.gait_phase_projection[2].weight)
        nn.init.zeros_(self.gait_phase_projection[2].bias)

    def forward(self, traj_features: torch.Tensor) -> torch.Tensor:
        out = self.base(traj_features)
        owner = self.owner_ref()
        if owner is None:
            return out

        if not _env_bool("EDGE_GAIT_PHASE_COND", False):
            return out

        phase = getattr(owner, "_edge_last_gait_phase_cond", None)
        if phase is None:
            return out

        phase = phase.to(device=out.device, dtype=out.dtype)
        if phase.shape[1] != out.shape[1]:
            phase = F.interpolate(
                phase.transpose(1, 2),
                size=out.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        if phase.shape[-1] != self.gait_dim:
            if phase.shape[-1] > self.gait_dim:
                phase = phase[..., : self.gait_dim]
            else:
                pad = torch.zeros(
                    *phase.shape[:-1],
                    self.gait_dim - phase.shape[-1],
                    device=phase.device,
                    dtype=phase.dtype,
                )
                phase = torch.cat([phase, pad], dim=-1)

        return out + torch.tanh(self.gait_phase_gate).to(out.dtype) * self.gait_phase_projection(phase)


def install_trajectory_gait_phase_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print(f"⚠️ trajectory gait phase patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_gait_phase_patch_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        latent_dim = int(getattr(self.null_trajectory_embed, "shape", [1, 1, 512])[-1])
        gait_dim = _env_int("EDGE_GAIT_PHASE_DIM", 6)

        if not isinstance(getattr(self, "trajectory_projection", None), GaitPhaseTrajectoryProjection):
            self.trajectory_projection = GaitPhaseTrajectoryProjection(
                self.trajectory_projection,
                owner=self,
                latent_dim=latent_dim,
                gait_dim=gait_dim,
            )

        if verbose:
            print(
                "🦶 Gait phase trajectory wrapper initialized: "
                f"enabled={_env_bool('EDGE_GAIT_PHASE_COND', False)}, "
                f"gait_dim={gait_dim}, "
                f"projection={type(self.trajectory_projection).__name__}"
            )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        self._edge_last_gait_phase_cond = None

        if isinstance(cond_embed, dict):
            gait = cond_embed.get("gait_phase", None)
            traj = cond_embed.get("trajectory", None)

            if gait is not None:
                gait = gait.to(device=device, dtype=dtype) if torch.is_tensor(gait) else torch.as_tensor(gait, device=device, dtype=dtype)
                if gait.ndim == 2:
                    gait = gait.unsqueeze(0)
                if gait.shape[0] == 1 and batch_size > 1:
                    gait = gait.expand(batch_size, -1, -1)
                if gait.shape[1] != seq_len:
                    gait = F.interpolate(gait.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
                self._edge_last_gait_phase_cond = gait
            elif traj is not None and _env_bool("EDGE_GAIT_PHASE_COND", False):
                traj = traj.to(device=device, dtype=dtype) if torch.is_tensor(traj) else torch.as_tensor(traj, device=device, dtype=dtype)
                if traj.ndim == 2:
                    traj = traj.unsqueeze(0)
                if traj.shape[0] == 1 and batch_size > 1:
                    traj = traj.expand(batch_size, -1, -1)
                if traj.shape[1] != seq_len:
                    traj = F.interpolate(traj.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)
                self._edge_last_gait_phase_cond = _build_gait_phase_from_traj(
                    traj[..., :2],
                    out_dim=_env_int("EDGE_GAIT_PHASE_DIM", 6),
                    freq_base=_env_float("EDGE_GAIT_PHASE_FREQ_BASE", 1.0),
                    speed_gain=_env_float("EDGE_GAIT_PHASE_SPEED_GAIN", 2.0),
                ).to(device=device, dtype=dtype)

        return original_prepare(self, cond_embed, batch_size, seq_len, device, dtype)

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder._edge_gait_phase_patch_installed = True

    if verbose:
        print(
            "✅ Installed gait phase trajectory patch: "
            f"enabled={_env_bool('EDGE_GAIT_PHASE_COND', False)}, "
            f"dim={_env_int('EDGE_GAIT_PHASE_DIM', 6)}"
        )
    return True


def install():
    return install_trajectory_gait_phase_patch(verbose=True)
