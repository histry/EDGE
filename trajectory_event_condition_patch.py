"""Native trajectory-event conditioning patch for EDGE.

Goal
----
Use X/Z ground-plane trajectory as native model condition, not as hard latent
replacement.  This patch adds a zero-init residual event adapter on top of the
existing trajectory_projection branch.

It turns a target trajectory into:
  X/Z path
  speed
  heading sin/cos
  curvature
  turn gate
  support-prepare gate
  expressive-response gate
  acceleration
  speed gate
  signed curvature

and injects these event tokens into trajectory tokens before the existing
trajectory encoder / decoder trajectory adapters consume them.

Default behavior is OFF. Enable with:
  EDGE_TRAJ_EVENT_COND=1

Recommended after v2b endpoint-continuous is stable:
  EDGE_TRAJ_PHYSICS_FEATURES=1
  EDGE_TRAJ_EVENT_COND=1
  EDGE_TRAJ_EVENT_INIT_GATE=0.0
  EDGE_TURN_EVENT_COUNT=3
  EDGE_TURN_SUPPORT_LAG=8
  EDGE_TURN_EXPR_LAG=4
  EDGE_TURN_GATE_SIGMA=5.0
"""

from __future__ import annotations

import os
import weakref
from functools import wraps
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


def env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _ensure_btd(x, batch_size, seq_len, device, dtype, name):
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, device=device, dtype=dtype)
    else:
        x = x.to(device=device, dtype=dtype)

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError("%s must be [B,T,D] or [T,D], got %s" % (name, tuple(x.shape)))

    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1)

    if x.shape[0] != batch_size:
        raise ValueError("%s batch mismatch: expected %d, got %d" % (name, batch_size, x.shape[0]))

    if x.shape[1] != seq_len:
        x = F.interpolate(
            x.transpose(1, 2),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    return x


def _pad_or_trim(x, dim):
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _build_event_features_from_traj(traj):
    """Return [B,T,D] event features from [B,T,2] trajectory."""
    try:
        from turn_aware_event_utils import event_feature_matrix_torch
        return event_feature_matrix_torch(
            traj,
            count=env_int("EDGE_TURN_EVENT_COUNT", env_int("EDGE_TURN_TOPK", 3)),
            support_lag=env_int("EDGE_TURN_SUPPORT_LAG", 8),
            expressive_lag=env_int("EDGE_TURN_EXPR_LAG", 4),
            min_gap=env_int("EDGE_TURN_MIN_GAP", 18),
            gate_sigma=env_float("EDGE_TURN_GATE_SIGMA", 5.0),
        )
    except Exception:
        # Safe fallback: differentiable physics-only event-ish features.
        B, T, _ = traj.shape
        vel = torch.zeros_like(traj)
        if T > 1:
            vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
            vel[:, 0] = vel[:, 1]
        speed = torch.sqrt((vel * vel).sum(dim=-1, keepdim=True) + 1e-8)
        speed_norm = speed / speed.amax(dim=1, keepdim=True).clamp_min(1e-6)
        heading = vel / speed.clamp_min(1e-6)
        heading_sin = heading[..., 1:2]
        heading_cos = heading[..., 0:1]
        curv = torch.zeros((B, T, 1), device=traj.device, dtype=traj.dtype)
        if T > 2:
            v1 = traj[:, 1:-1] - traj[:, :-2]
            v2 = traj[:, 2:] - traj[:, 1:-1]
            n1 = torch.sqrt((v1 * v1).sum(dim=-1, keepdim=True) + 1e-8)
            n2 = torch.sqrt((v2 * v2).sum(dim=-1, keepdim=True) + 1e-8)
            cos = (v1 * v2).sum(dim=-1, keepdim=True) / (n1 * n2).clamp_min(1e-8)
            curv[:, 1:-1] = 1.0 - cos.clamp(-1.0, 1.0)
        curv_norm = curv / curv.amax(dim=1, keepdim=True).clamp_min(1e-6)
        zero = torch.zeros_like(speed_norm)
        # Layout-compatible 12D fallback.
        return torch.cat(
            [
                traj[..., 0:1],
                traj[..., 1:2],
                speed_norm,
                heading_sin,
                heading_cos,
                curv_norm,
                curv_norm,  # turn gate
                zero,       # support gate
                zero,       # expressive gate
                zero,       # acceleration
                speed_norm, # speed gate
                curv_norm,  # signed curvature fallback
            ],
            dim=-1,
        )


class TrajectoryEventProjectionAdapter(nn.Module):
    """Zero-init residual adapter over trajectory token projection.

    It wraps the existing trajectory_projection module.  The base projection
    still receives [X,Z,dX,dZ].  Event features are projected as a residual.

    Gate starts at 0 by default, so loading old checkpoints remains safe.
    """

    def __init__(self, base, owner, latent_dim, event_dim=12):
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
        self.event_gate = nn.Parameter(torch.tensor(float(env_float("EDGE_TRAJ_EVENT_INIT_GATE", 0.0))))

        # Make the residual branch exactly zero at initialization unless the
        # user explicitly wants a nonzero random adapter.
        if env_bool("EDGE_TRAJ_EVENT_ZERO_LAST", True):
            last_linear = None
            for module in self.event_projection.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def forward(self, traj_features):
        out = self.base(traj_features)

        if not env_bool("EDGE_TRAJ_EVENT_COND", False):
            return out

        owner = self.owner_ref()
        if owner is None:
            return out

        event = getattr(owner, "_edge_last_traj_event_features", None)
        if event is None:
            return out

        event = event.to(device=out.device, dtype=out.dtype)
        if event.shape[1] != out.shape[1]:
            event = F.interpolate(
                event.transpose(1, 2),
                size=out.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        event = _pad_or_trim(event, self.event_dim)
        residual = self.event_projection(event)
        gate = torch.tanh(self.event_gate).to(dtype=out.dtype)

        return out + gate * residual


def install_trajectory_event_condition_patch(verbose=True):
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print("⚠️ trajectory event condition patch skipped: %s" % exc)
        return False

    if getattr(DanceDecoder, "_edge_trajectory_event_condition_patch_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        latent_dim = int(getattr(self.null_trajectory_embed, "shape", [1, 1, 512])[-1])
        event_dim = env_int("EDGE_TRAJ_EVENT_DIM", 12)

        if not isinstance(getattr(self, "trajectory_projection", None), TrajectoryEventProjectionAdapter):
            self.trajectory_projection = TrajectoryEventProjectionAdapter(
                self.trajectory_projection,
                owner=self,
                latent_dim=latent_dim,
                event_dim=event_dim,
            )

        if verbose:
            print(
                "🧭 Trajectory-event adapter initialized: "
                "enabled=%s, event_dim=%d, init_gate=%.4f"
                % (
                    env_bool("EDGE_TRAJ_EVENT_COND", False),
                    event_dim,
                    env_float("EDGE_TRAJ_EVENT_INIT_GATE", 0.0),
                )
            )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        audio_cond, trajectory_abs, energy_cond, rag_summary_cond = original_prepare(
            self,
            cond_embed,
            batch_size,
            seq_len,
            device,
            dtype,
        )

        self._edge_last_traj_event_features = None

        if not env_bool("EDGE_TRAJ_EVENT_COND", False):
            return audio_cond, trajectory_abs, energy_cond, rag_summary_cond

        event_features = None

        if isinstance(cond_embed, dict) and cond_embed.get("trajectory_event", None) is not None:
            event_features = _ensure_btd(
                cond_embed.get("trajectory_event"),
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
                name="trajectory_event",
            )
        elif trajectory_abs is not None and env_bool("EDGE_TRAJ_EVENT_AUTODETECT", True):
            event_features = _build_event_features_from_traj(trajectory_abs[..., :2])

        if event_features is not None:
            self._edge_last_traj_event_features = event_features.to(device=device, dtype=dtype)

            if verbose and not getattr(self, "_edge_traj_event_logged", False):
                try:
                    print(
                        "✅ Trajectory-event condition active: "
                        "shape=%s, gate=%.6f"
                        % (
                            tuple(self._edge_last_traj_event_features.shape),
                            float(torch.tanh(self.trajectory_projection.event_gate).detach().cpu().item())
                            if hasattr(self.trajectory_projection, "event_gate")
                            else 0.0,
                        )
                    )
                except Exception:
                    print("✅ Trajectory-event condition active")
                self._edge_traj_event_logged = True

        return audio_cond, trajectory_abs, energy_cond, rag_summary_cond

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder._edge_trajectory_event_condition_patch_installed = True

    if verbose:
        print(
            "✅ Installed trajectory-event condition patch: "
            "enabled=%s, autodetect=%s"
            % (
                env_bool("EDGE_TRAJ_EVENT_COND", False),
                env_bool("EDGE_TRAJ_EVENT_AUTODETECT", True),
            )
        )

    return True


def install():
    return install_trajectory_event_condition_patch(verbose=True)
