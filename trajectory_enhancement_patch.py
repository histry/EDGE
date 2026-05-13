"""Runtime patch: Fourier/physics/sparse-waypoint/bounded dynamic-CFG trajectory injection.

V2 safety update
----------------
The previous dynamic trajectory CFG applied:

    unc + (conditioned - unc) * w

to all 151 motion features. In the S-trajectory experiments this caused root X/Z
explosion when EDGE_DYNAMIC_TRAJ_CFG=1. Ablation showed:

    dynamic_cfg=1 -> root_path ≈ 103
    dynamic_cfg=0 -> root_path ≈ 17

regardless of RAG context. Therefore dynamic CFG must be bounded and root-safe.

This replacement keeps the original advanced trajectory projection logic, but
rewrites dynamic CFG as:

    conditioned + (conditioned - unc) * bounded_gain * feature_gate

Default safety:
    - root X/Z are preserved from conditioned prediction;
    - contacts are not CFG-amplified;
    - root_y is not CFG-amplified;
    - lower/torso/upper have small configurable gates;
    - bounded_gain is clamped;
    - legacy behavior is available only with EDGE_TRAJ_CFG_LEGACY=1.

Enable:
    EDGE_TRAJ_FOURIER_FEATURES=1
    EDGE_TRAJ_PHYSICS_FEATURES=1
    EDGE_TRAJ_SPARSE_WAYPOINT=1

Safe dynamic CFG:
    EDGE_DYNAMIC_TRAJ_CFG=1
    EDGE_TRAJ_CFG_PRESERVE_ROOT_XZ=1
    EDGE_TRAJ_CFG_FEATURE_GATE=1
    EDGE_TRAJ_CFG_EFFECT_SCALE=0.25
    EDGE_TRAJ_CFG_MAX_GAIN=1.0

Recommended current default:
    EDGE_DYNAMIC_TRAJ_CFG=0
"""

from __future__ import annotations

import os
import weakref
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajectory_representation_utils import (
    build_advanced_trajectory_features,
    build_sparse_waypoint_mask,
    dynamic_traj_cfg_weights,
    env_bool,
    env_float,
    env_int,
    trajectory_to_bev_heatmap,
)


CONTACT_SLICE = slice(0, 4)
ROOT_X_IDX = 4
ROOT_Y_IDX = 5
ROOT_Z_IDX = 6
ROT_START = 7
ROT_DIM = 6

LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
TORSO_JOINTS = [3, 6, 9]
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


def _ensure_btd(x, batch_size, seq_len, device, dtype, name: str):
    if x is None:
        return None
    x = x.to(device=device, dtype=dtype) if torch.is_tensor(x) else torch.as_tensor(x, device=device, dtype=dtype)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"{name} must be [B,T,D] or [T,D], got {tuple(x.shape)}")
    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1, -1)
    if x.shape[0] != batch_size:
        raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {x.shape[0]}")
    if x.shape[1] != seq_len:
        x = F.interpolate(
            x.transpose(1, 2),
            size=seq_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return x


def _rot_indices(joints):
    out = []
    for j in joints:
        start = ROT_START + ROT_DIM * int(j)
        out.extend(range(start, start + ROT_DIM))
    return out


def _make_feature_gate(nfeats: int, device, dtype) -> torch.Tensor:
    """Feature-wise dynamic-CFG gate.

    Gate is applied only to the extra CFG delta:
        conditioned + delta * gain * gate

    Therefore gate=0 means "use the normal conditioned output" for that feature,
    not "remove conditioning".
    """
    gate = torch.ones((nfeats,), device=device, dtype=dtype)

    if not env_bool("EDGE_TRAJ_CFG_FEATURE_GATE", True):
        return gate

    # Safe defaults: do not amplify root/contact channels.
    gate[:] = float(env_float("EDGE_TRAJ_CFG_GATE_DEFAULT", 0.35))

    if nfeats >= 4:
        gate[CONTACT_SLICE] = float(env_float("EDGE_TRAJ_CFG_GATE_CONTACTS", 0.0))
    if nfeats > ROOT_X_IDX:
        gate[ROOT_X_IDX] = float(env_float("EDGE_TRAJ_CFG_GATE_ROOT_XZ", 0.0))
    if nfeats > ROOT_Y_IDX:
        gate[ROOT_Y_IDX] = float(env_float("EDGE_TRAJ_CFG_GATE_ROOT_Y", 0.0))
    if nfeats > ROOT_Z_IDX:
        gate[ROOT_Z_IDX] = float(env_float("EDGE_TRAJ_CFG_GATE_ROOT_XZ", 0.0))

    pelvis_idx = [i for i in _rot_indices([0]) if i < nfeats]
    if pelvis_idx:
        gate[pelvis_idx] = float(env_float("EDGE_TRAJ_CFG_GATE_PELVIS_ROT", 0.10))

    for idxs, value in [
        (_rot_indices(LOWER_JOINTS), float(env_float("EDGE_TRAJ_CFG_GATE_LOWER", 0.35))),
        (_rot_indices(TORSO_JOINTS), float(env_float("EDGE_TRAJ_CFG_GATE_TORSO", 0.50))),
        (_rot_indices(UPPER_JOINTS), float(env_float("EDGE_TRAJ_CFG_GATE_UPPER", 0.50))),
    ]:
        idxs = [i for i in idxs if i < nfeats]
        if idxs:
            gate[idxs] = value

    return gate.clamp(0.0, 1.0)


def _bounded_dynamic_cfg(
    unc: torch.Tensor,
    conditioned: torch.Tensor,
    traj: torch.Tensor,
    guidance_weight: float,
) -> torch.Tensor:
    """Root-safe bounded dynamic trajectory CFG.

    Legacy formula is available with EDGE_TRAJ_CFG_LEGACY=1, but default is safe.
    """
    if env_bool("EDGE_TRAJ_CFG_LEGACY", False):
        w = dynamic_traj_cfg_weights(
            traj,
            base=env_float("EDGE_TRAJ_CFG_BASE", float(guidance_weight)),
            speed_w=env_float("EDGE_TRAJ_CFG_SPEED_W", 2.0),
            curvature_w=env_float("EDGE_TRAJ_CFG_CURVATURE_W", 1.0),
            min_w=env_float("EDGE_TRAJ_CFG_MIN", 1.0),
            max_w=env_float("EDGE_TRAJ_CFG_MAX", max(float(guidance_weight), 5.0)),
        ).to(device=conditioned.device, dtype=conditioned.dtype)
        return unc + (conditioned - unc) * w

    # Safe bounded defaults. Even if old env sets max=5.0, the effective gain
    # is additionally scaled and clamped below.
    w = dynamic_traj_cfg_weights(
        traj,
        base=env_float("EDGE_TRAJ_CFG_BASE", 1.0),
        speed_w=env_float("EDGE_TRAJ_CFG_SPEED_W", 0.50),
        curvature_w=env_float("EDGE_TRAJ_CFG_CURVATURE_W", 0.25),
        min_w=env_float("EDGE_TRAJ_CFG_MIN", 1.0),
        max_w=env_float("EDGE_TRAJ_CFG_MAX", 2.0),
    ).to(device=conditioned.device, dtype=conditioned.dtype)

    # Convert CFG weight to extra gain beyond normal conditioning.
    # w=1 -> conditioned output.
    effect_scale = float(env_float("EDGE_TRAJ_CFG_EFFECT_SCALE", 0.25))
    max_gain = float(env_float("EDGE_TRAJ_CFG_MAX_GAIN", 1.0))
    gain = ((w - 1.0) * effect_scale).clamp(0.0, max_gain)

    delta = conditioned - unc

    # Optional global delta clamp. Disabled by default.
    max_delta = float(env_float("EDGE_TRAJ_CFG_DELTA_CLAMP", 0.0))
    if max_delta > 0:
        delta = delta.clamp(-max_delta, max_delta)

    gate = _make_feature_gate(
        nfeats=conditioned.shape[-1],
        device=conditioned.device,
        dtype=conditioned.dtype,
    ).view(1, 1, -1)

    out = conditioned + delta * gain * gate

    # Hard preserve root X/Z by default. This is the most important fix for
    # the observed root_path=103 explosion.
    if env_bool("EDGE_TRAJ_CFG_PRESERVE_ROOT_XZ", True) and out.shape[-1] > ROOT_Z_IDX:
        out = out.clone()
        out[..., ROOT_X_IDX] = conditioned[..., ROOT_X_IDX]
        out[..., ROOT_Z_IDX] = conditioned[..., ROOT_Z_IDX]

    # Optional root step clamp if users intentionally allow root CFG.
    step_clamp = float(env_float("EDGE_TRAJ_CFG_ROOT_STEP_CLAMP", 0.0))
    if step_clamp > 0 and out.shape[1] > 1 and out.shape[-1] > ROOT_Z_IDX:
        out = out.clone()
        root = out[..., [ROOT_X_IDX, ROOT_Z_IDX]]
        prev = root[:, :1]
        steps = root[:, 1:] - root[:, :-1]
        norm = torch.linalg.norm(steps, dim=-1, keepdim=True).clamp_min(1e-8)
        steps = steps * (step_clamp / norm).clamp(max=1.0)
        root_clamped = torch.cat([prev, prev + torch.cumsum(steps, dim=1)], dim=1)
        out[..., ROOT_X_IDX] = root_clamped[..., 0]
        out[..., ROOT_Z_IDX] = root_clamped[..., 1]

    return out


class EdgeAdvancedTrajectoryProjection(nn.Module):
    """Residual wrapper around the original trajectory_projection."""

    def __init__(self, base: nn.Module, owner, latent_dim: int):
        super().__init__()
        self.base = base
        self.owner_ref = weakref.ref(owner)
        self.latent_dim = int(latent_dim)

        # Max feature dim for physics + Fourier. With 6 bands:
        # physics 4 + fourier 4*2*6 = 52.
        self.adv_dim = env_int("EDGE_TRAJ_ADV_DIM", 52)
        self.adv_projection = nn.Sequential(
            nn.Linear(self.adv_dim, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.adv_gate = nn.Parameter(torch.tensor(float(env_float("EDGE_TRAJ_ADV_INIT_GATE", 0.0))))

        self.mask_projection = nn.Sequential(
            nn.Linear(1, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.mask_gate = nn.Parameter(torch.tensor(float(env_float("EDGE_TRAJ_MASK_INIT_GATE", 0.0))))

        self.bev_projection = nn.Sequential(
            nn.Linear(1, self.latent_dim),
            nn.SiLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
        )
        self.bev_gate = nn.Parameter(torch.tensor(float(env_float("EDGE_TRAJ_BEV_INIT_GATE", 0.0))))

    def _pad_or_trim(self, x, dim):
        if x.shape[-1] == dim:
            return x
        if x.shape[-1] > dim:
            return x[..., :dim]
        pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=-1)

    def forward(self, traj_features):
        out = self.base(traj_features)
        owner = self.owner_ref()
        if owner is None:
            return out

        if env_bool("EDGE_TRAJ_FOURIER_FEATURES", False) or env_bool("EDGE_TRAJ_PHYSICS_FEATURES", False):
            adv = getattr(owner, "_edge_last_traj_adv_features", None)
            if adv is not None:
                adv = adv.to(device=out.device, dtype=out.dtype)
                if adv.shape[1] != out.shape[1]:
                    adv = F.interpolate(
                        adv.transpose(1, 2),
                        size=out.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                adv = self._pad_or_trim(adv, self.adv_dim)
                out = out + torch.tanh(self.adv_gate).to(out.dtype) * self.adv_projection(adv)

        if env_bool("EDGE_TRAJ_SPARSE_WAYPOINT", False):
            mask = getattr(owner, "_edge_last_traj_mask", None)
            if mask is not None:
                mask = mask.to(device=out.device, dtype=out.dtype)
                if mask.shape[1] != out.shape[1]:
                    mask = F.interpolate(
                        mask.transpose(1, 2),
                        size=out.shape[1],
                        mode="nearest",
                    ).transpose(1, 2)
                out = out + torch.tanh(self.mask_gate).to(out.dtype) * self.mask_projection(mask[..., :1])

        if env_bool("EDGE_TRAJ_BEV_COND", False):
            bev = getattr(owner, "_edge_last_bev_cond", None)
            if bev is not None:
                bev = bev.to(device=out.device, dtype=out.dtype)
                if bev.ndim == 4:
                    scalar = bev.mean(dim=(-1, -2)).unsqueeze(1).expand(-1, out.shape[1], -1)
                elif bev.ndim == 3:
                    scalar = bev
                else:
                    scalar = bev.reshape(bev.shape[0], 1, -1).mean(dim=-1, keepdim=True).expand(-1, out.shape[1], -1)
                if scalar.shape[1] != out.shape[1]:
                    scalar = F.interpolate(
                        scalar.transpose(1, 2),
                        size=out.shape[1],
                        mode="linear",
                        align_corners=False,
                    ).transpose(1, 2)
                out = out + torch.tanh(self.bev_gate).to(out.dtype) * self.bev_projection(scalar[..., :1])

        return out


def install_trajectory_enhancement_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print(f"⚠️ trajectory enhancement patch skipped: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_trajectory_enhancement_patch_installed", False):
        return True

    original_init = DanceDecoder.__init__
    original_prepare = DanceDecoder._prepare_cond_inputs
    original_guided_forward = DanceDecoder.guided_forward

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        latent_dim = int(getattr(self.null_trajectory_embed, "shape", [1, 1, 512])[-1])
        if not isinstance(getattr(self, "trajectory_projection", None), EdgeAdvancedTrajectoryProjection):
            self.trajectory_projection = EdgeAdvancedTrajectoryProjection(
                self.trajectory_projection,
                owner=self,
                latent_dim=latent_dim,
            )

        if verbose:
            print(
                "🧭 Advanced trajectory branch initialized: "
                f"fourier={env_bool('EDGE_TRAJ_FOURIER_FEATURES', False)}, "
                f"physics={env_bool('EDGE_TRAJ_PHYSICS_FEATURES', False)}, "
                f"sparse={env_bool('EDGE_TRAJ_SPARSE_WAYPOINT', False)}, "
                f"bev={env_bool('EDGE_TRAJ_BEV_COND', False)}"
            )

    @wraps(original_prepare)
    def patched_prepare(self, cond_embed, batch_size, seq_len, device, dtype):
        cond_for_original = cond_embed
        traj = cond_embed.get("trajectory", None) if isinstance(cond_embed, dict) else None

        self._edge_last_traj_adv_features = None
        self._edge_last_traj_mask = None
        self._edge_last_bev_cond = None

        if traj is not None:
            traj = _ensure_btd(traj, batch_size, seq_len, device, dtype, "trajectory")[..., :2]

            if env_bool("EDGE_TRAJ_FOURIER_FEATURES", False) or env_bool("EDGE_TRAJ_PHYSICS_FEATURES", False):
                self._edge_last_traj_adv_features = build_advanced_trajectory_features(
                    traj,
                    use_fourier=env_bool("EDGE_TRAJ_FOURIER_FEATURES", False),
                    use_physics=env_bool("EDGE_TRAJ_PHYSICS_FEATURES", False),
                    bands=env_int("EDGE_TRAJ_FOURIER_BANDS", 6),
                ).to(device=device, dtype=dtype)

            if env_bool("EDGE_TRAJ_SPARSE_WAYPOINT", False):
                mask = cond_embed.get("trajectory_mask", None) if isinstance(cond_embed, dict) else None
                if mask is None:
                    m = build_sparse_waypoint_mask(seq_len)
                    mask = torch.from_numpy(m).to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    mask = _ensure_btd(mask, batch_size, seq_len, device, dtype, "trajectory_mask")[..., :1]

                self._edge_last_traj_mask = mask

                # Do not destroy dense trajectory by default; the mask is an explicit signal.
                if env_bool("EDGE_TRAJ_APPLY_MASK_TO_COND", False) and isinstance(cond_embed, dict):
                    cond_for_original = dict(cond_embed)
                    cond_for_original["trajectory"] = traj * mask + traj.detach() * (1.0 - mask) * float(env_float("EDGE_TRAJ_MASKED_VALUE_SCALE", 0.0))

            if env_bool("EDGE_TRAJ_BEV_COND", False):
                bev = cond_embed.get("bev_map", None) if isinstance(cond_embed, dict) else None
                if bev is None and env_bool("EDGE_TRAJ_BUILD_BEV_FROM_TRAJ", True):
                    bev_np = trajectory_to_bev_heatmap(
                        traj.detach().cpu().numpy(),
                        size=env_int("EDGE_TRAJ_BEV_SIZE", 32),
                        sigma=env_float("EDGE_TRAJ_BEV_SIGMA", 1.5),
                    )
                    bev = torch.from_numpy(bev_np).to(device=device, dtype=dtype)
                if bev is not None:
                    self._edge_last_bev_cond = bev.to(device=device, dtype=dtype) if torch.is_tensor(bev) else torch.as_tensor(bev, device=device, dtype=dtype)

        return original_prepare(self, cond_for_original, batch_size, seq_len, device, dtype)

    @wraps(original_guided_forward)
    def patched_guided_forward(self, x, cond_embed, times, guidance_weight, force_mask=None, force_x_clean=None):
        if not env_bool("EDGE_DYNAMIC_TRAJ_CFG", False):
            return original_guided_forward(
                self,
                x,
                cond_embed,
                times,
                guidance_weight,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )

        if not isinstance(cond_embed, dict) or cond_embed.get("trajectory", None) is None:
            return original_guided_forward(
                self,
                x,
                cond_embed,
                times,
                guidance_weight,
                force_mask=force_mask,
                force_x_clean=force_x_clean,
            )

        b = x.shape[0]
        device = x.device

        drop_all = torch.zeros((b,), dtype=torch.bool, device=device)
        keep_all = torch.ones((b,), dtype=torch.bool, device=device)

        unc = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=1.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=drop_all,
            keep_traj_mask=drop_all,
            keep_energy_mask=drop_all,
        )

        conditioned = self.forward(
            x,
            cond_embed,
            times,
            cond_drop_prob=0.0,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
            keep_audio_mask=keep_all,
            keep_traj_mask=keep_all,
            keep_energy_mask=keep_all,
        )

        traj = cond_embed["trajectory"].to(device=device, dtype=x.dtype)[..., :2]
        if traj.shape[1] != x.shape[1]:
            traj = F.interpolate(
                traj.transpose(1, 2),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        out = _bounded_dynamic_cfg(
            unc=unc,
            conditioned=conditioned,
            traj=traj,
            guidance_weight=float(guidance_weight),
        )

        if not getattr(self, "_edge_dynamic_cfg_logged", False):
            print(
                "✅ Bounded Dynamic Trajectory CFG active: "
                f"legacy={env_bool('EDGE_TRAJ_CFG_LEGACY', False)}, "
                f"preserve_root_xz={env_bool('EDGE_TRAJ_CFG_PRESERVE_ROOT_XZ', True)}, "
                f"feature_gate={env_bool('EDGE_TRAJ_CFG_FEATURE_GATE', True)}, "
                f"effect_scale={env_float('EDGE_TRAJ_CFG_EFFECT_SCALE', 0.25)}, "
                f"max_gain={env_float('EDGE_TRAJ_CFG_MAX_GAIN', 1.0)}, "
                f"root_step_clamp={env_float('EDGE_TRAJ_CFG_ROOT_STEP_CLAMP', 0.0)}"
            )
            self._edge_dynamic_cfg_logged = True

        return out

    DanceDecoder.__init__ = patched_init
    DanceDecoder._prepare_cond_inputs = patched_prepare
    DanceDecoder.guided_forward = patched_guided_forward
    DanceDecoder._edge_trajectory_enhancement_patch_installed = True

    if verbose:
        print(
            "✅ Installed trajectory enhancement patch v2: "
            f"fourier={env_bool('EDGE_TRAJ_FOURIER_FEATURES', False)}, "
            f"physics={env_bool('EDGE_TRAJ_PHYSICS_FEATURES', False)}, "
            f"sparse={env_bool('EDGE_TRAJ_SPARSE_WAYPOINT', False)}, "
            f"dynamic_cfg={env_bool('EDGE_DYNAMIC_TRAJ_CFG', False)}, "
            f"dynamic_cfg_legacy={env_bool('EDGE_TRAJ_CFG_LEGACY', False)}, "
            f"bev={env_bool('EDGE_TRAJ_BEV_COND', False)}"
        )

    return True


def install():
    return install_trajectory_enhancement_patch(verbose=True)
