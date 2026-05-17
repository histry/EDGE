"""
V3D retrieved upper_safe_plus soft prior sampling patch.

Purpose
-------
Turn deterministic upper/lower merge into inference-time soft prior sampling.

It softly guides only upper_safe_plus rotation channels during DDPM sampling:
  spine / chest / neck / head / shoulders / elbows / wrists / hands

It does NOT guide:
  contacts, root position, pelvis root translation, knees, ankles, feet

Enable:
  EDGE_V3D_UPPER_SOFT_PRIOR=1
  EDGE_V3D_UPPER_PRIOR_STRENGTH=0.22

The condition dict must contain:
  cond["retrieved_prior_motion"] = normalized [B,T,151] prior motion
"""

from __future__ import annotations

import os
import torch


_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _rot_dims(joints):
    dims = []
    for j in joints:
        dims.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
    return dims


# SMPL-ish 24-joint 6D layout in EDGE 151D:
# 0:4 contact, 4:7 root pos, 7:151 joint rotations.
SPINE_TORSO = _rot_dims([3, 6, 9])
NECK_HEAD = _rot_dims([12, 15])
SHOULDERS_ARMS_HANDS = _rot_dims([13, 14, 16, 17, 18, 19, 20, 21, 22, 23])
UPPER_SAFE_PLUS = SPINE_TORSO + NECK_HEAD + SHOULDERS_ARMS_HANDS


def _make_feature_mask(x: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(x)

    body_part = os.environ.get("EDGE_V3D_UPPER_PRIOR_BODY_PART", "upper_safe_plus").strip().lower()

    if body_part in {"arms", "hands", "arms_hands"}:
        dims = SHOULDERS_ARMS_HANDS
    elif body_part in {"torso", "torso_only"}:
        dims = SPINE_TORSO + NECK_HEAD
    else:
        dims = UPPER_SAFE_PLUS

    mask[:, :, dims] = 1.0

    # Optional softer torso.
    torso_scale = _env_float("EDGE_V3D_TORSO_PRIOR_SCALE", 0.65)
    neck_scale = _env_float("EDGE_V3D_NECK_HEAD_PRIOR_SCALE", 0.85)
    arms_scale = _env_float("EDGE_V3D_ARMS_PRIOR_SCALE", 1.00)

    mask[:, :, SPINE_TORSO] *= torso_scale
    mask[:, :, NECK_HEAD] *= neck_scale
    mask[:, :, SHOULDERS_ARMS_HANDS] *= arms_scale

    return mask


def _resize_prior_to_x(prior: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    prior = prior.to(device=x.device, dtype=x.dtype)
    if prior.ndim != 3 or prior.shape[-1] != x.shape[-1]:
        raise ValueError(f"retrieved_prior_motion must be [B,T,C], got {tuple(prior.shape)}")

    if prior.shape[0] == 1 and x.shape[0] > 1:
        prior = prior.repeat(x.shape[0], 1, 1)
    elif prior.shape[0] != x.shape[0]:
        prior = prior[:1].repeat(x.shape[0], 1, 1)

    if prior.shape[1] != x.shape[1]:
        prior = torch.nn.functional.interpolate(
            prior.transpose(1, 2),
            size=x.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    return prior


def _soft_prior_weight(self, t: torch.Tensor) -> torch.Tensor:
    """
    Late-step guidance schedule.

    Early noisy steps: almost no guidance.
    Late denoising steps: stronger upper prior.
    """
    base = _env_float("EDGE_V3D_UPPER_PRIOR_STRENGTH", 0.22)
    start_frac = _env_float("EDGE_V3D_UPPER_PRIOR_START_FRAC", 0.55)
    gamma = _env_float("EDGE_V3D_UPPER_PRIOR_GAMMA", 1.5)

    total = max(1.0, float(getattr(self, "n_timestep", 1000)))
    progress = 1.0 - (t.to(dtype=torch.float32) / total)  # 0 early -> 1 late

    # guidance starts after progress >= start_frac
    ramp = ((progress - start_frac) / max(1e-6, 1.0 - start_frac)).clamp(0.0, 1.0)
    weight = base * torch.pow(ramp, gamma)
    return weight.view(-1, 1, 1)


def install_v3d_upper_soft_prior_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3D upper soft prior patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3d_upper_soft_prior_patched", False):
        if verbose:
            print("✅ V3D upper soft prior patch already installed")
        return True

    orig_p_sample = GaussianDiffusion.p_sample

    def patched_p_sample(self, x, cond, t, constraint=None, use_tto=True):
        x_out, pred_xstart = orig_p_sample(
            self,
            x,
            cond,
            t,
            constraint=constraint,
            use_tto=use_tto,
        )

        if not _env_bool("EDGE_V3D_UPPER_SOFT_PRIOR", False):
            return x_out, pred_xstart

        if not isinstance(cond, dict) or cond.get("retrieved_prior_motion", None) is None:
            return x_out, pred_xstart

        try:
            prior_clean = _resize_prior_to_x(cond["retrieved_prior_motion"], x_out)

            # Match the current x_{t-1} noise level approximately.
            t_next = (t - 1).clamp_min(0)
            prior_noisy = self.q_sample(prior_clean, t_next, noise=torch.zeros_like(prior_clean))

            mask = _make_feature_mask(x_out)
            w = _soft_prior_weight(self, t)

            x_guided = x_out * (1.0 - w * mask) + prior_noisy * (w * mask)

            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False):
                if not hasattr(self, "_edge_v3d_prior_debug_counter"):
                    self._edge_v3d_prior_debug_counter = 0
                self._edge_v3d_prior_debug_counter += 1
                if self._edge_v3d_prior_debug_counter <= 20 or self._edge_v3d_prior_debug_counter % 100 == 0:
                    print(
                        "🧪 V3D upper soft prior | "
                        f"t={int(t[0].detach().cpu())} "
                        f"w={float(w.mean().detach().cpu()):.4f} "
                        f"mask_mean={float(mask.mean().detach().cpu()):.4f}",
                        flush=True,
                    )

            return x_guided, pred_xstart

        except Exception as exc:
            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False):
                print(f"⚠️ V3D upper soft prior skipped: {exc}", flush=True)
            return x_out, pred_xstart

    GaussianDiffusion.p_sample = patched_p_sample
    GaussianDiffusion._edge_v3d_upper_soft_prior_patched = True

    if verbose:
        print("✅ Installed V3D retrieved upper_safe_plus soft prior sampling patch")
    return True
