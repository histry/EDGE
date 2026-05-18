"""
V3D / V4 / V4E retrieved motion soft prior patch for EDGE.

This patch keeps previous modes:
  - upper_safe_plus
  - style_fullbody

And adds V4E support-chain guidance:
  - root Y absolute guidance
  - root X/Z velocity guidance, not hard absolute trajectory copying
  - pelvis / hips / knees / ankles-feet rotation guidance
  - optional weak contact guidance
  - optional hybrid mode: soft blend + gradient step

Why V4E:
  V4D can activate knees/ankles numerically, but without rootY / pelvis / hips
  transfer, the foot motion has no support-chain and remains visually weak.
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


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


def _rot_dims(joints):
    dims = []
    for j in joints:
        dims.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
    return dims


# EDGE 151D:
#   0:4    foot contacts
#   4:7    root xyz, x=4, y=5, z=6
#   7:151  24 joints × 6D rotations
CONTACTS = [0, 1, 2, 3]
ROOT_XZ = [4, 6]
ROOT_Y = [5]

# SMPL-like 24-joint convention used by EDGE/SMPLSkeleton.
PELVIS = _rot_dims([0])
HIPS = _rot_dims([1, 2])
SPINE_TORSO = _rot_dims([3, 6, 9])
KNEES = _rot_dims([4, 5])
ANKLES_FEET = _rot_dims([7, 8, 10, 11])
NECK_HEAD = _rot_dims([12, 15])
SHOULDERS_ARMS_HANDS = _rot_dims([13, 14, 16, 17, 18, 19, 20, 21, 22, 23])

UPPER_SAFE_PLUS = SPINE_TORSO + NECK_HEAD + SHOULDERS_ARMS_HANDS
STYLE_FULLBODY_ROT = (
    PELVIS
    + HIPS
    + SPINE_TORSO
    + KNEES
    + ANKLES_FEET
    + NECK_HEAD
    + SHOULDERS_ARMS_HANDS
)
ALL_ROT = list(range(7, 151))


def _body_part_name() -> str:
    return _env_str("EDGE_V3D_UPPER_PRIOR_BODY_PART", "upper_safe_plus").lower()


def _is_style_body(body_part: str) -> bool:
    return body_part in {"style_fullbody", "fullbody_style", "dunhuang_style", "support_chain"}


def _make_feature_mask(x: torch.Tensor) -> torch.Tensor:
    """Spatial soft-blend mask.

    This mask is for blend guidance. V4E support-chain gradient guidance uses
    its own root-XZ velocity loss, so root X/Z does not need to be strongly
    blended as absolute position.
    """
    mask = torch.zeros_like(x)
    body_part = _body_part_name()

    if body_part in {"arms", "hands", "arms_hands"}:
        dims = SHOULDERS_ARMS_HANDS
    elif body_part in {"torso", "torso_only"}:
        dims = SPINE_TORSO + NECK_HEAD
    elif _is_style_body(body_part):
        dims = STYLE_FULLBODY_ROT
    elif body_part in {"all_rot", "full_rot"}:
        dims = ALL_ROT
    else:
        dims = UPPER_SAFE_PLUS

    mask[:, :, dims] = 1.0

    lower_style_scale = _env_float("EDGE_V3D_LOWER_STYLE_PRIOR_SCALE", 1.0)

    root_xz_scale = _env_float("EDGE_V3D_ROOT_XZ_PRIOR_SCALE", 0.0)
    root_y_default = 0.25 if _is_style_body(body_part) else 0.0
    root_y_scale = _env_float("EDGE_V3D_ROOT_Y_PRIOR_SCALE", root_y_default)
    contact_scale = _env_float("EDGE_V3D_CONTACT_PRIOR_SCALE", 0.0)

    if root_xz_scale > 0:
        mask[:, :, ROOT_XZ] = root_xz_scale
    if root_y_scale > 0:
        mask[:, :, ROOT_Y] = root_y_scale
    if contact_scale > 0:
        mask[:, :, CONTACTS] = contact_scale

    pelvis_scale = _env_float("EDGE_V3D_PELVIS_PRIOR_SCALE", 0.0)
    hips_scale = _env_float("EDGE_V3D_HIPS_PRIOR_SCALE", 0.0)
    knees_scale = _env_float("EDGE_V3D_KNEES_PRIOR_SCALE", 0.0)
    ankles_feet_scale = _env_float("EDGE_V3D_ANKLES_FEET_PRIOR_SCALE", 0.0)

    torso_scale = _env_float("EDGE_V3D_TORSO_PRIOR_SCALE", 0.65)
    neck_scale = _env_float("EDGE_V3D_NECK_HEAD_PRIOR_SCALE", 0.85)
    arms_scale = _env_float("EDGE_V3D_ARMS_PRIOR_SCALE", 1.00)

    if _is_style_body(body_part):
        pelvis_scale = pelvis_scale if pelvis_scale > 0 else 0.35
        hips_scale = hips_scale if hips_scale > 0 else 0.30
        knees_scale = knees_scale if knees_scale > 0 else 0.18
        ankles_feet_scale = ankles_feet_scale if ankles_feet_scale > 0 else 0.08

    mask[:, :, PELVIS] *= pelvis_scale * lower_style_scale
    mask[:, :, HIPS] *= hips_scale * lower_style_scale
    mask[:, :, KNEES] *= knees_scale * lower_style_scale
    mask[:, :, ANKLES_FEET] *= ankles_feet_scale * lower_style_scale

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


def _resize_rhythm_to_x(rhythm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    rhythm = rhythm.to(device=x.device, dtype=x.dtype)
    if rhythm.ndim == 2:
        rhythm = rhythm.unsqueeze(-1)
    if rhythm.ndim != 3:
        raise ValueError(f"rhythm_weight must be [B,T] or [B,T,1], got {tuple(rhythm.shape)}")

    if rhythm.shape[0] == 1 and x.shape[0] > 1:
        rhythm = rhythm.repeat(x.shape[0], 1, 1)
    elif rhythm.shape[0] != x.shape[0]:
        rhythm = rhythm[:1].repeat(x.shape[0], 1, 1)

    if rhythm.shape[1] != x.shape[1]:
        rhythm = torch.nn.functional.interpolate(
            rhythm.transpose(1, 2),
            size=x.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    rhythm = rhythm[..., :1]
    rhythm = rhythm - rhythm.amin(dim=1, keepdim=True)
    rhythm = rhythm / rhythm.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return rhythm.clamp(0.0, 1.0)


def _soft_prior_weight(self, t: torch.Tensor) -> torch.Tensor:
    base = _env_float("EDGE_V3D_UPPER_PRIOR_STRENGTH", 0.22)
    start_frac = _env_float("EDGE_V3D_UPPER_PRIOR_START_FRAC", 0.55)
    gamma = _env_float("EDGE_V3D_UPPER_PRIOR_GAMMA", 1.5)

    total = max(1.0, float(getattr(self, "n_timestep", 1000)))
    progress = 1.0 - (t.to(dtype=torch.float32) / total)
    ramp = ((progress - start_frac) / max(1e-6, 1.0 - start_frac)).clamp(0.0, 1.0)
    return (base * torch.pow(ramp, gamma)).view(-1, 1, 1)


def _apply_v4_rhythm_gain(w: torch.Tensor, x: torch.Tensor, cond: dict) -> torch.Tensor:
    if not _env_bool("EDGE_V4_RHYTHM_ADAPTIVE_PRIOR", False):
        return w

    rhythm = cond.get("rhythm_weight", None)
    if rhythm is None:
        return w

    rhythm = _resize_rhythm_to_x(rhythm, x)
    min_gain = _env_float("EDGE_V4_RHYTHM_MIN_GAIN", 0.65)
    max_gain = _env_float("EDGE_V4_RHYTHM_MAX_GAIN", 1.35)
    gain = min_gain + (max_gain - min_gain) * rhythm
    return w * gain


def _weighted_abs_loss(x, prior, dims, w, scale):
    if scale <= 0 or len(dims) == 0:
        return x.new_tensor(0.0)
    diff = x[:, :, dims] - prior[:, :, dims]
    return scale * (diff.square() * w).mean()


def _weighted_vel_loss(x, prior, dims, w, scale):
    if scale <= 0 or len(dims) == 0 or x.shape[1] < 2:
        return x.new_tensor(0.0)
    xv = x[:, 1:, dims] - x[:, :-1, dims]
    pv = prior[:, 1:, dims] - prior[:, :-1, dims]
    if w.ndim == 3 and w.shape[1] == x.shape[1]:
        wv = 0.5 * (w[:, 1:, :] + w[:, :-1, :])
    else:
        wv = w
    return scale * ((xv - pv).square() * wv).mean()


def _support_chain_gradient_step(x_out: torch.Tensor, prior_noisy: torch.Tensor, w: torch.Tensor):
    """V4E support-chain gradient guidance.

    This is feature-space guidance, not model gradient guidance.
    It works even when the outer DDPM sampler is wrapped in torch.no_grad().
    """
    if not _env_bool("EDGE_V4E_SUPPORT_CHAIN_PRIOR", False):
        return x_out, None

    mode = _env_str("EDGE_V4E_GUIDANCE_MODE", "hybrid").lower()
    if mode not in {"grad", "gradient", "hybrid"}:
        return x_out, None

    grad_step = _env_float("EDGE_V4E_GRAD_STEP", 1.25)
    grad_clip = _env_float("EDGE_V4E_GRAD_CLIP", 0.10)
    if grad_step <= 0:
        return x_out, None

    # Use float32 for stable autograd; cast back afterward.
    with torch.enable_grad():
        x_var = x_out.detach().float().requires_grad_(True)
        prior = prior_noisy.detach().float()
        ww = w.detach().float()
        if ww.ndim == 2:
            ww = ww.unsqueeze(-1)
        if ww.shape[1] == 1 and x_var.shape[1] > 1:
            ww = ww.expand(-1, x_var.shape[1], -1)

        root_y_scale = _env_float("EDGE_V4E_ROOT_Y_LOSS_SCALE", _env_float("EDGE_V3D_ROOT_Y_PRIOR_SCALE", 0.55))
        root_xz_abs_scale = _env_float("EDGE_V4E_ROOT_XZ_ABS_LOSS_SCALE", 0.0)
        root_xz_vel_scale = _env_float("EDGE_V4E_ROOT_XZ_VEL_LOSS_SCALE", 0.55)
        contact_scale = _env_float("EDGE_V4E_CONTACT_LOSS_SCALE", _env_float("EDGE_V3D_CONTACT_PRIOR_SCALE", 0.05))

        pelvis_scale = _env_float("EDGE_V4E_PELVIS_LOSS_SCALE", _env_float("EDGE_V3D_PELVIS_PRIOR_SCALE", 0.60))
        hips_scale = _env_float("EDGE_V4E_HIPS_LOSS_SCALE", _env_float("EDGE_V3D_HIPS_PRIOR_SCALE", 0.52))
        knees_scale = _env_float("EDGE_V4E_KNEES_LOSS_SCALE", _env_float("EDGE_V3D_KNEES_PRIOR_SCALE", 0.45))
        ankles_scale = _env_float("EDGE_V4E_ANKLES_FEET_LOSS_SCALE", _env_float("EDGE_V3D_ANKLES_FEET_PRIOR_SCALE", 0.25))

        torso_scale = _env_float("EDGE_V4E_TORSO_LOSS_SCALE", _env_float("EDGE_V3D_TORSO_PRIOR_SCALE", 1.20))
        neck_scale = _env_float("EDGE_V4E_NECK_HEAD_LOSS_SCALE", _env_float("EDGE_V3D_NECK_HEAD_PRIOR_SCALE", 1.10))
        arms_scale = _env_float("EDGE_V4E_ARMS_LOSS_SCALE", _env_float("EDGE_V3D_ARMS_PRIOR_SCALE", 0.45))

        loss = x_var.new_tensor(0.0)

        # Absolute root Y: Dunhuang sinking / rising center of mass.
        loss = loss + _weighted_abs_loss(x_var, prior, ROOT_Y, ww, root_y_scale)

        # Root X/Z should mainly match velocity, not absolute stage position.
        loss = loss + _weighted_abs_loss(x_var, prior, ROOT_XZ, ww, root_xz_abs_scale)
        loss = loss + _weighted_vel_loss(x_var, prior, ROOT_XZ, ww, root_xz_vel_scale)

        # Support chain.
        loss = loss + _weighted_abs_loss(x_var, prior, PELVIS, ww, pelvis_scale)
        loss = loss + _weighted_abs_loss(x_var, prior, HIPS, ww, hips_scale)
        loss = loss + _weighted_abs_loss(x_var, prior, KNEES, ww, knees_scale)
        loss = loss + _weighted_abs_loss(x_var, prior, ANKLES_FEET, ww, ankles_scale)

        # Keep torso/neck style, reduce arm dominance if configured.
        loss = loss + _weighted_abs_loss(x_var, prior, SPINE_TORSO, ww, torso_scale)
        loss = loss + _weighted_abs_loss(x_var, prior, NECK_HEAD, ww, neck_scale)
        loss = loss + _weighted_abs_loss(x_var, prior, SHOULDERS_ARMS_HANDS, ww, arms_scale)

        # Contact is optional and weak; avoid hard-copying contact binary channels.
        loss = loss + _weighted_abs_loss(x_var, prior, CONTACTS, ww, contact_scale)

        grad = torch.autograd.grad(loss, x_var, retain_graph=False, create_graph=False)[0]

        if grad_clip > 0:
            grad = grad.clamp(min=-grad_clip, max=grad_clip)

        x_next = x_var - grad_step * grad
        grad_rms = grad.detach().square().mean().sqrt()

    return x_next.detach().to(dtype=x_out.dtype, device=x_out.device), {
        "loss": float(loss.detach().cpu()),
        "grad_rms": float(grad_rms.detach().cpu()),
    }


def install_v3d_upper_soft_prior_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3D/V4/V4E soft prior patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3d_upper_soft_prior_patched", False):
        if verbose:
            print("✅ V3D/V4/V4E soft prior patch already installed")
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

            # Match current x_{t-1} noise level approximately.
            t_next = (t - 1).clamp_min(0)
            prior_noisy = self.q_sample(prior_clean, t_next, noise=torch.zeros_like(prior_clean))

            w = _soft_prior_weight(self, t)
            w = _apply_v4_rhythm_gain(w, x_out, cond)

            mode = _env_str("EDGE_V4E_GUIDANCE_MODE", "blend").lower()
            if _env_bool("EDGE_V4E_SUPPORT_CHAIN_PRIOR", False) and mode == "blend":
                mode = "hybrid"

            x_guided = x_out
            mask = _make_feature_mask(x_out)
            blend = None

            if mode in {"blend", "hybrid"}:
                max_blend = _env_float("EDGE_V3D_PRIOR_MAX_BLEND", 0.85)
                blend = (w * mask).clamp(0.0, max_blend)
                x_guided = x_guided * (1.0 - blend) + prior_noisy * blend

            grad_info = None
            if mode in {"grad", "gradient", "hybrid"}:
                x_guided, grad_info = _support_chain_gradient_step(x_guided, prior_noisy, w)

            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False) or _env_bool("EDGE_V4_RHYTHM_DEBUG", False) or _env_bool("EDGE_V4E_DEBUG", False):
                if not hasattr(self, "_edge_v3d_prior_debug_counter"):
                    self._edge_v3d_prior_debug_counter = 0
                self._edge_v3d_prior_debug_counter += 1

                if self._edge_v3d_prior_debug_counter <= 20 or self._edge_v3d_prior_debug_counter % 100 == 0:
                    body_part = _body_part_name()
                    msg = (
                        "🧪 V3D/V4/V4E prior | "
                        f"mode={mode} "
                        f"body={body_part} "
                        f"t={int(t[0].detach().cpu())} "
                        f"w_mean={float(w.mean().detach().cpu()):.4f} "
                        f"w_min={float(w.min().detach().cpu()):.4f} "
                        f"w_max={float(w.max().detach().cpu()):.4f} "
                        f"mask_mean={float(mask.mean().detach().cpu()):.4f}"
                    )
                    if blend is not None:
                        msg += (
                            f" blend_mean={float(blend.mean().detach().cpu()):.4f} "
                            f"blend_max={float(blend.max().detach().cpu()):.4f}"
                        )
                    if grad_info is not None:
                        msg += (
                            f" support_loss={grad_info['loss']:.6f} "
                            f"grad_rms={grad_info['grad_rms']:.6f}"
                        )
                    if _env_bool("EDGE_V4_RHYTHM_ADAPTIVE_PRIOR", False) and cond.get("rhythm_weight", None) is not None:
                        rhythm = _resize_rhythm_to_x(cond["rhythm_weight"], x_out)
                        msg += (
                            f" rhythm_mean={float(rhythm.mean().detach().cpu()):.4f} "
                            f"rhythm_min={float(rhythm.min().detach().cpu()):.4f} "
                            f"rhythm_max={float(rhythm.max().detach().cpu()):.4f}"
                        )
                    print(msg, flush=True)

            return x_guided, pred_xstart

        except Exception as exc:
            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False) or _env_bool("EDGE_V4_RHYTHM_DEBUG", False) or _env_bool("EDGE_V4E_DEBUG", False):
                print(f"⚠️ V3D/V4/V4E soft prior skipped: {exc}", flush=True)
            return x_out, pred_xstart

    GaussianDiffusion.p_sample = patched_p_sample
    GaussianDiffusion._edge_v3d_upper_soft_prior_patched = True

    if verbose:
        print("✅ Installed V3D/V4/V4E support-chain soft prior patch")
    return True
