"""
V3D/V4 retrieved motion soft prior patch.

Original V3D/V4 only guided upper_safe_plus rotations:
  torso / neck / shoulders / arms / hands

This replacement keeps the old behavior by default, but adds a
style-fullbody prior mode for Dunhuang dance:

  body_part=upper_safe_plus:
      old behavior, upper-body only.

  body_part=style_fullbody:
      soft spatial prior over:
        - root Y only, not root X/Z by default
        - pelvis rotation
        - hips rotation
        - knees rotation
        - ankles / feet rotation, very weak
        - spine / torso / neck / head
        - shoulders / arms / hands

The goal is to preserve Dunhuang "three-bend" silhouette:
  head-neck + chest-waist + pelvis/legs S-curve,
without hard-copying root X/Z or foot contacts.

Important:
  - Root X/Z remains disabled by default.
  - Contact channels remain disabled by default.
  - Lower body is weakly guided, not hard projected.
  - This is still a soft prior in diffusion space, not post-hoc IK.

Enable V3D:
  EDGE_V3D_UPPER_SOFT_PRIOR=1
  EDGE_V3D_UPPER_PRIOR_STRENGTH=0.35

Enable style-fullbody:
  EDGE_V3D_UPPER_PRIOR_BODY_PART=style_fullbody
  EDGE_V3D_ROOT_Y_PRIOR_SCALE=0.25
  EDGE_V3D_PELVIS_PRIOR_SCALE=0.35
  EDGE_V3D_HIPS_PRIOR_SCALE=0.30
  EDGE_V3D_KNEES_PRIOR_SCALE=0.18
  EDGE_V3D_ANKLES_FEET_PRIOR_SCALE=0.08

Enable V4 rhythm-adaptive prior:
  EDGE_V4_RHYTHM_ADAPTIVE_PRIOR=1
  cond["rhythm_weight"] = [B,T,1] or [B,T]
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


def _rot_dims(joints):
    dims = []
    for j in joints:
        dims.extend(range(7 + 6 * j, 7 + 6 * (j + 1)))
    return dims


# EDGE 151D:
#   0:4    foot contacts
#   4:7    root xyz, where x=4, y=5, z=6
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


def _make_feature_mask(x: torch.Tensor) -> torch.Tensor:
    """Build a spatially weighted soft-prior mask.

    This is deliberately not a 0/1 hard split for Dunhuang:
    upper receives stronger guidance, pelvis/hips/knees receive weaker guidance,
    feet/contact/root_xz are either off or very weak unless explicitly enabled.
    """
    mask = torch.zeros_like(x)

    body_part = os.environ.get(
        "EDGE_V3D_UPPER_PRIOR_BODY_PART", "upper_safe_plus"
    ).strip().lower()

    if body_part in {"arms", "hands", "arms_hands"}:
        dims = SHOULDERS_ARMS_HANDS
    elif body_part in {"torso", "torso_only"}:
        dims = SPINE_TORSO + NECK_HEAD
    elif body_part in {"style_fullbody", "fullbody_style", "dunhuang_style"}:
        dims = STYLE_FULLBODY_ROT
    elif body_part in {"all_rot", "full_rot"}:
        dims = ALL_ROT
    else:
        dims = UPPER_SAFE_PLUS

    mask[:, :, dims] = 1.0

    # Global lower-body style multiplier.
    lower_style_scale = _env_float("EDGE_V3D_LOWER_STYLE_PRIOR_SCALE", 1.0)

    # Non-rotation channels. Disabled by default except root Y in style-fullbody.
    root_xz_scale = _env_float("EDGE_V3D_ROOT_XZ_PRIOR_SCALE", 0.0)
    root_y_scale_default = 0.25 if body_part in {
        "style_fullbody",
        "fullbody_style",
        "dunhuang_style",
    } else 0.0
    root_y_scale = _env_float("EDGE_V3D_ROOT_Y_PRIOR_SCALE", root_y_scale_default)
    contact_scale = _env_float("EDGE_V3D_CONTACT_PRIOR_SCALE", 0.0)

    if root_xz_scale > 0:
        mask[:, :, ROOT_XZ] = root_xz_scale
    if root_y_scale > 0:
        mask[:, :, ROOT_Y] = root_y_scale
    if contact_scale > 0:
        mask[:, :, CONTACTS] = contact_scale

    # Rotation-channel scales.
    pelvis_scale = _env_float("EDGE_V3D_PELVIS_PRIOR_SCALE", 0.0)
    hips_scale = _env_float("EDGE_V3D_HIPS_PRIOR_SCALE", 0.0)
    knees_scale = _env_float("EDGE_V3D_KNEES_PRIOR_SCALE", 0.0)
    ankles_feet_scale = _env_float("EDGE_V3D_ANKLES_FEET_PRIOR_SCALE", 0.0)

    # Preserve old upper-only defaults.
    torso_scale = _env_float("EDGE_V3D_TORSO_PRIOR_SCALE", 0.65)
    neck_scale = _env_float("EDGE_V3D_NECK_HEAD_PRIOR_SCALE", 0.85)
    arms_scale = _env_float("EDGE_V3D_ARMS_PRIOR_SCALE", 1.00)

    if body_part in {"style_fullbody", "fullbody_style", "dunhuang_style"}:
        if pelvis_scale <= 0:
            pelvis_scale = 0.35
        if hips_scale <= 0:
            hips_scale = 0.30
        if knees_scale <= 0:
            knees_scale = 0.18
        if ankles_feet_scale <= 0:
            ankles_feet_scale = 0.08
        if torso_scale <= 0:
            torso_scale = 1.15
        if neck_scale <= 0:
            neck_scale = 1.10

    # Lower-body style should be weak but nonzero for Dunhuang silhouette.
    mask[:, :, PELVIS] *= pelvis_scale * lower_style_scale
    mask[:, :, HIPS] *= hips_scale * lower_style_scale
    mask[:, :, KNEES] *= knees_scale * lower_style_scale
    mask[:, :, ANKLES_FEET] *= ankles_feet_scale * lower_style_scale

    # Upper and spine style.
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
    weight = base * torch.pow(ramp, gamma)
    return weight.view(-1, 1, 1)


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


def install_v3d_upper_soft_prior_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3D/V4 soft prior patch skipped: {exc}")
        return False

    # In a fresh Python process this will be False. Keeping the same flag
    # preserves compatibility with existing scripts.
    if getattr(GaussianDiffusion, "_edge_v3d_upper_soft_prior_patched", False):
        if verbose:
            print("✅ V3D/V4 soft prior patch already installed")
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
            # This keeps guidance inside the diffusion trajectory instead of
            # directly hard-overwriting clean x0 features.
            t_next = (t - 1).clamp_min(0)
            prior_noisy = self.q_sample(prior_clean, t_next, noise=torch.zeros_like(prior_clean))

            mask = _make_feature_mask(x_out)
            w = _soft_prior_weight(self, t)
            w = _apply_v4_rhythm_gain(w, x_out, cond)

            max_blend = _env_float("EDGE_V3D_PRIOR_MAX_BLEND", 0.85)
            blend = (w * mask).clamp(0.0, max_blend)

            x_guided = x_out * (1.0 - blend) + prior_noisy * blend

            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False) or _env_bool("EDGE_V4_RHYTHM_DEBUG", False):
                if not hasattr(self, "_edge_v3d_prior_debug_counter"):
                    self._edge_v3d_prior_debug_counter = 0
                self._edge_v3d_prior_debug_counter += 1
                if self._edge_v3d_prior_debug_counter <= 20 or self._edge_v3d_prior_debug_counter % 100 == 0:
                    body_part = os.environ.get("EDGE_V3D_UPPER_PRIOR_BODY_PART", "upper_safe_plus")
                    msg = (
                        "🧪 V3D/V4 soft prior | "
                        f"body={body_part} "
                        f"t={int(t[0].detach().cpu())} "
                        f"w_mean={float(w.mean().detach().cpu()):.4f} "
                        f"w_min={float(w.min().detach().cpu()):.4f} "
                        f"w_max={float(w.max().detach().cpu()):.4f} "
                        f"mask_mean={float(mask.mean().detach().cpu()):.4f} "
                        f"blend_mean={float(blend.mean().detach().cpu()):.4f} "
                        f"blend_max={float(blend.max().detach().cpu()):.4f}"
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
            if _env_bool("EDGE_V3D_UPPER_PRIOR_DEBUG", False) or _env_bool("EDGE_V4_RHYTHM_DEBUG", False):
                print(f"⚠️ V3D/V4 soft prior skipped: {exc}", flush=True)
            return x_out, pred_xstart

    GaussianDiffusion.p_sample = patched_p_sample
    GaussianDiffusion._edge_v3d_upper_soft_prior_patched = True

    if verbose:
        print("✅ Installed V3D/V4 style-fullbody soft prior patch")
    return True
