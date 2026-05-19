from __future__ import annotations

import os
import torch


_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in _TRUE


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _v3_enabled() -> bool:
    return _env_bool("EDGE_V3_UNIT_RECON", False) or (
        os.environ.get("EDGE_TRAIN_PROFILE", "").strip().lower() == "v3_unit_recon"
    )


def _stable_enabled() -> bool:
    return _v3_enabled() and _env_bool("EDGE_V3_BASE_LOSS_STABILITY", True)


def _safe_tensor(x: torch.Tensor, clip: float) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=clip, neginf=-clip)
    return x.clamp(-clip, clip)


def _huber(diff: torch.Tensor) -> torch.Tensor:
    a = diff.abs()
    return torch.where(a < 1.0, 0.5 * diff.pow(2), a - 0.5)


def _target_scale(target: torch.Tensor, floor: float) -> torch.Tensor:
    scale = torch.sqrt(target.detach().pow(2).reshape(target.shape[0], -1).mean(dim=1) + 1e-8)
    scale = scale.clamp_min(float(floor))
    while scale.ndim < target.ndim:
        scale = scale.unsqueeze(-1)
    return scale


def _robust_per_sample_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    scale_floor: float,
    input_clip: float,
    diff_clip: float,
    sample_cap: float,
) -> torch.Tensor:
    pred = _safe_tensor(pred, input_clip)
    target = _safe_tensor(target, input_clip)
    scale = _target_scale(target, scale_floor)
    diff = ((pred - target) / scale).clamp(-diff_clip, diff_clip)
    per = _huber(diff).reshape(diff.shape[0], -1).mean(dim=1)
    per = torch.nan_to_num(per, nan=0.0, posinf=sample_cap, neginf=0.0)
    return per.clamp(0.0, sample_cap)


def install_v3_base_loss_stability_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3 base loss stability patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3_base_loss_stability_patched", False):
        if verbose:
            print("✅ V3 base loss stability patch already installed")
        return True

    orig_reconstruction_loss = GaussianDiffusion._reconstruction_loss
    orig_velocity_loss = GaussianDiffusion._velocity_loss

    def patched_reconstruction_loss(self, model_motion_x0, target_motion_x0, t):
        if not _stable_enabled():
            return orig_reconstruction_loss(self, model_motion_x0, target_motion_x0, t)

        per = _robust_per_sample_loss(
            model_motion_x0,
            target_motion_x0,
            scale_floor=_env_float("EDGE_V3_RECON_SCALE_FLOOR", 0.05),
            input_clip=_env_float("EDGE_V3_BASE_INPUT_CLIP", 8.0),
            diff_clip=_env_float("EDGE_V3_BASE_DIFF_CLIP", 8.0),
            sample_cap=_env_float("EDGE_V3_RECON_SAMPLE_CAP", 200.0),
        )

        # Keep p2 behavior if enabled in the original diffusion config.
        try:
            per = self._p2_apply(per, t)
        except Exception:
            pass

        return per.mean()

    def patched_velocity_loss(self, model_motion_x0, target_motion_x0, t):
        if not _stable_enabled():
            return orig_velocity_loss(self, model_motion_x0, target_motion_x0, t)

        if model_motion_x0.shape[1] < 2:
            return model_motion_x0.new_tensor(0.0)

        pred_vel = model_motion_x0[:, 1:] - model_motion_x0[:, :-1]
        target_vel = target_motion_x0[:, 1:] - target_motion_x0[:, :-1]

        per = _robust_per_sample_loss(
            pred_vel,
            target_vel,
            scale_floor=_env_float("EDGE_V3_VEL_SCALE_FLOOR", 0.02),
            input_clip=_env_float("EDGE_V3_BASE_VEL_INPUT_CLIP", 8.0),
            diff_clip=_env_float("EDGE_V3_BASE_DIFF_CLIP", 8.0),
            sample_cap=_env_float("EDGE_V3_VEL_SAMPLE_CAP", 100.0),
        )

        try:
            per = self._p2_apply(per, t)
        except Exception:
            pass

        return per.mean()

    GaussianDiffusion._reconstruction_loss = patched_reconstruction_loss
    GaussianDiffusion._velocity_loss = patched_velocity_loss
    GaussianDiffusion._edge_v3_base_loss_stability_patched = True

    if verbose:
        print("✅ Installed V3 base loss stability patch: robust recon/velocity Huber + per-sample caps")
    return True


def install_v3_disable_raw_physical_losses_patch(verbose: bool = True) -> bool:
    """
    Disable legacy raw physical losses in V3 unit reconstruction.

    Why:
      In V3H footwork runs, robust support-chain supervision is already injected
      via v3c_visible_fk_patch._motion_energy_loss. The legacy raw FK/foot/
      biomech/body-stability losses in diffusion.py activate after physical_w
      warmup at epoch 6 and can explode because they use raw MSE/FK values.

    Enable with:
      EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES=1
    """
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3 disable raw physical losses patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3_disable_raw_phys_patched", False):
        if verbose:
            print("✅ V3 disable raw physical losses patch already installed")
        return True

    orig_contact_loss = GaussianDiffusion._contact_loss
    orig_fk_loss = GaussianDiffusion._fk_loss
    orig_foot_sliding_loss = GaussianDiffusion._foot_sliding_loss
    orig_anti_freeze_loss = GaussianDiffusion._anti_freeze_loss
    orig_biomech_loss = GaussianDiffusion._biomech_loss
    orig_root_turn_loss = GaussianDiffusion._root_turn_loss
    orig_contact_turn_loss = GaussianDiffusion._contact_turn_loss
    orig_body_stability_loss = GaussianDiffusion._body_stability_loss

    def _disabled() -> bool:
        return _v3_enabled() and _env_bool("EDGE_V3_DISABLE_RAW_PHYSICAL_LOSSES", True)

    def patched_contact_loss(self, model_motion_x0, target_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_contact_loss(self, model_motion_x0, target_motion_x0)

    def patched_fk_loss(self, model_motion_x0, target_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_fk_loss(self, model_motion_x0, target_motion_x0)

    def patched_foot_sliding_loss(self, model_motion_x0, target_motion_x0=None):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_foot_sliding_loss(self, model_motion_x0, target_motion_x0)

    def patched_anti_freeze_loss(self, model_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_anti_freeze_loss(self, model_motion_x0)

    def patched_biomech_loss(self, model_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_biomech_loss(self, model_motion_x0)

    def patched_root_turn_loss(self, model_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_root_turn_loss(self, model_motion_x0)

    def patched_contact_turn_loss(self, model_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_contact_turn_loss(self, model_motion_x0)

    def patched_body_stability_loss(self, model_motion_x0):
        if _disabled():
            return model_motion_x0.new_tensor(0.0)
        return orig_body_stability_loss(self, model_motion_x0)

    GaussianDiffusion._contact_loss = patched_contact_loss
    GaussianDiffusion._fk_loss = patched_fk_loss
    GaussianDiffusion._foot_sliding_loss = patched_foot_sliding_loss
    GaussianDiffusion._anti_freeze_loss = patched_anti_freeze_loss
    GaussianDiffusion._biomech_loss = patched_biomech_loss
    GaussianDiffusion._root_turn_loss = patched_root_turn_loss
    GaussianDiffusion._contact_turn_loss = patched_contact_turn_loss
    GaussianDiffusion._body_stability_loss = patched_body_stability_loss
    GaussianDiffusion._edge_v3_disable_raw_phys_patched = True

    if verbose:
        print("✅ Installed V3 raw physical-loss disable patch: legacy FK/foot/biomech/stability losses disabled in V3")
    return True
