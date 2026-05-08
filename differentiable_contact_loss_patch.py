"""Runtime patch for differentiable contact loss in EDGE training.

V11.1 logic-gap fix:
- Adds EDGE_DCL_CONTACT_SOURCE=auto|target_channels|target_fk|pred_fk_height|pred_fk_height_speed|hybrid.
- In auto mode, overly dense target contact channels fall back to FK/height contact.
- Keeps using existing --foot_loss_weight, so no train.py loss plumbing changes are needed.
"""
from __future__ import annotations

import os
from functools import wraps

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
    return str(os.environ.get(name, default)).strip()


def install_differentiable_contact_loss_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion, maybe_unnormalize
        from losses.contact_loss import differentiable_contact_velocity_loss
    except Exception as exc:
        if verbose:
            print(f"⚠️ Differentiable contact loss patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_differentiable_contact_loss_patch_v111_installed", False):
        return True

    original_foot_sliding_loss = GaussianDiffusion._foot_sliding_loss

    @wraps(original_foot_sliding_loss)
    def patched_foot_sliding_loss(self, model_motion_x0, target_motion_x0=None):
        if not _env_bool("EDGE_DIFF_CONTACT_LOSS", False):
            return original_foot_sliding_loss(self, model_motion_x0, target_motion_x0)

        if model_motion_x0.shape[-1] != 151:
            return model_motion_x0.new_tensor(0.0)

        try:
            pred_physical = maybe_unnormalize(self.normalizer, model_motion_x0)
            target_physical = (
                maybe_unnormalize(self.normalizer, target_motion_x0)
                if target_motion_x0 is not None
                else None
            )

            pred_joints = self._fk_positions(pred_physical)
            target_joints = None

            contact_source = _env_str("EDGE_DCL_CONTACT_SOURCE", "auto").lower()
            needs_target_fk = contact_source in {"target_fk", "target_fk_height", "target_fk_height_speed", "target_fk_strict"} or _env_bool("EDGE_DCL_USE_FK_CONTACT_LABELS", False)
            if target_physical is not None and needs_target_fk:
                with torch.no_grad():
                    target_joints = self._fk_positions(target_physical)

            loss, debug = differentiable_contact_velocity_loss(
                pred_physical=pred_physical,
                target_physical=target_physical,
                pred_joints=pred_joints,
                target_joints=target_joints,
                fps=float(1.0 / max(float(getattr(self, "dt", 1.0 / 30.0)), 1e-8)),
                contact_threshold=_env_float("EDGE_DCL_CONTACT_THRESHOLD", 0.5),
                use_fk_contact_labels=_env_bool("EDGE_DCL_USE_FK_CONTACT_LABELS", False),
                height_threshold=_env_float("EDGE_DCL_HEIGHT_THRESHOLD", 0.035),
                speed_threshold=_env_float("EDGE_DCL_SPEED_THRESHOLD", 0.08),
                horizontal_only=_env_bool("EDGE_DCL_HORIZONTAL_ONLY", True),
                contact_source=contact_source,
            )

            self._edge_last_dcl_debug = debug
            if _env_bool("EDGE_DCL_VERBOSE", False):
                print(f"🦶 Differentiable contact loss: {float(loss.detach().cpu().item()):.8f}, {debug}")
            return loss

        except Exception as exc:
            if _env_bool("EDGE_DCL_STRICT", False):
                raise
            if _env_bool("EDGE_DCL_VERBOSE", False):
                print(f"⚠️ Differentiable contact loss fallback to original: {exc}")
            return original_foot_sliding_loss(self, model_motion_x0, target_motion_x0)

    GaussianDiffusion._foot_sliding_loss = patched_foot_sliding_loss
    GaussianDiffusion._edge_differentiable_contact_loss_patch_installed = True
    GaussianDiffusion._edge_differentiable_contact_loss_patch_v111_installed = True

    if verbose:
        print(
            "✅ Installed differentiable contact loss patch v11.1: "
            f"enabled={_env_bool('EDGE_DIFF_CONTACT_LOSS', False)}, "
            f"contact_source={_env_str('EDGE_DCL_CONTACT_SOURCE', 'auto')}, "
            "uses existing --foot_loss_weight."
        )
    return True


def install():
    return install_differentiable_contact_loss_patch(verbose=True)
