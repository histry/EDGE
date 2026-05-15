from __future__ import annotations
import os
import torch

_TRUE = {"1", "true", "yes", "y", "on"}

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return str(v).strip().lower() in _TRUE

def _expand_mask(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] == 1:
        return mask.expand_as(value)
    if mask.shape[-1] == value.shape[-1]:
        return mask
    raise ValueError(f"constraint mask last dim must be 1 or {value.shape[-1]}, got {mask.shape[-1]}")

def install_recon_contract_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ EDGE recon contract patch not installed: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_recon_contract_patch_installed", False):
        if verbose:
            print("✅ EDGE recon contract patch already installed.")
        return True

    original_model_predictions = GaussianDiffusion.model_predictions

    def _edge_should_project_xstart(self):
        return (
            bool(getattr(self, "hard_keyframe_project", False))
            or _env_bool("EDGE_HARD_KEYFRAME_PROJECT", False)
            or _env_bool("EDGE_INFER_PROJECT_XSTART", False)
        )

    def _edge_project_clean_xstart(self, x_start, constraint):
        if constraint is None or not _edge_should_project_xstart(self):
            return x_start
        mask = constraint.get("mask", None)
        value = constraint.get("value", None)
        if mask is None or value is None:
            return x_start
        mask = mask.to(device=x_start.device, dtype=x_start.dtype)
        value = value.to(device=x_start.device, dtype=x_start.dtype)
        feature_mask = _expand_mask(mask, x_start)
        return x_start * (1.0 - feature_mask) + value * feature_mask

    def model_predictions_patched(self, x, cond, t, weight=None, clip_x_start=False, constraint=None):
        pred_noise, x_start = original_model_predictions(
            self, x, cond, t, weight=weight, clip_x_start=clip_x_start, constraint=constraint
        )
        if constraint is not None and _edge_should_project_xstart(self):
            x_start = self._edge_project_clean_xstart(x_start, constraint)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    GaussianDiffusion._edge_project_clean_xstart = _edge_project_clean_xstart
    GaussianDiffusion.model_predictions = model_predictions_patched
    GaussianDiffusion._edge_recon_contract_patch_installed = True

    if verbose:
        print("✅ Installed EDGE reconstruction-contract patch.")
    return True
