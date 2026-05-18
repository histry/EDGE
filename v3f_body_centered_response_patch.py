from __future__ import annotations

import os
import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}
_SMPL_CACHE = {}

def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)

def _enabled():
    return _env_bool("EDGE_V3F_BODY_CENTERED", False)

def _maybe_unnormalize(normalizer, x):
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        return x
    mean = torch.as_tensor(normalizer.mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(normalizer.std, device=x.device, dtype=x.dtype)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return x * std + mean

def _get_smpl(device):
    key = str(device)
    if key not in _SMPL_CACHE:
        from vis import SMPLSkeleton
        _SMPL_CACHE[key] = SMPLSkeleton(device=device)
    return _SMPL_CACHE[key]

def _fk_positions_151(motion_151):
    if motion_151.ndim != 3 or motion_151.shape[-1] != 151:
        raise ValueError(f"expected [B,T,151], got {tuple(motion_151.shape)}")
    from dataset.quaternion import ax_from_6v
    b, t, _ = motion_151.shape
    pos = motion_151[:, :, 4:7]
    q6 = motion_151[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    return _get_smpl(motion_151.device).forward(qax, pos)

def _norm01(x, eps=1e-6):
    x = x - x.amin(dim=1, keepdim=True)
    return x / x.amax(dim=1, keepdim=True).clamp_min(eps)

def _safe_corr_loss(a, b):
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-6)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-6)
    corr = (a * b).sum(dim=1)
    return (1.0 - corr).mean()

def _body_centered_response_loss(self, pred_x0, target_x0):
    if pred_x0.ndim != 3 or pred_x0.shape[-1] != 151 or pred_x0.shape[1] < 3:
        return pred_x0.new_tensor(0.0)

    pred_phys = _maybe_unnormalize(getattr(self, "normalizer", None), pred_x0.float())
    target_phys = _maybe_unnormalize(getattr(self, "normalizer", None), target_x0.float())

    pred_pos = _fk_positions_151(pred_phys)
    target_pos = _fk_positions_151(target_phys)

    # Body-centered FK: remove global/root translation.
    pred_bc = pred_pos - pred_pos[:, :, 0:1, :]
    target_bc = target_pos - target_pos[:, :, 0:1, :]

    torso_joints = [3, 6, 9, 12, 15]
    arms_joints = [16, 17, 18, 19, 20, 21, 22, 23]

    pred_torso = pred_bc[:, :, torso_joints, :]
    target_torso = target_bc[:, :, torso_joints, :]
    pred_arms = pred_bc[:, :, arms_joints, :]
    target_arms = target_bc[:, :, arms_joints, :]

    # Torso range floor.
    pred_torso_range = torch.linalg.norm(
        pred_torso.amax(dim=1) - pred_torso.amin(dim=1), dim=-1
    ).mean(dim=1)
    target_torso_range = torch.linalg.norm(
        target_torso.amax(dim=1) - target_torso.amin(dim=1), dim=-1
    ).mean(dim=1)

    torso_floor = _env_float("EDGE_V3F_TORSO_RANGE_FLOOR", 0.55)
    torso_range_loss = torch.relu(
        torso_floor * target_torso_range.detach() - pred_torso_range
    ).pow(2).mean()

    # Torso temporal envelope.
    pred_torso_v = torch.linalg.norm(pred_torso[:, 1:] - pred_torso[:, :-1], dim=-1).mean(dim=-1)
    target_torso_v = torch.linalg.norm(target_torso[:, 1:] - target_torso[:, :-1], dim=-1).mean(dim=-1)

    pred_arm_v = torch.linalg.norm(pred_arms[:, 1:] - pred_arms[:, :-1], dim=-1).mean(dim=-1)
    target_arm_v = torch.linalg.norm(target_arms[:, 1:] - target_arms[:, :-1], dim=-1).mean(dim=-1)

    torso_env_loss = F.mse_loss(_norm01(pred_torso_v), _norm01(target_torso_v))
    arm_env_loss = F.mse_loss(_norm01(pred_arm_v), _norm01(target_arm_v))
    response_loss = _safe_corr_loss(pred_torso_v, pred_arm_v)

    # Root-stable: prevent global root drift from becoming the motion source.
    root_xz = pred_phys[:, :, [4, 6]]
    root_v = torch.linalg.norm(root_xz[:, 1:] - root_xz[:, :-1], dim=-1)
    root_stable_loss = root_v.pow(2).mean()

    loss = (
        _env_float("EDGE_V3F_TORSO_RANGE_WEIGHT", 8.0) * torso_range_loss
        + _env_float("EDGE_V3F_TORSO_ENV_WEIGHT", 4.0) * torso_env_loss
        + _env_float("EDGE_V3F_RESPONSE_WEIGHT", 2.0) * response_loss
        + _env_float("EDGE_V3F_ARM_ENV_WEIGHT", 1.0) * arm_env_loss
        + _env_float("EDGE_V3F_ROOT_STABLE_WEIGHT", 0.5) * root_stable_loss
    )

    if _env_bool("EDGE_V3F_DEBUG", False):
        if not hasattr(self, "_edge_v3f_debug_counter"):
            self._edge_v3f_debug_counter = 0
        self._edge_v3f_debug_counter += 1
        if self._edge_v3f_debug_counter <= 20 or self._edge_v3f_debug_counter % 100 == 0:
            print(
                "🧪 V3F body-centered | "
                f"torso_range={float(pred_torso_range.mean().detach().cpu()):.6f} "
                f"target_torso_range={float(target_torso_range.mean().detach().cpu()):.6f} "
                f"torso_range_loss={float(torso_range_loss.detach().cpu()):.6f} "
                f"torso_env={float(torso_env_loss.detach().cpu()):.6f} "
                f"response={float(response_loss.detach().cpu()):.6f} "
                f"arm_env={float(arm_env_loss.detach().cpu()):.6f} "
                f"root_stable={float(root_stable_loss.detach().cpu()):.6f}",
                flush=True,
            )

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

def install_v3f_body_centered_response_patch(verbose=True):
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3F patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3f_body_centered_patched", False):
        if verbose:
            print("✅ V3F body-centered response patch already installed")
        return True

    orig_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        base = orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        if not _enabled():
            return base
        return base + _env_float("EDGE_V3F_WEIGHT", 1.0) * _body_centered_response_loss(
            self, model_motion_x0, target_motion_x0
        )

    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion._edge_v3f_body_centered_patched = True

    if verbose:
        print("✅ Installed V3F body-centered torso-response patch")
    return True
