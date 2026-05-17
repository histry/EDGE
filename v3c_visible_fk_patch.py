
from __future__ import annotations

import os
import math
from typing import Any, Dict

import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}
_DCT_CACHE: Dict[tuple, torch.Tensor] = {}
_SMPL_CACHE: Dict[str, Any] = {}

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

def _v3_enabled() -> bool:
    return _env_bool("EDGE_V3_UNIT_RECON", False) or (
        os.environ.get("EDGE_TRAIN_PROFILE", "").strip().lower() == "v3_unit_recon"
    )

def _visible_fk_enabled() -> bool:
    return _v3_enabled() and _env_bool("EDGE_V3C_VISIBLE_FK", False)

def _maybe_unnormalize_tensor(normalizer: Any, x: torch.Tensor) -> torch.Tensor:
    if normalizer is None or not hasattr(normalizer, "mean") or not hasattr(normalizer, "std"):
        return x
    mean = torch.as_tensor(normalizer.mean, device=x.device, dtype=x.dtype)
    std = torch.as_tensor(normalizer.std, device=x.device, dtype=x.dtype)
    while mean.ndim < x.ndim:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return x * std + mean

def _get_smpl(device: torch.device):
    key = str(device)
    smpl = _SMPL_CACHE.get(key)
    if smpl is None:
        from vis import SMPLSkeleton
        smpl = SMPLSkeleton(device=device)
        _SMPL_CACHE[key] = smpl
    return smpl

def _fk_positions_151(motion_151: torch.Tensor) -> torch.Tensor:
    if motion_151.ndim != 3 or motion_151.shape[-1] != 151:
        raise ValueError(f"expected [B,T,151], got {tuple(motion_151.shape)}")
    from dataset.quaternion import ax_from_6v
    b, t, _ = motion_151.shape
    pos = motion_151[:, :, 4:7]
    q6 = motion_151[:, :, 7:].reshape(b, t, 24, 6)
    qax = ax_from_6v(q6)
    smpl = _get_smpl(motion_151.device)
    return smpl.forward(qax, pos)

def _dct_basis(length: int, keep: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    keep = max(1, min(int(keep), int(length)))
    key = (int(length), int(keep), str(device), str(dtype))
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached
    n = torch.arange(length, device=device, dtype=dtype).view(length, 1)
    k = torch.arange(keep, device=device, dtype=dtype).view(1, keep)
    basis = torch.cos(math.pi / float(length) * (n + 0.5) * k)
    basis[:, 0] *= math.sqrt(1.0 / float(length))
    if keep > 1:
        basis[:, 1:] *= math.sqrt(2.0 / float(length))
    _DCT_CACHE[key] = basis
    return basis

def _dct_lowpass(x: torch.Tensor, keep: int) -> torch.Tensor:
    if x.ndim != 3 or x.shape[1] <= 1:
        return x
    basis = _dct_basis(x.shape[1], keep, x.device, x.dtype)
    coeff = torch.einsum("btc,tk->bkc", x, basis)
    return torch.einsum("bkc,tk->btc", coeff, basis)

def _upper_safe_rot_features(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 151:
        return x
    joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    dims = []
    for j in joints:
        start = 7 + 6 * j
        dims.extend(range(start, min(start + 6, 151)))
    return x[..., dims]

def _optional_upper_rot_loss(pred_phys: torch.Tensor, target_phys: torch.Tensor) -> torch.Tensor:
    if not _env_bool("EDGE_V3C_UPPER_ROT_LOSS", False):
        return pred_phys.new_tensor(0.0)
    pred = _upper_safe_rot_features(pred_phys.float())
    target = _upper_safe_rot_features(target_phys.float())
    loss = _env_float("EDGE_V3C_UPPER_ROT_WEIGHT", 0.05) * F.mse_loss(pred, target)
    loss = loss + _env_float("EDGE_V3C_UPPER_ROT_VEL_WEIGHT", 0.10) * F.mse_loss(
        pred[:, 1:] - pred[:, :-1],
        target[:, 1:] - target[:, :-1],
    )
    dct_w = _env_float("EDGE_V3C_UPPER_ROT_DCT_WEIGHT", 0.0)
    if dct_w > 0:
        keep = _env_int("EDGE_V3C_UPPER_ROT_DCT_KEEP", 8)
        loss = loss + dct_w * F.mse_loss(_dct_lowpass(pred, keep), _dct_lowpass(target, keep))
    return loss

def _visible_upper_fk_loss(self, pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
    if pred_x0.ndim != 3 or pred_x0.shape[-1] != 151 or pred_x0.shape[1] < 3:
        return pred_x0.new_tensor(0.0)

    pred_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), pred_x0.float())
    target_phys = _maybe_unnormalize_tensor(getattr(self, "normalizer", None), target_x0.float())

    pred_pos = _fk_positions_151(pred_phys)
    target_pos = _fk_positions_151(target_phys)

    joint_mode = os.environ.get("EDGE_V3C_VISIBLE_JOINTS", "upper_safe_plus").strip().lower()
    if joint_mode in {"arms", "hands"}:
        upper_joints = [16, 17, 18, 19, 20, 21, 22, 23]
    else:
        upper_joints = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    pred_u = pred_pos[:, :, upper_joints, :]
    target_u = target_pos[:, :, upper_joints, :]

    pred_v = torch.linalg.norm(pred_u[:, 1:] - pred_u[:, :-1], dim=-1)
    target_v = torch.linalg.norm(target_u[:, 1:] - target_u[:, :-1], dim=-1)

    speed_loss = F.mse_loss(pred_v, target_v)

    pred_env = pred_v.mean(dim=-1)
    target_env = target_v.mean(dim=-1)
    pred_env_n = pred_env / pred_env.amax(dim=1, keepdim=True).clamp_min(1e-6)
    target_env_n = target_env / target_env.amax(dim=1, keepdim=True).clamp_min(1e-6)
    envelope_loss = F.mse_loss(pred_env_n, target_env_n)

    pred_activity = pred_v.mean(dim=(1, 2))
    target_activity = target_v.mean(dim=(1, 2))
    activity_floor = _env_float("EDGE_V3C_ACTIVITY_FLOOR", 0.70)
    activity_floor_loss = torch.relu(activity_floor * target_activity.detach() - pred_activity).pow(2).mean()

    pred_range = torch.linalg.norm(pred_u.amax(dim=1) - pred_u.amin(dim=1), dim=-1).mean(dim=1)
    target_range = torch.linalg.norm(target_u.amax(dim=1) - target_u.amin(dim=1), dim=-1).mean(dim=1)
    range_floor = _env_float("EDGE_V3C_RANGE_FLOOR", 0.65)
    range_floor_loss = torch.relu(range_floor * target_range.detach() - pred_range).pow(2).mean()

    hand_joints = [20, 21, 22, 23]
    pred_h = pred_pos[:, :, hand_joints, :]
    target_h = target_pos[:, :, hand_joints, :]
    pred_hv = torch.linalg.norm(pred_h[:, 1:] - pred_h[:, :-1], dim=-1)
    target_hv = torch.linalg.norm(target_h[:, 1:] - target_h[:, :-1], dim=-1)
    hand_speed_loss = F.mse_loss(pred_hv, target_hv)
    pred_hrange = torch.linalg.norm(pred_h.amax(dim=1) - pred_h.amin(dim=1), dim=-1).mean(dim=1)
    target_hrange = torch.linalg.norm(target_h.amax(dim=1) - target_h.amin(dim=1), dim=-1).mean(dim=1)
    hand_range_floor_loss = torch.relu(range_floor * target_hrange.detach() - pred_hrange).pow(2).mean()

    loss = (
        _env_float("EDGE_V3C_FK_SPEED_WEIGHT", 1.00) * speed_loss
        + _env_float("EDGE_V3C_FK_ENVELOPE_WEIGHT", 0.50) * envelope_loss
        + _env_float("EDGE_V3C_FK_ACTIVITY_WEIGHT", 3.00) * activity_floor_loss
        + _env_float("EDGE_V3C_FK_RANGE_WEIGHT", 2.00) * range_floor_loss
        + _env_float("EDGE_V3C_FK_HAND_WEIGHT", 1.00) * hand_speed_loss
        + _env_float("EDGE_V3C_FK_HAND_RANGE_WEIGHT", 2.00) * hand_range_floor_loss
    )
    loss = loss + _optional_upper_rot_loss(pred_phys, target_phys)

    if _env_bool("EDGE_V3C_VISIBLE_FK_DEBUG", False):
        if not hasattr(self, "_edge_v3c_visible_debug_counter"):
            self._edge_v3c_visible_debug_counter = 0
        self._edge_v3c_visible_debug_counter += 1
        if self._edge_v3c_visible_debug_counter <= 20 or self._edge_v3c_visible_debug_counter % 100 == 0:
            print(
                "🧪 V3C visible-FK | "
                f"pred_act={float(pred_activity.mean().detach().cpu()):.6f} "
                f"target_act={float(target_activity.mean().detach().cpu()):.6f} "
                f"pred_range={float(pred_range.mean().detach().cpu()):.6f} "
                f"target_range={float(target_range.mean().detach().cpu()):.6f} "
                f"pred_hand_range={float(pred_hrange.mean().detach().cpu()):.6f} "
                f"target_hand_range={float(target_hrange.mean().detach().cpu()):.6f} "
                f"speed={float(speed_loss.detach().cpu()):.6f} "
                f"env={float(envelope_loss.detach().cpu()):.6f} "
                f"act_floor={float(activity_floor_loss.detach().cpu()):.6f} "
                f"range_floor={float(range_floor_loss.detach().cpu()):.6f}",
                flush=True,
            )

    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

def install_v3c_visible_fk_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3C visible-FK patch skipped: cannot import GaussianDiffusion: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3c_visible_fk_patched", False):
        if verbose:
            print("✅ V3C visible-FK patch already installed")
        return True

    orig_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _visible_fk_enabled():
            return orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        fk_loss = _visible_upper_fk_loss(self, model_motion_x0, target_motion_x0)
        final_w = _env_float("EDGE_V3C_VISIBLE_FK_WEIGHT", 10.0)
        return (final_w * fk_loss) / 0.05

    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion._edge_v3c_visible_fk_patched = True

    if verbose:
        print("✅ Installed V3C FK-visible upper-body motion patch")
    return True

install_v3_visible_motion_patch = install_v3c_visible_fk_patch
install_visible_fk_patch = install_v3c_visible_fk_patch
