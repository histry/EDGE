"""
V3 Temporal Unit Reconstruction patch for EDGE-Dunhuang.

Purpose
-------
Switch training from sparse keyframe / endpoint-driven control to clean
full 45-frame motion-unit reconstruction.

This patch is intentionally conservative:
- it does not change model architecture;
- it disables accidental keyframe / trajectory / audio / RAG conditions when
  EDGE_V3_UNIT_RECON=1;
- it reuses the existing EDGE_X0_RECON_LOSS path in model/diffusion.py;
- it adds an optional DCT low-frequency temporal reconstruction term through
  GaussianDiffusion._motion_energy_loss, which is already called by p_losses.

Usage
-----
Set before running train.py:

    export EDGE_TRAIN_PROFILE=v3_unit_recon
    export EDGE_V3_UNIT_RECON=1
    export EDGE_X0_RECON_LOSS=1
    export EDGE_X0_RECON_LOSS_WEIGHT=0.8

Recommended for V3:
    --keyframe_condition_prob 0.0
    --keyframe_loss_weight 0.0
    --mid_keyframe_condition_prob 0.0
    --mid_keyframe_count 0
    --disable_traj_cond
    --trajectory_loss_weight 0.0
    --trajectory_velocity_loss_weight 0.0
    --audio_pairing_mode none
    --mmr_loss_weight 0.0
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


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


def _v3_enabled() -> bool:
    return _env_bool("EDGE_V3_UNIT_RECON", False) or (
        os.environ.get("EDGE_TRAIN_PROFILE", "").strip().lower() == "v3_unit_recon"
    )


def _clone_cond_without_controls(cond: Any, x_start: torch.Tensor) -> Any:
    """Return a condition dict with non-reconstruction controls removed.

    We keep an audio tensor only as zeros because DanceDecoder accepts missing
    audio but many dataset paths already provide audio-shaped conditions. A zero
    tensor makes the contract explicit and avoids accidental weak/proxy music
    conditioning during V3.
    """
    if not isinstance(cond, dict):
        if torch.is_tensor(cond):
            return {"audio": torch.zeros_like(cond)}
        return cond

    cleaned: Dict[str, Any] = {}

    audio = cond.get("audio", None)
    if torch.is_tensor(audio):
        cleaned["audio"] = torch.zeros_like(audio)
    # Keep bookkeeping fields only if they are harmless.
    audio_paired = cond.get("audio_paired", None)
    if torch.is_tensor(audio_paired):
        cleaned["audio_paired"] = torch.zeros_like(audio_paired)

    # Explicitly drop these controls for V3.
    # trajectory: would turn V3 into route following
    # energy/onset/beat: would let model solve envelope matching instead of unit reconstruction
    # rag/text/unit prior: belongs to V4 after the temporal prior is learned
    for key in (
        "trajectory",
        "trajectory_event",
        "energy",
        "onset",
        "beat",
        "rag_summary",
        "rag_context",
        "text_context",
        "unit_prior",
        "unit_prior_motion",
        "unit_prior_mask",
    ):
        if key in cleaned:
            cleaned.pop(key, None)

    return cleaned


_DCT_CACHE: Dict[tuple, torch.Tensor] = {}


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
    """Low-pass reconstruct x [B,T,C] with first keep DCT coefficients."""
    if x.ndim != 3 or x.shape[1] <= 1:
        return x
    basis = _dct_basis(x.shape[1], keep, x.device, x.dtype)  # [T,K]
    coeff = torch.einsum("btc,tk->bkc", x, basis)
    recon = torch.einsum("bkc,tk->btc", coeff, basis)
    return recon


def _feature_select(x: torch.Tensor) -> torch.Tensor:
    """Select reconstruction features for the temporal prior.

    Default: all non-contact channels, i.e. root xyz + rotations.
    Options:
      EDGE_V3_TEMPORAL_FEATURES=all
      EDGE_V3_TEMPORAL_FEATURES=no_contact
      EDGE_V3_TEMPORAL_FEATURES=rot
      EDGE_V3_TEMPORAL_FEATURES=upper_torso
      EDGE_V3_TEMPORAL_FEATURES=rootxz
    """
    mode = os.environ.get("EDGE_V3_TEMPORAL_FEATURES", "no_contact").strip().lower()
    if x.shape[-1] != 151:
        return x

    if mode == "all":
        return x
    if mode == "rootxz":
        return x[..., [4, 6]]
    if mode == "rot":
        return x[..., 7:151]
    if mode in {"upper", "upper_torso", "torso_upper"}:
        # Conservative SMPL-ish set: root/spine/chest/neck/head + shoulders/arms/hands.
        # 151D layout: rotations start at dim 7, each joint has 6D.
        joints = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        dims = []
        for j in joints:
            start = 7 + 6 * j
            dims.extend(range(start, min(start + 6, 151)))
        return x[..., dims]
    # no_contact
    return x[..., 4:151]


def _temporal_unit_loss(pred_x0: torch.Tensor, target_x0: torch.Tensor) -> torch.Tensor:
    """Low-frequency full-unit reconstruction loss.

    This is not a local jerk/progress rule. It asks the predicted clean x0 to
    match the low-frequency temporal structure of the whole GT unit.
    """
    if pred_x0.shape[1] <= 1:
        return pred_x0.new_tensor(0.0)

    keep = _env_int("EDGE_V3_DCT_KEEP", 6)
    pred = _feature_select(pred_x0.float())
    target = _feature_select(target_x0.float())

    pred_lp = _dct_lowpass(pred, keep)
    target_lp = _dct_lowpass(target, keep)
    lowfreq = F.mse_loss(pred_lp, target_lp)

    vel_w = _env_float("EDGE_V3_VELOCITY_WEIGHT", 0.05)
    acc_w = _env_float("EDGE_V3_ACCEL_WEIGHT", 0.01)

    vel = F.mse_loss(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1])

    if pred.shape[1] >= 3:
        pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
        target_acc = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        acc = F.mse_loss(pred_acc, target_acc)
    else:
        acc = pred_x0.new_tensor(0.0)

    return lowfreq + vel_w * vel + acc_w * acc


def install_v3_unit_reconstruction_patch(verbose: bool = True):
    """Install V3 unit-reconstruction behavior on GaussianDiffusion."""
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V3 unit reconstruction patch skipped: cannot import GaussianDiffusion: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_v3_unit_recon_patched", False):
        if verbose:
            print("✅ V3 Temporal Unit Reconstruction patch already installed")
        return True

    orig_init = GaussianDiffusion.__init__
    orig_p_losses = GaussianDiffusion.p_losses
    orig_motion_energy_loss = GaussianDiffusion._motion_energy_loss

    def patched_init(self, *args, **kwargs):
        if _v3_enabled():
            # Fail-safe overrides even if an old shell script forgot to pass zeros.
            kwargs["keyframe_condition_prob"] = 0.0
            kwargs["keyframe_loss_weight"] = 0.0
            kwargs["mid_keyframe_condition_prob"] = 0.0
            kwargs["mid_keyframe_count"] = 0
            kwargs["trajectory_loss_weight"] = 0.0
            kwargs["trajectory_velocity_loss_weight"] = 0.0
            kwargs["beat_guidance_weight"] = 0.0
            kwargs["sync_loss_weight"] = 0.0
            kwargs["energy_loss_weight"] = 0.0
            kwargs["root_lower_coupling_loss_weight"] = 0.0
            kwargs["hard_keyframe_project"] = False
        orig_init(self, *args, **kwargs)
        if _v3_enabled():
            self.keyframe_condition_prob = 0.0
            self.keyframe_loss_weight = 0.0
            self.mid_keyframe_condition_prob = 0.0
            self.mid_keyframe_count = 0
            self.trajectory_loss_weight = 0.0
            self.trajectory_velocity_loss_weight = 0.0
            self.beat_guidance_weight = 0.0
            self.sync_loss_weight = 0.0
            self.energy_loss_weight = 0.0
            self.root_lower_coupling_loss_weight = 0.0
            self.hard_keyframe_project = False
            # Force zero/proxy audio dropout in V3. The condition tensor is also zeroed below.
            self.force_audio_only_drop = True
            if verbose:
                print(
                    "✅ V3 unit recon guards active: keyframe/mid/trajectory/beat/"
                    "RAG-style controls must not drive this run."
                )

    def patched_p_losses(self, x_start, cond, t, noise=None, current_epoch=None, constraint=None):
        if _v3_enabled():
            cond = _clone_cond_without_controls(cond, x_start)
            constraint = None
            # Keep runtime attributes safe even if another patch mutates them.
            self.keyframe_condition_prob = 0.0
            self.keyframe_loss_weight = 0.0
            self.mid_keyframe_condition_prob = 0.0
            self.mid_keyframe_count = 0
            self.trajectory_loss_weight = 0.0
            self.trajectory_velocity_loss_weight = 0.0
            self.beat_guidance_weight = 0.0
            self.sync_loss_weight = 0.0
            self.root_lower_coupling_loss_weight = 0.0
            self.hard_keyframe_project = False
        return orig_p_losses(self, x_start, cond, t, noise=noise, current_epoch=current_epoch, constraint=constraint)

    def patched_motion_energy_loss(self, model_motion_x0, target_motion_x0):
        if not _v3_enabled():
            return orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)

        # Existing p_losses multiplies this return value by physical_w * 0.05.
        # Divide by 0.05 so EDGE_V3_TEMPORAL_WEIGHT has the intuitive final scale
        # after warmup: physical_w * EDGE_V3_TEMPORAL_WEIGHT * temporal_loss.
        temporal_w = _env_float("EDGE_V3_TEMPORAL_WEIGHT", 0.20)
        envelope_w = _env_float("EDGE_V3_ENERGY_ENVELOPE_WEIGHT", 0.00)
        temporal = _temporal_unit_loss(model_motion_x0, target_motion_x0)
        envelope = orig_motion_energy_loss(self, model_motion_x0, target_motion_x0)
        return (temporal_w * temporal + envelope_w * envelope) / 0.05

    GaussianDiffusion.__init__ = patched_init
    GaussianDiffusion.p_losses = patched_p_losses
    GaussianDiffusion._motion_energy_loss = patched_motion_energy_loss
    GaussianDiffusion._edge_v3_unit_recon_patched = True

    if verbose:
        print("✅ Installed V3 Temporal Unit Reconstruction patch")
    return True


# Backward-compatible alias for possible copy-paste use.
install_unit_reconstruction_patch = install_v3_unit_reconstruction_patch
