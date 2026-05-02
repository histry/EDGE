"""Native trajectory-control patches for EDGE.

Drop this file in the EDGE project root together with ``sitecustomize.py``.
Python imports ``sitecustomize`` automatically when running from the project
root, so the patch is installed before EDGE/GaussianDiffusion are constructed.

What this patch adds without rewriting the whole repository:
1) trajectory-specific classifier-free guidance (CFG) at inference;
2) time-dependent trajectory loss in training;
3) relative-velocity / acceleration / endpoint trajectory supervision.

Environment variables
---------------------
Inference:
    EDGE_TRAJ_GUIDANCE_WEIGHT=1.8        # 1.0 disables trajectory-specific CFG
    EDGE_TRAJ_GUIDANCE_START_FRAC=0.15   # active for t_frac >= start and <= end
    EDGE_TRAJ_GUIDANCE_END_FRAC=1.00
    EDGE_TRAJ_GUIDANCE_POWER=1.0         # larger = more early-noise emphasis

Training:
    EDGE_TRAJ_LOSS_EARLY_BOOST=3.0       # high-t macro trajectory weight boost
    EDGE_TRAJ_LOSS_POWER=1.0
    EDGE_TRAJ_ENDPOINT_WEIGHT=2.0
    EDGE_TRAJ_ACC_WEIGHT=0.10

The patch is intentionally conservative: if the current model does not expose
``keep_audio_mask`` / ``keep_traj_mask`` in its forward signature, it falls back
to the original model_predictions implementation.
"""
from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    text = os.environ.get(name, None)
    if text is None:
        return bool(default)
    return str(text).strip().lower() not in {"0", "false", "no", "off", ""}


def _has_kwarg(model: Any, name: str) -> bool:
    """Return True if model.forward accepts the requested keyword."""
    target = model.module if hasattr(model, "module") else model
    try:
        sig = inspect.signature(target.forward)
    except Exception:
        return False
    return name in sig.parameters


def _to_device_bool(shape, device, value: bool) -> torch.Tensor:
    return torch.full(shape, bool(value), dtype=torch.bool, device=device)


def _extract_force_from_constraint(constraint: Optional[Dict[str, torch.Tensor]]):
    if constraint is None:
        return None, None
    return constraint.get("mask", None), constraint.get("value", None)


def _call_model_with_keep_masks(
    model,
    x: torch.Tensor,
    cond,
    t: torch.Tensor,
    *,
    keep_audio: bool,
    keep_traj: bool,
    force_mask=None,
    force_x_clean=None,
):
    b = x.shape[0]
    keep_audio_mask = _to_device_bool((b,), x.device, keep_audio)
    keep_traj_mask = _to_device_bool((b,), x.device, keep_traj)
    return model(
        x,
        cond,
        t,
        cond_drop_prob=0.0,
        force_mask=force_mask,
        force_x_clean=force_x_clean,
        keep_audio_mask=keep_audio_mask,
        keep_traj_mask=keep_traj_mask,
    )


def _trajectory_cfg_scale(diffusion, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Per-sample trajectory CFG scale.

    DDPM uses high t at early noisy steps and low t near the clean final steps.
    We emphasize high/mid t so the model first forms the global route, then let
    late denoising focus on local pose details.
    """
    base = float(getattr(diffusion, "trajectory_guidance_weight", 1.0))
    if base <= 1.0:
        return torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype)

    n_timestep = max(int(getattr(diffusion, "n_timestep", 1000)) - 1, 1)
    t_frac = (t.float() / float(n_timestep)).to(device=x.device, dtype=x.dtype)
    start = float(getattr(diffusion, "trajectory_guidance_start_frac", 0.0))
    end = float(getattr(diffusion, "trajectory_guidance_end_frac", 1.0))
    power = max(float(getattr(diffusion, "trajectory_guidance_power", 1.0)), 1e-6)

    active = ((t_frac >= start) & (t_frac <= end)).to(dtype=x.dtype)
    early = t_frac.clamp(0.0, 1.0).pow(power)
    scale = 1.0 + (base - 1.0) * active * early
    return scale.view(-1, 1, 1)


def _target_traj_like(model_motion_x0: torch.Tensor, cond) -> Optional[torch.Tensor]:
    if not isinstance(cond, dict) or cond.get("trajectory", None) is None:
        return None
    target_traj = cond["trajectory"].to(
        device=model_motion_x0.device,
        dtype=model_motion_x0.dtype,
    )
    if target_traj.shape[1] != model_motion_x0.shape[1]:
        target_traj = F.interpolate(
            target_traj.transpose(1, 2),
            size=model_motion_x0.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return target_traj[..., :2]


def _pred_root_xz(diffusion, model_motion_x0: torch.Tensor) -> torch.Tensor:
    if model_motion_x0.shape[-1] == 151:
        return model_motion_x0[:, :, [diffusion.root_x_idx, diffusion.root_z_idx]]
    # Fallback for other representation variants: assume xyz starts at 0.
    return model_motion_x0[:, :, [0, 2]]


def install_native_trajectory_control_patch(verbose: Optional[bool] = None):
    """Patch model.diffusion.GaussianDiffusion once."""
    if verbose is None:
        verbose = _env_bool("EDGE_TRAJ_PATCH_VERBOSE", True)

    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:  # pragma: no cover
        if verbose:
            print(f"⚠️ native trajectory patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_native_trajectory_patch_installed", False):
        return True

    original_init = GaussianDiffusion.__init__
    original_model_predictions = GaussianDiffusion.model_predictions
    original_trajectory_training_loss = GaussianDiffusion._trajectory_training_loss

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        # Inference CFG. 1.0 keeps old behavior exactly.
        self.trajectory_guidance_weight = _env_float("EDGE_TRAJ_GUIDANCE_WEIGHT", 1.0)
        self.trajectory_guidance_start_frac = _env_float("EDGE_TRAJ_GUIDANCE_START_FRAC", 0.0)
        self.trajectory_guidance_end_frac = _env_float("EDGE_TRAJ_GUIDANCE_END_FRAC", 1.0)
        self.trajectory_guidance_power = _env_float("EDGE_TRAJ_GUIDANCE_POWER", 1.0)

        # Training loss schedule.
        self.trajectory_loss_early_boost = _env_float("EDGE_TRAJ_LOSS_EARLY_BOOST", 3.0)
        self.trajectory_loss_power = _env_float("EDGE_TRAJ_LOSS_POWER", 1.0)
        self.trajectory_endpoint_loss_weight = _env_float("EDGE_TRAJ_ENDPOINT_WEIGHT", 2.0)
        self.trajectory_acc_loss_weight = _env_float("EDGE_TRAJ_ACC_WEIGHT", 0.10)

        if verbose and self.trajectory_guidance_weight > 1.0:
            print(
                "🧭 Native trajectory CFG enabled: "
                f"weight={self.trajectory_guidance_weight}, "
                f"active=[{self.trajectory_guidance_start_frac}, {self.trajectory_guidance_end_frac}], "
                f"power={self.trajectory_guidance_power}"
            )

    def patched_model_predictions(
        self,
        x,
        cond,
        t,
        weight=None,
        clip_x_start=False,
        constraint=None,
    ):
        # If there is no trajectory or guidance disabled, preserve original behavior.
        has_traj = isinstance(cond, dict) and cond.get("trajectory", None) is not None
        traj_weight = float(getattr(self, "trajectory_guidance_weight", 1.0))
        if (not has_traj) or traj_weight <= 1.0:
            return original_model_predictions(
                self,
                x,
                cond,
                t,
                weight=weight,
                clip_x_start=clip_x_start,
                constraint=constraint,
            )

        # Need the current DanceDecoder forward to support independent audio/traj masks.
        if not (_has_kwarg(self.model, "keep_audio_mask") and _has_kwarg(self.model, "keep_traj_mask")):
            return original_model_predictions(
                self,
                x,
                cond,
                t,
                weight=weight,
                clip_x_start=clip_x_start,
                constraint=constraint,
            )

        force_mask, force_x_clean = _extract_force_from_constraint(constraint)

        # Audio-only baseline preserves music/rhythm conditioning.
        audio_only = _call_model_with_keep_masks(
            self.model,
            x,
            cond,
            t,
            keep_audio=True,
            keep_traj=False,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
        )
        audio_traj = _call_model_with_keep_masks(
            self.model,
            x,
            cond,
            t,
            keep_audio=True,
            keep_traj=True,
            force_mask=force_mask,
            force_x_clean=force_x_clean,
        )

        scale = _trajectory_cfg_scale(self, t, x)
        model_output = audio_only + (audio_traj - audio_only) * scale

        if self.predict_epsilon:
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
        else:
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if clip_x_start:
            if x_start.shape[-1] > 7:
                x_start = torch.cat(
                    [x_start[..., :7], x_start[..., 7:].clamp(-1.0, 1.0)],
                    dim=-1,
                )
            else:
                x_start = x_start.clamp(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def patched_trajectory_training_loss(self, model_motion_x0, cond, t):
        """Time-dependent trajectory loss with relative velocity and endpoint terms.

        Return signature matches the original implementation: (position_loss,
        velocity_loss).  Acceleration is folded into the second returned value so
        existing training loops do not need to change.
        """
        zero = model_motion_x0.new_tensor(0.0)
        target_traj = _target_traj_like(model_motion_x0, cond)
        if target_traj is None or float(getattr(self, "trajectory_loss_weight", 0.0)) <= 0.0:
            return zero, zero

        pred_traj = _pred_root_xz(self, model_motion_x0)
        if pred_traj.shape[1] != target_traj.shape[1]:
            t_len = min(pred_traj.shape[1], target_traj.shape[1])
            pred_traj = pred_traj[:, :t_len]
            target_traj = target_traj[:, :t_len]

        # High-t steps correspond to early noisy denoising. Boost trajectory loss
        # there so the model learns macro route layout before local pose details.
        n_timestep = max(int(getattr(self, "n_timestep", 1000)) - 1, 1)
        t_frac = (t.float() / float(n_timestep)).to(device=pred_traj.device, dtype=pred_traj.dtype)
        boost = max(float(getattr(self, "trajectory_loss_early_boost", 0.0)), 0.0)
        power = max(float(getattr(self, "trajectory_loss_power", 1.0)), 1e-6)
        time_weight = (1.0 + boost * t_frac.clamp(0.0, 1.0).pow(power)).view(-1)

        pos_per = (pred_traj - target_traj).pow(2).mean(dim=(1, 2))

        endpoint_weight = max(float(getattr(self, "trajectory_endpoint_loss_weight", 0.0)), 0.0)
        if endpoint_weight > 0.0 and pred_traj.shape[1] >= 2:
            endpoint_per = (
                (pred_traj[:, 0] - target_traj[:, 0]).pow(2).mean(dim=-1)
                + (pred_traj[:, -1] - target_traj[:, -1]).pow(2).mean(dim=-1)
            ) * 0.5
            pos_per = pos_per + endpoint_weight * endpoint_per

        traj_pos = (pos_per * time_weight).mean()

        if pred_traj.shape[1] > 1:
            pred_vel = pred_traj[:, 1:] - pred_traj[:, :-1]
            target_vel = target_traj[:, 1:] - target_traj[:, :-1]
            vel_per = (pred_vel - target_vel).pow(2).mean(dim=(1, 2))
            traj_vel = (vel_per * time_weight).mean()
        else:
            traj_vel = zero

        acc_weight = max(float(getattr(self, "trajectory_acc_loss_weight", 0.0)), 0.0)
        if acc_weight > 0.0 and pred_traj.shape[1] > 2:
            pred_acc = pred_traj[:, 2:] - 2.0 * pred_traj[:, 1:-1] + pred_traj[:, :-2]
            target_acc = target_traj[:, 2:] - 2.0 * target_traj[:, 1:-1] + target_traj[:, :-2]
            acc_per = (pred_acc - target_acc).pow(2).mean(dim=(1, 2))
            traj_acc = (acc_per * time_weight).mean()
            traj_vel = traj_vel + acc_weight * traj_acc

        return traj_pos, traj_vel

    GaussianDiffusion.__init__ = patched_init
    GaussianDiffusion.model_predictions = patched_model_predictions
    GaussianDiffusion._trajectory_training_loss = patched_trajectory_training_loss
    GaussianDiffusion._native_trajectory_original_model_predictions = original_model_predictions
    GaussianDiffusion._native_trajectory_original_trajectory_training_loss = original_trajectory_training_loss
    GaussianDiffusion._native_trajectory_patch_installed = True

    if verbose:
        print("✅ Installed native trajectory control patch (CFG + time-dependent loss).")
    return True


# Optional explicit entrypoint for scripts that do not rely on sitecustomize.
def install():
    return install_native_trajectory_control_patch()
