"""Runtime safety patches for EDGE diffusion.

This file is imported by sitecustomize.py / generate_controlled.py in the
current EDGE branch.  It intentionally keeps the stage-4/5 model architecture
inside model/model.py, but installs small runtime fixes for:

1) loss warmup when validation does not pass current_epoch;
2) TTO update optimizer, replacing raw SGD with optional momentum/Adam;
3) robust default TTO attributes.

Environment knobs:
    EDGE_WARMUP_MISSING_EPOCH_POLICY=last|zero|one
    EDGE_TTO_OPTIMIZER=adam|momentum|sgd
    EDGE_TTO_MOMENTUM=0.9
    EDGE_TTO_ADAM_BETA1=0.9
    EDGE_TTO_ADAM_BETA2=0.999
    EDGE_TTO_EPS=1e-8
    EDGE_TTO_GRAD_CLIP=1.0
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip().lower()


def _extract_current_epoch(args, kwargs) -> Optional[int]:
    if "current_epoch" in kwargs and kwargs["current_epoch"] is not None:
        try:
            return int(kwargs["current_epoch"])
        except Exception:
            return None

    # Most EDGE p_losses signatures put current_epoch near the end.  We only
    # capture explicit int-like values after the usual x/cond/t prefix.
    for value in reversed(args):
        if value is None:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
    return None


def _replace_none_current_epoch(self, args, kwargs):
    """Replace a None current_epoch before calling the original warmup."""
    if "current_epoch" in kwargs:
        if kwargs["current_epoch"] is None:
            kwargs = dict(kwargs)
            kwargs["current_epoch"] = _fallback_epoch(self)
        return args, kwargs

    if not args:
        return args, kwargs

    args = list(args)
    # Common signature: _linear_warmup(current_epoch, warmup_epochs)
    if args[0] is None:
        args[0] = _fallback_epoch(self)
    return tuple(args), kwargs


def _fallback_epoch(self) -> int:
    policy = _env_str("EDGE_WARMUP_MISSING_EPOCH_POLICY", "last")
    if policy == "one":
        return 10**9  # effectively fully warmed
    if policy == "zero":
        return 0

    last_epoch = getattr(self, "_edge_last_current_epoch", None)
    if last_epoch is not None:
        return int(last_epoch)

    # Safer than returning 1.0 warmup early in training.  This prevents val loss
    # spikes when validation forgets to pass current_epoch.
    return 0


def _trajectory_target(cond, x_start):
    if not isinstance(cond, dict) or cond.get("trajectory", None) is None:
        return None
    target = cond["trajectory"].to(device=x_start.device, dtype=x_start.dtype)
    if target.shape[1] != x_start.shape[1]:
        target = F.interpolate(
            target.transpose(1, 2),
            size=x_start.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
    return target[..., :2]


def _pred_root_xz(diffusion, x_start):
    if x_start.shape[-1] == 151:
        return x_start[:, :, [diffusion.root_x_idx, diffusion.root_z_idx]]
    return x_start[:, :, [0, 2]]


def _generic_tto_loss(diffusion, x_start, cond, t, constraint=None):
    zero = x_start.new_tensor(0.0)
    loss = zero

    target = _trajectory_target(cond, x_start)
    if target is not None:
        pred = _pred_root_xz(diffusion, x_start)
        loss = loss + float(getattr(diffusion, "tto_trajectory_loss_weight", 4.0)) * F.mse_loss(pred, target)

        if pred.shape[1] > 1:
            pred_v = pred[:, 1:] - pred[:, :-1]
            target_v = target[:, 1:] - target[:, :-1]
            loss = loss + float(getattr(diffusion, "tto_trajectory_velocity_loss_weight", 0.5)) * F.mse_loss(pred_v, target_v)

        if pred.shape[1] > 2:
            pred_acc = pred[:, 2:] - 2.0 * pred[:, 1:-1] + pred[:, :-2]
            loss = loss + float(getattr(diffusion, "tto_root_acc_loss_weight", 0.05)) * pred_acc.pow(2).mean()

    if constraint is not None:
        try:
            kf = diffusion._keyframe_loss(x_start, constraint)
            loss = loss + float(getattr(diffusion, "keyframe_loss_weight", 2.0)) * kf
        except Exception:
            pass

    try:
        foot = diffusion._foot_sliding_loss(x_start, target_motion_x0=None)
        loss = loss + float(getattr(diffusion, "tto_foot_loss_weight", 0.25)) * foot
    except Exception:
        pass

    return loss


def _parse_apply_tto_args(args, kwargs):
    # Expected in current EDGE: _apply_tto(x, cond, t, constraint=None)
    if len(args) < 3:
        return None
    x, cond, t = args[0], args[1], args[2]
    rest = args[3:]
    constraint = kwargs.get("constraint", None)
    if constraint is None and rest:
        # First remaining positional argument is often constraint.
        maybe_constraint = rest[0]
        if isinstance(maybe_constraint, dict):
            constraint = maybe_constraint
    return x, cond, t, constraint


def _adam_or_momentum_tto(self, x, cond, t, constraint=None):
    steps = int(getattr(self, "tto_steps", 1))
    if steps <= 0:
        return x

    lr = float(getattr(self, "tto_lr", 0.03))
    optimizer = _env_str("EDGE_TTO_OPTIMIZER", getattr(self, "tto_optimizer", "adam"))

    beta1 = float(getattr(self, "tto_adam_beta1", _env_float("EDGE_TTO_ADAM_BETA1", 0.9)))
    beta2 = float(getattr(self, "tto_adam_beta2", _env_float("EDGE_TTO_ADAM_BETA2", 0.999)))
    eps = float(getattr(self, "tto_eps", _env_float("EDGE_TTO_EPS", 1e-8)))
    momentum = float(getattr(self, "tto_momentum", _env_float("EDGE_TTO_MOMENTUM", 0.9)))
    grad_clip = float(getattr(self, "tto_grad_clip", _env_float("EDGE_TTO_GRAD_CLIP", 1.0)))

    x_opt = x.detach().clone()
    m = torch.zeros_like(x_opt)
    v = torch.zeros_like(x_opt)

    for step in range(1, steps + 1):
        x_var = x_opt.detach().requires_grad_(True)

        _, x_start = self.model_predictions(
            x_var,
            cond,
            t,
            clip_x_start=False,
            constraint=constraint,
        )
        loss = _generic_tto_loss(self, x_start, cond, t, constraint=constraint)

        if not torch.isfinite(loss):
            break

        grad = torch.autograd.grad(loss, x_var, retain_graph=False, create_graph=False, allow_unused=False)[0]
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        if grad_clip > 0:
            grad_norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
            factor = (grad_clip / grad_norm).clamp(max=1.0)
            grad = grad * factor

        if optimizer == "sgd":
            update = grad
        elif optimizer == "momentum":
            m = momentum * m + grad
            update = m
        else:
            # Adam is the default because complex pose/path constraints can
            # otherwise oscillate with raw SGD.
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * grad.pow(2)
            m_hat = m / (1.0 - beta1 ** step)
            v_hat = v / (1.0 - beta2 ** step)
            update = m_hat / (v_hat.sqrt() + eps)

        x_opt = x_var.detach() - lr * update.detach()

    return x_opt.detach()


def install_native_trajectory_control_patch(verbose=True):
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ diffusion safety patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_runtime_safety_patch_installed", False):
        return True

    # Capture current_epoch from training p_losses so validation can reuse it.
    if hasattr(GaussianDiffusion, "p_losses"):
        original_p_losses = GaussianDiffusion.p_losses

        def patched_p_losses(self, *args, **kwargs):
            epoch = _extract_current_epoch(args, kwargs)
            if epoch is not None:
                self._edge_last_current_epoch = int(epoch)
            return original_p_losses(self, *args, **kwargs)

        GaussianDiffusion.p_losses = patched_p_losses
        GaussianDiffusion._edge_original_p_losses = original_p_losses

    # Fix warmup when current_epoch is None.
    if hasattr(GaussianDiffusion, "_linear_warmup"):
        original_linear_warmup = GaussianDiffusion._linear_warmup

        def patched_linear_warmup(self, *args, **kwargs):
            args, kwargs = _replace_none_current_epoch(self, args, kwargs)
            return original_linear_warmup(self, *args, **kwargs)

        GaussianDiffusion._linear_warmup = patched_linear_warmup
        GaussianDiffusion._edge_original_linear_warmup = original_linear_warmup

    # Add robust TTO defaults at construction time.
    original_init = GaussianDiffusion.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.tto_optimizer = _env_str("EDGE_TTO_OPTIMIZER", "adam")
        self.tto_momentum = _env_float("EDGE_TTO_MOMENTUM", 0.9)
        self.tto_adam_beta1 = _env_float("EDGE_TTO_ADAM_BETA1", 0.9)
        self.tto_adam_beta2 = _env_float("EDGE_TTO_ADAM_BETA2", 0.999)
        self.tto_eps = _env_float("EDGE_TTO_EPS", 1e-8)
        self.tto_grad_clip = _env_float("EDGE_TTO_GRAD_CLIP", 1.0)

    GaussianDiffusion.__init__ = patched_init
    GaussianDiffusion._edge_original_init = original_init

    # Replace _apply_tto with optimizer-aware implementation when present.
    if hasattr(GaussianDiffusion, "_apply_tto"):
        original_apply_tto = GaussianDiffusion._apply_tto

        def patched_apply_tto(self, *args, **kwargs):
            parsed = _parse_apply_tto_args(args, kwargs)
            if parsed is None:
                return original_apply_tto(self, *args, **kwargs)

            x, cond, t, constraint = parsed
            optimizer = _env_str("EDGE_TTO_OPTIMIZER", getattr(self, "tto_optimizer", "adam"))

            if optimizer == "original":
                return original_apply_tto(self, *args, **kwargs)

            try:
                return _adam_or_momentum_tto(self, x, cond, t, constraint=constraint)
            except Exception as exc:
                if not getattr(self, "_edge_tto_fallback_warned", False):
                    print(f"⚠️ Adam/Momentum TTO failed; falling back to original TTO: {exc}")
                    self._edge_tto_fallback_warned = True
                return original_apply_tto(self, *args, **kwargs)

        GaussianDiffusion._apply_tto = patched_apply_tto
        GaussianDiffusion._edge_original_apply_tto = original_apply_tto

    GaussianDiffusion._edge_runtime_safety_patch_installed = True

    if verbose:
        print(
            "✅ Installed EDGE runtime safety patch: "
            "warmup-current_epoch fallback + Adam/Momentum TTO."
        )
    return True


def install():
    return install_native_trajectory_control_patch()
