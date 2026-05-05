"""Runtime safety patches for EDGE diffusion.

Drop-in replacement for trajectory_native_control.py.

This runtime patch keeps the main model implementation in model/model.py and
patches GaussianDiffusion at import time for:

1) warmup current_epoch fallback when validation does not pass current_epoch;
2) Adam/Momentum TTO with local torch.enable_grad();
3) explicit root-lower velocity matching loss for trajectory-controlled
   generation/training.

Main V7 fix:
- The previous explicit lower-body velocity loss called an undefined
  maybe_unnormalize(), then swallowed the exception and returned 0.
- This file defines _maybe_unnormalize_safe() locally and keeps the operation
  differentiable for torch tensors.
- The explicit loss now returns a raw loss. EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT
  is applied exactly once in the patch wrapper.

Environment knobs:
    EDGE_WARMUP_MISSING_EPOCH_POLICY=last|zero|one

    EDGE_TTO_OPTIMIZER=adam|momentum|sgd|original
    EDGE_TTO_MOMENTUM=0.9
    EDGE_TTO_ADAM_BETA1=0.9
    EDGE_TTO_ADAM_BETA2=0.999
    EDGE_TTO_EPS=1e-8
    EDGE_TTO_GRAD_CLIP=1.0

    EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT=0.0
    EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD=0.006
    EDGE_EXPLICIT_LOWER_RELVEL_RATIO=0.30
    EDGE_EXPLICIT_LOWER_MIN_MOTION=0.003
    EDGE_EXPLICIT_LOWER_STRICT=0
    EDGE_EXPLICIT_LOWER_DEBUG=0
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip().lower()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


_WARNED_MESSAGES: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    print(message)


def _maybe_unnormalize_safe(normalizer, x: torch.Tensor) -> torch.Tensor:
    """Differentiable unnormalize helper.

    Important:
    - Do NOT detach/cpu torch tensors here. This loss must backpropagate.
    - Current dataset.preprocess.Normalizer stores mean/std as numpy arrays and
      its torch branch is differentiable. We implement the same logic locally to
      avoid relying on a missing import.
    - On failure, return x so the loss is still computed in normalized space.
    """
    if normalizer is None:
        return x

    try:
        if torch.is_tensor(x):
            if hasattr(normalizer, "mean") and hasattr(normalizer, "std"):
                mean = torch.as_tensor(normalizer.mean, device=x.device, dtype=x.dtype)
                std = torch.as_tensor(normalizer.std, device=x.device, dtype=x.dtype)
                return x * std + mean

            out = normalizer.unnormalize(x)
            if torch.is_tensor(out):
                return out.to(device=x.device, dtype=x.dtype)

            out = torch.as_tensor(out, device=x.device, dtype=x.dtype)
            return out

        return normalizer.unnormalize(x)

    except Exception as exc:
        msg = f"⚠️ Unnormalize failed in explicit lower loss; using current space instead: {exc}"
        if _env_bool("EDGE_EXPLICIT_LOWER_STRICT", False):
            raise RuntimeError(msg) from exc
        _warn_once("explicit_lower_unnormalize_failed", msg)
        return x


def _extract_current_epoch(args, kwargs) -> Optional[int]:
    if "current_epoch" in kwargs and kwargs["current_epoch"] is not None:
        try:
            return int(kwargs["current_epoch"])
        except Exception:
            return None

    # Most EDGE p_losses signatures put current_epoch near the end.
    for value in reversed(args):
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, int):
            return int(value)
    return None


def _fallback_epoch(self) -> int:
    policy = _env_str("EDGE_WARMUP_MISSING_EPOCH_POLICY", "last")
    if policy == "one":
        return 10**9  # effectively fully warmed
    if policy == "zero":
        return 0

    last_epoch = getattr(self, "_edge_last_current_epoch", None)
    if last_epoch is not None:
        return int(last_epoch)

    return 0


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
    if args[0] is None:
        args[0] = _fallback_epoch(self)
    return tuple(args), kwargs


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
        except Exception as exc:
            if _env_bool("EDGE_EXPLICIT_LOWER_DEBUG", False):
                _warn_once("tto_keyframe_loss_failed", f"⚠️ TTO keyframe loss skipped: {exc}")

    try:
        foot = diffusion._foot_sliding_loss(x_start, target_motion_x0=None)
        loss = loss + float(getattr(diffusion, "tto_foot_loss_weight", 0.25)) * foot
    except Exception as exc:
        if _env_bool("EDGE_EXPLICIT_LOWER_DEBUG", False):
            _warn_once("tto_foot_loss_failed", f"⚠️ TTO foot loss skipped: {exc}")

    return loss


def _parse_apply_tto_args(args, kwargs):
    # Expected in current EDGE: _apply_tto(x, cond, t, constraint=None)
    if len(args) < 3:
        return None
    x, cond, t = args[0], args[1], args[2]
    rest = args[3:]
    constraint = kwargs.get("constraint", None)
    if constraint is None and rest:
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

    # Sampling often runs under torch.no_grad(); TTO needs local gradients.
    with torch.enable_grad():
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

            grad = torch.autograd.grad(
                loss,
                x_var,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
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
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * grad.pow(2)
                m_hat = m / (1.0 - beta1 ** step)
                v_hat = v / (1.0 - beta2 ** step)
                update = m_hat / (v_hat.sqrt() + eps)

            x_opt = x_var.detach() - lr * update.detach()

    return x_opt.detach()


def _edge_explicit_lower_velocity_loss(diffusion, model_motion_x0, target_motion_x0=None, cond=None):
    """Raw explicit root-lower velocity matching loss.

    Returns an unweighted scalar loss. The environment weight is applied exactly
    once in patched_kinematic_sync_loss().

    The loss encourages visible lower-body relative motion when root X/Z speed
    is above a threshold. This targets the failure mode: pelvis/root follows the
    commanded trajectory while legs remain visually under-active.
    """
    enabled_weight = _env_float("EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT", 0.0)
    if enabled_weight <= 0.0:
        if model_motion_x0 is None:
            return torch.tensor(0.0)
        return model_motion_x0.new_tensor(0.0)

    if model_motion_x0 is None:
        raise ValueError("model_motion_x0 is None in explicit lower velocity loss")

    if model_motion_x0.shape[-1] != 151 or model_motion_x0.shape[1] < 2:
        return model_motion_x0.new_tensor(0.0)

    try:
        normalizer = getattr(diffusion, "normalizer", None)
        physical = _maybe_unnormalize_safe(normalizer, model_motion_x0)

        root_x_idx = getattr(diffusion, "root_x_idx", 4)
        root_z_idx = getattr(diffusion, "root_z_idx", 6)
        root_slice = getattr(diffusion, "root_slice", slice(4, 7))

        root = physical[:, :, [root_x_idx, root_z_idx]]
        root_speed = torch.linalg.norm(root[:, 1:] - root[:, :-1], dim=-1)

        joints = diffusion._fk_positions(physical)
        feet = joints[:, :, [7, 8, 10, 11], :]
        pelvis = physical[:, :, root_slice].unsqueeze(2)

        feet_rel = feet - pelvis
        lower_rel_speed = torch.linalg.norm(
            feet_rel[:, 1:, :, [0, 2]] - feet_rel[:, :-1, :, [0, 2]],
            dim=-1,
        ).mean(dim=-1)

        threshold = _env_float("EDGE_EXPLICIT_LOWER_ROOT_SPEED_THRESHOLD", 0.006)
        ratio = _env_float("EDGE_EXPLICIT_LOWER_RELVEL_RATIO", 0.30)
        min_motion = _env_float("EDGE_EXPLICIT_LOWER_MIN_MOTION", 0.003)

        gate = root_speed > float(threshold)
        if not bool(gate.any().item()):
            return model_motion_x0.new_tensor(0.0)

        target_lower = min_motion + ratio * root_speed
        loss = F.relu(target_lower - lower_rel_speed)
        raw = loss[gate].mean()

        if _env_bool("EDGE_EXPLICIT_LOWER_DEBUG", False):
            _warn_once(
                "explicit_lower_debug_once",
                "🔎 Explicit lower velocity loss active: "
                f"raw={float(raw.detach().cpu()):.6f}, "
                f"weight={enabled_weight}, ratio={ratio}, "
                f"threshold={threshold}, min_motion={min_motion}",
            )

        return raw

    except Exception as exc:
        msg = f"⚠️ Explicit lower velocity loss failed: {exc}"
        if _env_bool("EDGE_EXPLICIT_LOWER_STRICT", False):
            raise RuntimeError(msg) from exc
        _warn_once("explicit_lower_loss_failed", msg)
        return model_motion_x0.new_tensor(0.0)


def _edge_add_explicit_loss_to_output(out, extra):
    if extra is None or not torch.is_tensor(extra):
        return out
    if torch.is_tensor(out):
        return out + extra
    if isinstance(out, tuple) and len(out) > 0:
        first = out[0] + extra if torch.is_tensor(out[0]) else out[0]
        return (first,) + tuple(out[1:])
    if isinstance(out, list) and len(out) > 0:
        out = list(out)
        if torch.is_tensor(out[0]):
            out[0] = out[0] + extra
        return out
    return out


def _edge_patch_explicit_lower_velocity(GaussianDiffusion):
    if getattr(GaussianDiffusion, "_edge_explicit_lower_velocity_patch_installed", False):
        return
    if not hasattr(GaussianDiffusion, "_kinematic_sync_loss"):
        return

    original_kinematic_sync_loss = GaussianDiffusion._kinematic_sync_loss

    def patched_kinematic_sync_loss(self, *args, **kwargs):
        out = original_kinematic_sync_loss(self, *args, **kwargs)

        model_motion_x0 = args[0] if args else kwargs.get("model_motion_x0", None)
        target_motion_x0 = args[1] if len(args) > 1 else kwargs.get("target_motion_x0", None)
        cond = args[2] if len(args) > 2 else kwargs.get("cond", None)

        extra_raw = _edge_explicit_lower_velocity_loss(
            self,
            model_motion_x0,
            target_motion_x0=target_motion_x0,
            cond=cond,
        )
        weight = _env_float("EDGE_EXPLICIT_LOWER_VEL_LOSS_WEIGHT", 0.0)
        return _edge_add_explicit_loss_to_output(out, weight * extra_raw)

    GaussianDiffusion._kinematic_sync_loss = patched_kinematic_sync_loss
    GaussianDiffusion._edge_original_kinematic_sync_loss = original_kinematic_sync_loss
    GaussianDiffusion._edge_explicit_lower_velocity_patch_installed = True


def install_native_trajectory_control_patch(verbose=True):
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ diffusion safety patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_runtime_safety_patch_installed", False):
        # The explicit lower velocity patch may not have been present in older
        # imports, so attempt it even if the broader runtime patch is installed.
        _edge_patch_explicit_lower_velocity(GaussianDiffusion)
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

    _edge_patch_explicit_lower_velocity(GaussianDiffusion)

    GaussianDiffusion._edge_runtime_safety_patch_installed = True

    if verbose:
        print(
            "✅ Installed EDGE runtime safety patch: "
            "warmup-current_epoch fallback + Adam/Momentum TTO + explicit lower velocity."
        )
    return True


def install():
    return install_native_trajectory_control_patch()
