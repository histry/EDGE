"""Weak tolerance-band trajectory energy guidance for EDGE sampling.

This is NOT the main trajectory-control method.

Main method:
  native trajectory/event condition branch in the model.

This patch is an optional inference-time safety correction:
  EDGE_WEAK_TRAJ_ENERGY=1

Design:
  - works in normalized root X/Z space by default;
  - uses tolerance band, so root inside the band receives no attraction;
  - scheduled in middle/late denoising only;
  - gradient clipped;
  - small lr by default;
  - no hard replacement of latent or motion.

Recommended:
  EDGE_WEAK_TRAJ_ENERGY=0 during training and formal native evaluation.
  EDGE_WEAK_TRAJ_ENERGY=1 only for final system / controllability sweep.
"""

from __future__ import annotations

import os
from functools import wraps

import torch
import torch.nn.functional as F


_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}

ROOT_X_IDX = 4
ROOT_Z_IDX = 6


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


def env_int(name, default):
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _resize_traj(target, T):
    if target.shape[1] == T:
        return target
    return F.interpolate(
        target.transpose(1, 2),
        size=T,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def _safe_norm(x, dim=-1, eps=1e-8):
    return torch.sqrt(torch.sum(x * x, dim=dim) + eps)


def _should_apply_energy(diffusion, cond, t):
    if not env_bool("EDGE_WEAK_TRAJ_ENERGY", False):
        return False
    if not isinstance(cond, dict) or cond.get("trajectory", None) is None:
        return False

    interval = max(1, env_int("EDGE_WEAK_TRAJ_ENERGY_INTERVAL", 1))
    time_value = int(t[0].detach().cpu().item())
    if time_value % interval != 0:
        return False

    total = max(1, int(getattr(diffusion, "n_timestep", 1000)))
    frac = float(time_value) / float(total)

    # Do not guide very early high-noise steps or final clean steps by default.
    min_frac = env_float("EDGE_WEAK_TRAJ_ENERGY_MIN_T_FRAC", 0.05)
    max_frac = env_float("EDGE_WEAK_TRAJ_ENERGY_MAX_T_FRAC", 0.65)
    if frac < min_frac or frac > max_frac:
        return False

    return True


def _trajectory_energy_loss(diffusion, pred_xstart, cond):
    traj = cond.get("trajectory", None)
    if traj is None:
        return pred_xstart.new_tensor(0.0)

    target = traj.to(device=pred_xstart.device, dtype=pred_xstart.dtype)[..., :2]
    target = _resize_traj(target, pred_xstart.shape[1])

    if pred_xstart.shape[-1] >= 151:
        root_xz = pred_xstart[..., [ROOT_X_IDX, ROOT_Z_IDX]]
    else:
        root_xz = pred_xstart[..., [0, 2]]

    # Normalized-space tolerance band by default.
    tol = env_float("EDGE_WEAK_TRAJ_ENERGY_TOL", 0.08)
    dist = _safe_norm(root_xz - target, dim=-1)
    excess = F.relu(dist - tol)
    pos_loss = (excess * excess).mean()

    vel_loss = pred_xstart.new_tensor(0.0)
    vel_w = env_float("EDGE_WEAK_TRAJ_ENERGY_VEL_W", 0.15)
    if root_xz.shape[1] > 1 and vel_w > 0:
        rv = root_xz[:, 1:] - root_xz[:, :-1]
        tv = target[:, 1:] - target[:, :-1]
        vel_loss = F.mse_loss(rv, tv)

    acc_loss = pred_xstart.new_tensor(0.0)
    acc_w = env_float("EDGE_WEAK_TRAJ_ENERGY_ACC_W", 0.03)
    if root_xz.shape[1] > 2 and acc_w > 0:
        ra = root_xz[:, 2:] - 2.0 * root_xz[:, 1:-1] + root_xz[:, :-2]
        acc_loss = (ra * ra).mean()

    return pos_loss + vel_w * vel_loss + acc_w * acc_loss


def _apply_weak_traj_energy(diffusion, x, cond, t, constraint=None):
    if not _should_apply_energy(diffusion, cond, t):
        return x

    steps = max(1, env_int("EDGE_WEAK_TRAJ_ENERGY_STEPS", 1))
    lr = env_float("EDGE_WEAK_TRAJ_ENERGY_LR", 0.015)
    max_grad_norm = env_float("EDGE_WEAK_TRAJ_ENERGY_MAX_GRAD_NORM", 1.0)

    x_opt = x.detach()

    for _ in range(steps):
        with torch.enable_grad():
            x_opt = x_opt.detach().requires_grad_(True)

            _, pred_xstart = diffusion.model_predictions(
                x_opt,
                cond,
                t,
                clip_x_start=False,
                constraint=constraint,
            )

            loss = _trajectory_energy_loss(diffusion, pred_xstart, cond)
            grad = torch.autograd.grad(loss, x_opt, allow_unused=True)[0]

            if grad is None:
                break

            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            flat = grad.flatten(1)
            norm = _safe_norm(flat, dim=1).clamp_min(1e-8)
            scale = (float(max_grad_norm) / norm).clamp(max=1.0)
            grad = grad * scale.view(-1, *([1] * (grad.ndim - 1)))

            x_opt = x_opt - float(lr) * grad

    if env_bool("EDGE_WEAK_TRAJ_ENERGY_DEBUG", False):
        try:
            print(
                "🧪 weak trajectory energy step | "
                "t=%d lr=%.5f loss=%.6f"
                % (
                    int(t[0].detach().cpu().item()),
                    float(lr),
                    float(loss.detach().cpu().item()) if "loss" in locals() else -1.0,
                ),
                flush=True,
            )
        except Exception:
            pass

    return x_opt.detach()


def install_weak_trajectory_energy_guidance_patch(verbose=True):
    try:
        from model.diffusion import GaussianDiffusion, move_condition_to_device
    except Exception as exc:
        if verbose:
            print("⚠️ weak trajectory energy guidance patch skipped: %s" % exc)
        return False

    if getattr(GaussianDiffusion, "_edge_weak_traj_energy_patch_installed", False):
        return True

    original_p_sample = GaussianDiffusion.p_sample
    original_ddim_sample = GaussianDiffusion.ddim_sample

    @wraps(original_p_sample)
    def patched_p_sample(self, x, cond, t, constraint=None, use_tto=True):
        # Optional weak energy is applied before the normal reverse step.
        x = _apply_weak_traj_energy(self, x, cond, t, constraint=constraint)
        return original_p_sample(self, x, cond, t, constraint=constraint, use_tto=use_tto)

    def patched_ddim_sample(self, shape, cond, constraint=None, sampling_timesteps=50, eta=0.0, **kwargs):
        batch = shape[0]
        device = self.betas.device
        total_timesteps = self.n_timestep

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = move_condition_to_device(cond, device)

        from tqdm import tqdm

        for time, time_next in tqdm(time_pairs, desc="ddim sampling"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            x = _apply_weak_traj_energy(self, x, cond, time_cond, constraint=constraint)

            with torch.no_grad():
                pred_noise, x_start = self.model_predictions(
                    x,
                    cond,
                    time_cond,
                    clip_x_start=self.clip_denoised,
                    constraint=constraint,
                )

                if time_next < 0:
                    x = x_start
                    final_time_cond = torch.zeros((batch,), device=device, dtype=torch.long)
                    x = self._project_known_keyframes(x, constraint, final_time_cond)
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1.0 - alpha / alpha_next) * (1.0 - alpha_next) / (1.0 - alpha)).sqrt()
                c = (1.0 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)
                x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

                next_time_cond = torch.full(
                    (batch,),
                    max(time_next, 0),
                    device=device,
                    dtype=torch.long,
                )
                x = self._project_known_keyframes(x, constraint, next_time_cond)

        return x

    GaussianDiffusion.p_sample = patched_p_sample
    GaussianDiffusion.ddim_sample = patched_ddim_sample
    GaussianDiffusion._edge_weak_traj_energy_patch_installed = True

    if verbose:
        print(
            "✅ Installed weak trajectory energy guidance patch: "
            "enabled=%s, tol=%.4f, lr=%.5f"
            % (
                env_bool("EDGE_WEAK_TRAJ_ENERGY", False),
                env_float("EDGE_WEAK_TRAJ_ENERGY_TOL", 0.08),
                env_float("EDGE_WEAK_TRAJ_ENERGY_LR", 0.015),
            )
        )

    return True


def install():
    return install_weak_trajectory_energy_guidance_patch(verbose=True)
