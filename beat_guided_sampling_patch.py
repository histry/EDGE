"""Beat / energy guided sampling patch for EDGE diffusion.

V11.1 improvements:
- Supports full-body energy with EDGE_BEAT_GUIDANCE_FEATURES=all.
- Adds scheduled guidance so early noisy steps are not over-constrained.
- Adds gradient clipping and per-sample normalization.

Recommended inference:
    EDGE_BEAT_GUIDANCE=1
    EDGE_BEAT_GUIDANCE_WEIGHT=0.03
    EDGE_BEAT_GUIDANCE_TARGET=1.35
    EDGE_BEAT_GUIDANCE_FEATURES=all
"""
from __future__ import annotations

import os
from functools import wraps
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_TRUE = {"1", "true", "yes", "y", "on"}
ROT_SLICE = slice(7, 151)
UPPER_JOINTS = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


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


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _upper_rot_indices(device=None) -> torch.Tensor:
    idx = []
    for j in UPPER_JOINTS:
        idx.extend(range(7 + 6 * j, 7 + 6 * j + 6))
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _normalize_01(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.amin(dim=1, keepdim=True)
    return x / x.amax(dim=1, keepdim=True).clamp_min(1e-8)


def _beat_mask_from_audio(audio: torch.Tensor, seq_len: int) -> torch.Tensor:
    audio = audio.float()
    if audio.ndim != 3:
        raise ValueError(f"audio must be [B,T,C], got {tuple(audio.shape)}")
    if audio.shape[1] != seq_len:
        audio = F.interpolate(audio.transpose(1, 2), size=seq_len, mode="linear", align_corners=False).transpose(1, 2)

    if audio.shape[-1] > 768:
        onset = torch.relu(audio[..., 768])
    elif audio.shape[-1] >= 35:
        onset = torch.relu(audio[..., 0] + 0.5 * audio[..., -2] + 0.5 * audio[..., -1])
    else:
        onset = torch.zeros(audio.shape[:2], device=audio.device, dtype=audio.dtype)
        if audio.shape[1] > 1:
            onset[:, 1:] = torch.linalg.norm(audio[:, 1:] - audio[:, :-1], dim=-1)
            onset[:, 0] = onset[:, 1]
    onset = _normalize_01(onset)
    q = _env_float("EDGE_BEAT_MASK_QUANTILE", 0.88)
    thr = torch.quantile(onset.detach(), q, dim=1, keepdim=True)
    return (onset >= thr).float()


def _beat_mask_from_path(path: str, batch: int, seq_len: int, device, dtype) -> torch.Tensor:
    arr = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[0] == 1 and batch > 1:
        arr = np.repeat(arr, batch, axis=0)
    if arr.shape[1] != seq_len:
        t = torch.from_numpy(arr).to(device=device, dtype=dtype)[:, None]
        t = F.interpolate(t, size=seq_len, mode="linear", align_corners=False)[:, 0]
        return (t > 0.5).float()
    return torch.from_numpy(arr[:batch]).to(device=device, dtype=dtype).float()


def build_beat_mask(cond, batch: int, seq_len: int, device, dtype) -> Optional[torch.Tensor]:
    path = _env_str("EDGE_BEAT_MASK_PATH", "")
    if path:
        return _beat_mask_from_path(path, batch, seq_len, device, dtype)
    if isinstance(cond, dict) and torch.is_tensor(cond.get("beat_mask", None)):
        bm = cond["beat_mask"].to(device=device, dtype=dtype)
        if bm.ndim == 3:
            bm = bm[..., 0]
        if bm.shape[1] != seq_len:
            bm = F.interpolate(bm[:, None], size=seq_len, mode="linear", align_corners=False)[:, 0]
        return (bm > 0.5).float()
    if isinstance(cond, dict) and torch.is_tensor(cond.get("audio", None)):
        return _beat_mask_from_audio(cond["audio"].to(device=device, dtype=dtype), seq_len)
    return None


def motion_energy(x0: torch.Tensor) -> torch.Tensor:
    features = _env_str("EDGE_BEAT_GUIDANCE_FEATURES", "upper").lower()
    if x0.shape[-1] == 151:
        if features == "upper":
            feat = x0.index_select(-1, _upper_rot_indices(x0.device))
        elif features == "all":
            feat = x0
        elif features in {"root", "root_xz"}:
            feat = x0[..., [4, 6]]
        else:
            feat = x0[..., ROT_SLICE]
    else:
        feat = x0
    vel = torch.zeros(feat.shape[:2], device=feat.device, dtype=feat.dtype)
    if feat.shape[1] > 1:
        vel[:, 1:] = torch.linalg.norm(feat[:, 1:] - feat[:, :-1], dim=-1)
        vel[:, 0] = vel[:, 1]
    return vel


def energy_guidance_loss(x0: torch.Tensor, beat_mask: torch.Tensor) -> torch.Tensor:
    energy = _normalize_01(motion_energy(x0))
    beat_mask = beat_mask.to(device=x0.device, dtype=x0.dtype)
    if beat_mask.shape[1] != x0.shape[1]:
        beat_mask = F.interpolate(beat_mask[:, None], size=x0.shape[1], mode="linear", align_corners=False)[:, 0]

    target = _env_float("EDGE_BEAT_GUIDANCE_TARGET", 1.35)
    offbeat_w = _env_float("EDGE_BEAT_GUIDANCE_OFFBEAT_WEIGHT", 0.05)
    beat_term = -target * (beat_mask * energy).sum() / beat_mask.sum().clamp_min(1.0)
    offbeat = 1.0 - beat_mask
    offbeat_term = offbeat_w * (offbeat * energy.pow(2)).sum() / offbeat.sum().clamp_min(1.0)
    return beat_term + offbeat_term


def _timestep_schedule_weight(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(t):
        return torch.tensor(1.0)
    t_float = t.float()
    denom = t_float.max().clamp_min(1.0)
    progress = 1.0 - (t_float / denom)
    start = _env_float("EDGE_BEAT_GUIDANCE_START_PROGRESS", 0.25)
    end = _env_float("EDGE_BEAT_GUIDANCE_END_PROGRESS", 1.0)
    w = ((progress - start) / max(end - start, 1e-6)).clamp(0.0, 1.0)
    return w.view(-1, 1, 1)


def install_beat_guided_sampling_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ Beat-guided sampling patch skipped: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_beat_guided_sampling_patch_v111_installed", False):
        return True

    original_model_predictions = GaussianDiffusion.model_predictions

    @wraps(original_model_predictions)
    def patched_model_predictions(self, x, cond, t, *args, **kwargs):
        pred_noise, x_start = original_model_predictions(self, x, cond, t, *args, **kwargs)

        if not _env_bool("EDGE_BEAT_GUIDANCE", False) or getattr(self, "training", False):
            return pred_noise, x_start

        weight = _env_float("EDGE_BEAT_GUIDANCE_WEIGHT", _env_float("EDGE_BEAT_GUIDANCE_ALPHA", 0.02))
        if weight <= 0:
            return pred_noise, x_start

        beat_mask = build_beat_mask(cond, x_start.shape[0], x_start.shape[1], x_start.device, x_start.dtype)
        if beat_mask is None or not bool((beat_mask.sum() > 0).detach().cpu().item()):
            return pred_noise, x_start

        with torch.enable_grad():
            x0_leaf = x_start.detach().clone().requires_grad_(True)
            loss = energy_guidance_loss(x0_leaf, beat_mask)
            grad = torch.autograd.grad(loss, x0_leaf, retain_graph=False, create_graph=False)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            max_norm = _env_float("EDGE_BEAT_GUIDANCE_MAX_GRAD_NORM", 5.0)
            gnorm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=-1).clamp_min(1e-8)
            scale = torch.clamp(max_norm / gnorm, max=1.0).view(-1, 1, 1)
            sched = _timestep_schedule_weight(t).to(device=x_start.device, dtype=x_start.dtype)
            x0_guided = (x0_leaf - weight * sched * grad * scale).detach()

        if _env_bool("EDGE_BEAT_GUIDANCE_VERBOSE", False):
            print(
                "🥁 Beat-guided sampling: "
                f"loss={float(loss.detach().cpu().item()):.6f}, "
                f"beat_ratio={float(beat_mask.mean().detach().cpu().item()):.4f}, "
                f"weight={weight}"
            )

        return self.predict_noise_from_start(x, t, x0_guided), x0_guided

    GaussianDiffusion.model_predictions = patched_model_predictions
    GaussianDiffusion._edge_beat_guided_sampling_patch_installed = True
    GaussianDiffusion._edge_beat_guided_sampling_patch_v111_installed = True

    if verbose:
        print(
            "✅ Installed beat-guided sampling patch v11.1: "
            f"enabled={_env_bool('EDGE_BEAT_GUIDANCE', False)}, "
            f"weight={_env_float('EDGE_BEAT_GUIDANCE_WEIGHT', 0.02)}, "
            f"features={_env_str('EDGE_BEAT_GUIDANCE_FEATURES', 'upper')}"
        )
    return True


def install():
    return install_beat_guided_sampling_patch(verbose=True)
