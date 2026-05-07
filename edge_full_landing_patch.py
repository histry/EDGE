"""Full-landing runtime patch for EDGE V10/V9 control experiments.

This patch is intentionally checkpoint-compatible:
- It does NOT add trainable parameters to DanceDecoder.
- It does NOT change state_dict shapes.
- It only changes how conditions are prepared at runtime.

It completes the practical landing of three ideas:

1) ZeroInitTrajectoryAdapter usage improvement
   The model already has ZeroInitTrajectoryAdapter.  This patch adds a safe
   translation-invariant trajectory representation switch:
       EDGE_TRAJECTORY_REP=absolute|relative|relative_abs_vel|velocity_only

   In all non-absolute modes, the trajectory condition is translated to start at
   zero.  This removes global X/Z offset while preserving the path shape and
   frame-to-frame velocity that the existing trajectory encoder already computes.

2) Energy-Conditioned CFG landing
   The model already supports cond["energy"] and EDGE_ENERGY_CFG_SCALE.
   This patch ensures inference can always supply an energy condition:
       EDGE_DEFAULT_ENERGY=0.65
       EDGE_AUDIO_ENERGY_AS_COND=1
       EDGE_MUSIC_TENSION_AS_ENERGY=1

   If cond["energy"] is missing, it is generated from a scalar default or from
   audio feature RMS/onset.  This enables Energy CFG ablations without editing
   generate_controlled.py.

3) Context Token RAG practical v1.5
   The model already supports V9 RAG Summary Token.  The full PoseEncoder +
   Cross-Attention RAG is still future work, but this patch adds a conservative
   runtime context enhancement for existing [B,T,7] rag_summary tensors:
       EDGE_CONTEXT_RAG_ENHANCE=1
       EDGE_CONTEXT_RAG_SMOOTH=5
       EDGE_CONTEXT_RAG_SCALE=1.0

   It keeps the same rag_summary_dim=7, so V9 checkpoints remain compatible.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default)).strip().lower()


def _safe_norm01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    dims = tuple(range(1, x.ndim))
    lo = x.amin(dim=dims, keepdim=True)
    hi = x.amax(dim=dims, keepdim=True)
    return (x - lo) / (hi - lo).clamp_min(eps)


def _smooth_1d_batch(x: torch.Tensor, window: int) -> torch.Tensor:
    """Smooth [B,T,1] with avg-pool1d, preserving shape."""
    window = int(max(1, window))
    if window <= 1 or x.ndim != 3 or x.shape[1] < 3:
        return x
    if window % 2 == 0:
        window += 1
    y = x.transpose(1, 2)
    y = F.pad(y, (window // 2, window // 2), mode="replicate")
    y = F.avg_pool1d(y, kernel_size=window, stride=1)
    return y.transpose(1, 2)


def _make_audio_energy(audio_cond: torch.Tensor) -> torch.Tensor:
    """Create [B,T,1] normalized energy/tension from audio features."""
    audio = audio_cond.float()
    # RMS energy across feature channels.
    rms = torch.sqrt(torch.mean(audio.pow(2), dim=-1, keepdim=True) + 1e-8)
    rms = _safe_norm01(rms)

    if _env_bool("EDGE_MUSIC_TENSION_AS_ENERGY", False):
        onset = torch.zeros_like(rms)
        if audio.shape[-1] > 768:
            onset = audio[..., 768:769].clamp_min(0.0)
            onset = _safe_norm01(onset)
        elif audio.shape[1] > 1:
            diff = torch.zeros_like(rms)
            diff[:, 1:] = torch.linalg.norm(audio[:, 1:] - audio[:, :-1], dim=-1, keepdim=True)
            diff[:, 0] = diff[:, 1]
            onset = _safe_norm01(diff)

        rise = torch.zeros_like(rms)
        if rms.shape[1] > 1:
            rise[:, 1:] = torch.clamp(rms[:, 1:] - rms[:, :-1], min=0.0)
        rise = _safe_norm01(rise)

        out = 0.50 * rms + 0.35 * onset + 0.15 * rise
    else:
        out = rms

    out = _safe_norm01(out)
    smooth = _env_int("EDGE_AUDIO_ENERGY_SMOOTH", 5)
    out = _smooth_1d_batch(out, smooth)
    return out.clamp(0.0, 1.0)


def _prepare_trajectory_representation(trajectory: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if trajectory is None:
        return None
    mode = _env_str("EDGE_TRAJECTORY_REP", "absolute")
    if mode in {"", "abs", "absolute", "absolute_xz"}:
        return trajectory

    # Translation-invariant path: remove global X/Z offset while preserving shape.
    if mode in {"relative", "relative_abs", "relative_abs_vel", "local", "velocity_only", "delta"}:
        out = trajectory - trajectory[:, :1]
        scale = _env_float("EDGE_TRAJECTORY_RELATIVE_SCALE", 1.0)
        if abs(scale - 1.0) > 1e-8:
            out = out * scale
        return out

    if mode in {"centered", "centered_abs_vel"}:
        out = trajectory - trajectory.mean(dim=1, keepdim=True)
        scale = _env_float("EDGE_TRAJECTORY_RELATIVE_SCALE", 1.0)
        return out * scale

    # Unknown values should not crash experiments.
    if not getattr(_prepare_trajectory_representation, "_warned", False):
        print(f"⚠️ Unknown EDGE_TRAJECTORY_REP={mode!r}; using absolute trajectory.")
        _prepare_trajectory_representation._warned = True
    return trajectory


def _enhance_rag_summary(rag_summary: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if rag_summary is None or not _env_bool("EDGE_CONTEXT_RAG_ENHANCE", False):
        return rag_summary
    if rag_summary.ndim not in {2, 3}:
        return rag_summary

    scale = _env_float("EDGE_CONTEXT_RAG_SCALE", 1.0)
    if scale <= 0:
        return rag_summary

    if rag_summary.ndim == 2:
        # [B,D] constant summary: normalize only.
        mean = rag_summary.mean(dim=-1, keepdim=True)
        std = rag_summary.std(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = (rag_summary - mean) / std
        return (1.0 - scale) * rag_summary + scale * normalized

    # [B,T,D]: smooth plus local contrast.  Keep same dim.
    x = rag_summary
    window = _env_int("EDGE_CONTEXT_RAG_SMOOTH", 5)
    if window > 1:
        y = x.transpose(1, 2)
        if window % 2 == 0:
            window += 1
        y = F.pad(y, (window // 2, window // 2), mode="replicate")
        smooth = F.avg_pool1d(y, kernel_size=window, stride=1).transpose(1, 2)
    else:
        smooth = x

    # Preserve global summary while slightly emphasizing temporal context.
    contrast = x - smooth
    out = smooth + 0.25 * contrast
    return (1.0 - scale) * x + scale * out


def install_full_landing_patch(verbose: bool = True) -> bool:
    try:
        from model.model import DanceDecoder
    except Exception as exc:
        if verbose:
            print(f"⚠️ EDGE full-landing patch skipped: cannot import DanceDecoder: {exc}")
        return False

    if getattr(DanceDecoder, "_edge_full_landing_patch_installed", False):
        return True

    original_prepare = DanceDecoder._prepare_cond_inputs

    def patched_prepare_cond_inputs(self, cond_embed, batch_size, seq_len, device, dtype):
        audio_cond, trajectory_cond, energy_cond, rag_summary_cond = original_prepare(
            self, cond_embed, batch_size, seq_len, device, dtype
        )

        trajectory_cond = _prepare_trajectory_representation(trajectory_cond)

        if energy_cond is None:
            default_energy = os.environ.get("EDGE_DEFAULT_ENERGY", "").strip()
            if default_energy:
                e = _env_float("EDGE_DEFAULT_ENERGY", 0.55)
                energy_cond = torch.full((batch_size, 1), e, device=device, dtype=dtype).clamp(0.0, 1.0)
            elif _env_bool("EDGE_AUDIO_ENERGY_AS_COND", False) or _env_bool("EDGE_MUSIC_TENSION_AS_ENERGY", False):
                energy_cond = _make_audio_energy(audio_cond).to(device=device, dtype=dtype)

        rag_summary_cond = _enhance_rag_summary(rag_summary_cond)

        return audio_cond, trajectory_cond, energy_cond, rag_summary_cond

    DanceDecoder._prepare_cond_inputs = patched_prepare_cond_inputs
    DanceDecoder._edge_original_prepare_cond_inputs = original_prepare
    DanceDecoder._edge_full_landing_patch_installed = True

    if verbose:
        print(
            "✅ Installed EDGE full-landing patch: "
            "relative trajectory option + inference energy condition + context RAG summary enhancement."
        )
    return True


def install():
    return install_full_landing_patch(verbose=True)


if __name__ == "__main__":
    install()
