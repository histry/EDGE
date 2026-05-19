from __future__ import annotations

import os
from typing import Any, Dict

import torch

from hf_event_contrastive import (
    env_bool,
    env_float,
    env_int,
    hf_event_alignment_loss,
    load_hf_event_encoder,
)


_TRUE = {"1", "true", "yes", "y", "on"}


def _enabled() -> bool:
    return env_bool("EDGE_HF_EVENT_CONTRASTIVE", False)


def _warmup(epoch, start: int, end: int) -> float:
    if epoch is None:
        return 1.0
    e = float(epoch)
    if e <= start:
        return 0.0
    if e >= end:
        return 1.0
    return float(e - start) / float(max(1, end - start))


def _get_audio_from_cond(cond: Any):
    if not isinstance(cond, dict):
        return None
    audio = cond.get("audio", None)
    if audio is None:
        audio = cond.get("music", None)
    return audio


def _install_edge_loss_key_patch(verbose: bool = True) -> bool:
    try:
        from EDGE import EDGE
    except Exception:
        return False

    if getattr(EDGE, "_edge_hf_event_loss_keys_patched", False):
        return True

    orig_loss_keys = EDGE._loss_keys

    def patched_loss_keys():
        keys = list(orig_loss_keys())
        if "HF Event Loss" not in keys:
            keys.append("HF Event Loss")
        return keys

    EDGE._loss_keys = staticmethod(patched_loss_keys)
    EDGE._edge_hf_event_loss_keys_patched = True

    if verbose:
        print("✅ Installed HF event loss-key patch: adds 'HF Event Loss' metric when present")
    return True


def install_hf_event_contrastive_patch(verbose: bool = True) -> bool:
    """
    Optional high-frequency audio-motion event contrastive guidance.

    Default OFF:
      EDGE_HF_EVENT_CONTRASTIVE=0

    Enable:
      EDGE_HF_EVENT_CONTRASTIVE=1
      EDGE_HF_EVENT_WEIGHT=0.01
      EDGE_HF_EVENT_ENCODER_CKPT=checkpoints/hf_event_contrastive/.../hf_event_encoder.pt

    Notes:
      - If audio is zero/proxy, the loss returns 0.
      - The encoder checkpoint is loaded frozen.
      - We capture model_out by wrapping self.model.forward during p_losses,
        so no large copy of diffusion.py is needed.
    """
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ HF event contrastive patch skipped: cannot import GaussianDiffusion: {exc}")
        return False

    if getattr(GaussianDiffusion, "_edge_hf_event_contrastive_patched", False):
        _install_edge_loss_key_patch(verbose=False)
        if verbose:
            print("✅ HF event contrastive patch already installed")
        return True

    orig_p_losses = GaussianDiffusion.p_losses

    def _get_encoder(self, device: torch.device):
        ckpt = os.environ.get("EDGE_HF_EVENT_ENCODER_CKPT", "").strip()
        if not ckpt:
            return None
        cached_path = getattr(self, "_edge_hf_event_encoder_path", None)
        cached_encoder = getattr(self, "_edge_hf_event_encoder", None)
        if cached_encoder is not None and cached_path == ckpt:
            return cached_encoder

        encoder = load_hf_event_encoder(ckpt, device=device, freeze=True)
        self._edge_hf_event_encoder = encoder
        self._edge_hf_event_encoder_path = ckpt
        return encoder

    def patched_p_losses(self, x_start, cond, t, noise=None, current_epoch=None, constraint=None):
        if not _enabled():
            return orig_p_losses(
                self,
                x_start,
                cond,
                t,
                noise=noise,
                current_epoch=current_epoch,
                constraint=constraint,
            )

        capture: Dict[str, torch.Tensor] = {}
        model = self.model
        orig_forward = model.forward

        def wrapped_forward(*args, **kwargs):
            # Expected args: x_noisy, cond, t, ...
            if len(args) >= 1 and torch.is_tensor(args[0]):
                capture["x_noisy"] = args[0]
            if len(args) >= 3 and torch.is_tensor(args[2]):
                capture["t"] = args[2]
            out = orig_forward(*args, **kwargs)
            if torch.is_tensor(out):
                capture["model_out"] = out
            return out

        model.forward = wrapped_forward
        try:
            total_loss, losses = orig_p_losses(
                self,
                x_start,
                cond,
                t,
                noise=noise,
                current_epoch=current_epoch,
                constraint=constraint,
            )
        finally:
            model.forward = orig_forward

        weight = env_float("EDGE_HF_EVENT_WEIGHT", 0.0)
        if weight <= 0:
            return total_loss, losses

        audio = _get_audio_from_cond(cond)
        if audio is None:
            return total_loss, losses

        if "model_out" not in capture:
            return total_loss, losses

        model_out = capture["model_out"]

        if getattr(self, "predict_epsilon", False):
            x_noisy = capture.get("x_noisy", None)
            tt = capture.get("t", t)
            if x_noisy is None:
                return total_loss, losses
            model_motion_x0 = self.predict_start_from_noise(x_noisy, tt, model_out)
        else:
            model_motion_x0 = model_out

        start = env_int("EDGE_HF_EVENT_WARMUP_START", 20)
        end = env_int("EDGE_HF_EVENT_WARMUP_END", 80)
        warm = _warmup(current_epoch, start=start, end=end)
        if warm <= 0:
            hf_term = total_loss.new_tensor(0.0)
        else:
            encoder = _get_encoder(self, x_start.device)
            hf_loss, detail = hf_event_alignment_loss(
                model_motion_x0,
                audio,
                encoder=encoder,
                temperature=env_float("EDGE_HF_EVENT_TEMPERATURE", 0.1),
                use_supcon=env_bool("EDGE_HF_EVENT_USE_SUPCON", True),
            )
            hf_term = float(weight) * float(warm) * hf_loss

            if env_bool("EDGE_HF_EVENT_DEBUG", False):
                if not hasattr(self, "_edge_hf_event_debug_counter"):
                    self._edge_hf_event_debug_counter = 0
                self._edge_hf_event_debug_counter += 1
                if self._edge_hf_event_debug_counter <= 20 or self._edge_hf_event_debug_counter % 100 == 0:
                    try:
                        print(
                            "🎵 HF event contrastive | "
                            f"epoch={current_epoch} warm={warm:.3f} "
                            f"weight={weight:.5f} "
                            f"hf_loss={float(hf_loss.detach().cpu()):.6f} "
                            f"term={float(hf_term.detach().cpu()):.6f} "
                            f"cos={float(detail['hf_cos'].detach().cpu()):.6f} "
                            f"supcon={float(detail['hf_supcon'].detach().cpu()):.6f}",
                            flush=True,
                        )
                    except Exception:
                        pass

        total_loss = total_loss + hf_term

        if env_bool("EDGE_HF_EVENT_APPEND_METRIC", True):
            losses = tuple(losses) + (torch.nan_to_num(hf_term, nan=0.0, posinf=1e4, neginf=0.0),)

        return total_loss, losses

    GaussianDiffusion.p_losses = patched_p_losses
    GaussianDiffusion._edge_hf_event_contrastive_patched = True

    _install_edge_loss_key_patch(verbose=verbose)

    if verbose:
        print("✅ Installed HF Audio-Motion Event Contrastive patch (default inactive unless EDGE_HF_EVENT_CONTRASTIVE=1)")
    return True
