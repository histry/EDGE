#!/usr/bin/env python3
"""Runtime bridge for V9 RAG Summary Token inference.

This patch fixes the inference-side gap in the first V9/V10 implementation.

It does two things:

1. EDGE construction patch
   If EDGE_ENABLE_RAG_SUMMARY_TOKEN=1, EDGE_RAG_SUMMARY_UNIT_PATHS is set, or a
   checkpoint contains rag_summary_projection weights, EDGE.__init__ is forced
   to construct the V9 RAG Summary Token branch.  This prevents messages like:

       ignored unexpected keys=['null_rag_summary_embed', 'rag_type_embed',
       'rag_summary_projection.0.weight', ...]

2. Sampling condition patch
   During p_sample_loop / ddim_sample, if EDGE_RAG_SUMMARY_UNIT_PATHS is set,
   cond["rag_summary"]=[B,T,D] is built from the retrieved unit .npy files and
   attached before model inference.

Expected env from generate_v10_choreo.py:
   EDGE_ENABLE_RAG_SUMMARY_TOKEN=1
   EDGE_RAG_SUMMARY_UNIT_PATHS=unit1.npy,unit2.npy
   EDGE_RAG_SUMMARY_MID_FRAMES=50,100
   EDGE_RAG_SUMMARY_DIM=7
   EDGE_RAG_SUMMARY_BLEND_RADIUS=18
"""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch


_PATCH_INSTALLED = False
_RAG_CACHE: dict[tuple, torch.Tensor] = {}


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _split_csv(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for item in _split_csv(text):
        try:
            out.append(int(round(float(item))))
        except Exception:
            pass
    return out


def _extract_checkpoint_path(args: Tuple[Any, ...], kwargs: dict) -> str:
    # EDGE.__init__(feature_type, checkpoint_path="", ...)
    if "checkpoint_path" in kwargs and kwargs.get("checkpoint_path"):
        return str(kwargs.get("checkpoint_path"))
    if len(args) >= 2 and args[1]:
        return str(args[1])
    return ""


def _state_dict_from_checkpoint(ckpt: Any) -> Optional[dict]:
    if isinstance(ckpt, dict):
        for key in ("ema_state_dict", "model_state_dict", "state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    return None


def _checkpoint_has_rag_branch(checkpoint_path: str) -> bool:
    if not checkpoint_path:
        return False
    path = Path(checkpoint_path)
    if not path.exists():
        return False

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return False

    state = _state_dict_from_checkpoint(ckpt)
    if not isinstance(state, dict):
        return False
    return any("rag_summary_projection" in str(k) or "null_rag_summary_embed" in str(k) for k in state.keys())


def _force_edge_rag_init(verbose: bool = True) -> None:
    """Patch EDGE.EDGE.__init__ so inference can load V9 checkpoints correctly."""
    try:
        import EDGE as edge_module
    except Exception as exc:
        if verbose:
            print(f"⚠️ V9 RAG EDGE init patch skipped: {exc}")
        return

    EDGEClass = getattr(edge_module, "EDGE", None)
    if EDGEClass is None or getattr(EDGEClass, "_v9_rag_init_patch_installed", False):
        return

    original_init = EDGEClass.__init__

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        checkpoint_path = _extract_checkpoint_path(args, kwargs)
        checkpoint_has_rag = _checkpoint_has_rag_branch(checkpoint_path)

        should_enable = (
            _env_flag("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "0")
            or bool(os.environ.get("EDGE_RAG_SUMMARY_UNIT_PATHS", "").strip())
            or checkpoint_has_rag
        )

        if should_enable:
            kwargs["enable_rag_summary_token"] = True
            kwargs["rag_summary_dim"] = int(kwargs.get("rag_summary_dim", _env_int("EDGE_RAG_SUMMARY_DIM", 7)))
            kwargs["rag_summary_drop_prob"] = float(kwargs.get("rag_summary_drop_prob", 0.0))
            os.environ.setdefault("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "1")
            if verbose:
                print(
                    "✅ V9 RAG Summary Token inference enabled before EDGE init: "
                    f"dim={kwargs['rag_summary_dim']}, checkpoint_has_rag={checkpoint_has_rag}"
                )

        return original_init(self, *args, **kwargs)

    EDGEClass.__init__ = patched_init
    EDGEClass._v9_rag_init_patch_installed = True


def _unwrap_model(model):
    if model is not None and hasattr(model, "module"):
        return model.module
    return model


def _get_model_rag_dim(diffusion) -> int:
    model = _unwrap_model(getattr(diffusion, "model", None))
    return int(getattr(model, "rag_summary_dim", _env_int("EDGE_RAG_SUMMARY_DIM", 7)))


def _model_rag_enabled(diffusion) -> bool:
    if _env_flag("EDGE_ENABLE_RAG_SUMMARY_TOKEN", "0"):
        return True
    model = _unwrap_model(getattr(diffusion, "model", None))
    return bool(getattr(model, "enable_rag_summary_token", False))


def _infer_shape_and_cond(args: Tuple[Any, ...], kwargs: dict) -> tuple[Optional[tuple], Optional[dict], str]:
    """Return (shape, cond, location)."""
    shape = args[0] if len(args) >= 1 and isinstance(args[0], tuple) else kwargs.get("shape")

    if len(args) >= 2 and isinstance(args[1], dict):
        return shape, args[1], "args:1"
    if isinstance(kwargs.get("cond"), dict):
        return shape, kwargs["cond"], "kwargs"

    return shape, None, ""


def _replace_cond(args: Tuple[Any, ...], kwargs: dict, cond: dict, location: str) -> tuple[Tuple[Any, ...], dict]:
    if location == "args:1":
        args_list = list(args)
        args_list[1] = cond
        return tuple(args_list), kwargs
    if location == "kwargs":
        kwargs = dict(kwargs)
        kwargs["cond"] = cond
        return args, kwargs
    return args, kwargs


def _load_unit_summary(path: str, dim: int, device, dtype) -> torch.Tensor:
    from rag_context_tokens import summarize_151_unit

    arr = np.load(path, allow_pickle=True)
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1 and arr.shape[0] == 151:
        # A single pose has no temporal activity.  Return zeros rather than
        # pretending it is a motion unit.
        return torch.zeros(dim, device=device, dtype=dtype)

    summary = summarize_151_unit(arr)
    vec = summary.to_tensor(device=device, dtype=dtype).reshape(-1)
    if vec.numel() < dim:
        vec = torch.cat([vec, torch.zeros(dim - vec.numel(), device=device, dtype=dtype)])
    elif vec.numel() > dim:
        vec = vec[:dim]
    return vec


def _build_temporal_rag_summary(
    unit_paths: Sequence[str],
    frames: Sequence[int],
    batch_size: int,
    seq_len: int,
    dim: int,
    device,
    dtype,
) -> torch.Tensor:
    valid_paths = [str(p) for p in unit_paths if p and Path(p).exists()]
    if not valid_paths:
        return torch.zeros((batch_size, seq_len, dim), device=device, dtype=dtype)

    radius = max(1, _env_int("EDGE_RAG_SUMMARY_BLEND_RADIUS", 18))
    cache_key = (
        tuple(valid_paths),
        tuple(int(x) for x in frames),
        int(batch_size),
        int(seq_len),
        int(dim),
        str(device),
        str(dtype),
        int(radius),
    )
    cached = _RAG_CACHE.get(cache_key)
    if cached is not None:
        return cached.to(device=device, dtype=dtype)

    vecs = []
    for path in valid_paths:
        try:
            vecs.append(_load_unit_summary(path, dim=dim, device=device, dtype=dtype))
        except Exception as exc:
            print(f"⚠️ Failed to summarize RAG unit for V9 token: {path} | {exc}")

    if not vecs:
        rag = torch.zeros((batch_size, seq_len, dim), device=device, dtype=dtype)
        _RAG_CACHE[cache_key] = rag.detach().cpu()
        return rag

    vec_stack = torch.stack(vecs, dim=0)
    mean_vec = vec_stack.mean(dim=0)
    timeline = mean_vec.view(1, 1, dim).expand(batch_size, seq_len, dim).clone()

    if frames and len(frames) == len(vecs):
        weighted = torch.zeros_like(timeline)
        weights = torch.zeros((batch_size, seq_len, 1), device=device, dtype=dtype)

        t = torch.arange(seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
        for vec, frame in zip(vecs, frames):
            frame = max(0, min(seq_len - 1, int(frame)))
            dist = torch.abs(t - float(frame))
            w = torch.clamp(1.0 - dist / float(radius), min=0.0, max=1.0)
            weighted = weighted + w * vec.view(1, 1, dim)
            weights = weights + w

        local = weighted / weights.clamp_min(1e-8)
        timeline = torch.where(weights > 1e-6, local, timeline)

    _RAG_CACHE[cache_key] = timeline.detach().cpu()
    return timeline


def _infer_device_dtype_from_cond(cond: dict) -> tuple[torch.device, torch.dtype, int, int]:
    for value in cond.values():
        if torch.is_tensor(value):
            dtype = value.dtype if value.dtype.is_floating_point else torch.float32
            return value.device, dtype, int(value.shape[0]), int(value.shape[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, torch.float32, 1, 150


def _maybe_attach_rag_summary(diffusion, cond: dict, shape: Optional[tuple]) -> dict:
    if not isinstance(cond, dict):
        return cond
    if cond.get("rag_summary", None) is not None:
        return cond

    unit_paths = _split_csv(os.environ.get("EDGE_RAG_SUMMARY_UNIT_PATHS", ""))
    if not unit_paths:
        return cond

    if not _model_rag_enabled(diffusion):
        print("⚠️ EDGE_RAG_SUMMARY_UNIT_PATHS is set, but model RAG branch is not enabled; skipping rag_summary.")
        return cond

    device, dtype, inferred_b, inferred_t = _infer_device_dtype_from_cond(cond)
    if shape is not None and len(shape) >= 2:
        batch_size = int(shape[0])
        seq_len = int(shape[1])
    else:
        batch_size, seq_len = inferred_b, inferred_t

    dim = _get_model_rag_dim(diffusion)
    frames = _parse_ints(os.environ.get("EDGE_RAG_SUMMARY_MID_FRAMES", ""))

    rag = _build_temporal_rag_summary(
        unit_paths=unit_paths,
        frames=frames,
        batch_size=batch_size,
        seq_len=seq_len,
        dim=dim,
        device=device,
        dtype=dtype,
    )

    cond = dict(cond)
    cond["rag_summary"] = rag
    print(
        "✅ V9 RAG summary attached to inference cond: "
        f"units={len(unit_paths)}, frames={frames if frames else 'mean'}, shape={tuple(rag.shape)}"
    )
    return cond


def _patch_diffusion_sampling(verbose: bool = True) -> None:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ V9 RAG diffusion patch skipped: {exc}")
        return

    if getattr(GaussianDiffusion, "_v9_rag_sampling_patch_installed", False):
        return

    def wrap_sampler(method_name: str) -> None:
        original = getattr(GaussianDiffusion, method_name, None)
        if original is None:
            return

        @wraps(original)
        def patched(self, *args, **kwargs):
            shape, cond, location = _infer_shape_and_cond(args, kwargs)
            if isinstance(cond, dict):
                cond = _maybe_attach_rag_summary(self, cond, shape)
                args2, kwargs2 = _replace_cond(args, kwargs, cond, location)
                return original(self, *args2, **kwargs2)
            return original(self, *args, **kwargs)

        setattr(GaussianDiffusion, method_name, patched)

    wrap_sampler("p_sample_loop")
    wrap_sampler("ddim_sample")

    GaussianDiffusion._v9_rag_sampling_patch_installed = True
    if verbose:
        print("✅ Installed V9 RAG summary inference patch for DDPM/DDIM sampling.")


def install_v9_rag_inference_patch(verbose: bool = True) -> bool:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return True

    _force_edge_rag_init(verbose=verbose)
    _patch_diffusion_sampling(verbose=verbose)

    _PATCH_INSTALLED = True
    return True


def install():
    return install_v9_rag_inference_patch(verbose=True)
