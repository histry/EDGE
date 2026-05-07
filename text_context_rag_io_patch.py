"""Training/inference IO patch for Text/Pose Context RAG."""
from __future__ import annotations

import os
from functools import wraps
from typing import Any, Optional, Tuple
import torch

from text_context_rag_utils import build_context_tensors_from_paths, build_self_context_from_motion, env_bool, env_int, split_csv


def _infer_shape_and_cond(args: Tuple[Any, ...], kwargs: dict):
    shape = args[0] if len(args) >= 1 and isinstance(args[0], tuple) else kwargs.get("shape")
    if len(args) >= 2 and isinstance(args[1], dict):
        return shape, args[1], "args:1"
    if isinstance(kwargs.get("cond"), dict):
        return shape, kwargs["cond"], "kwargs"
    return shape, None, ""


def _replace_cond(args: Tuple[Any, ...], kwargs: dict, cond: dict, location: str):
    if location == "args:1":
        args = list(args)
        args[1] = cond
        return tuple(args), kwargs
    if location == "kwargs":
        kwargs = dict(kwargs)
        kwargs["cond"] = cond
        return args, kwargs
    return args, kwargs


def _infer_device_dtype_batch(cond: dict, shape: Optional[tuple]):
    for value in cond.values():
        if torch.is_tensor(value):
            dtype = value.dtype if value.dtype.is_floating_point else torch.float32
            return value.device, dtype, int(value.shape[0])
    b = int(shape[0]) if shape is not None and len(shape) else 1
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float32, b


def _context_paths_from_env():
    paths = split_csv(os.environ.get("EDGE_RAG_CONTEXT_UNIT_PATHS", ""))
    if not paths:
        paths = split_csv(os.environ.get("EDGE_RAG_SUMMARY_UNIT_PATHS", ""))
    return paths


def _maybe_attach_inference_context(cond: dict, shape: Optional[tuple]) -> dict:
    if not env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
        return cond
    if cond.get("rag_context", None) is not None:
        return cond
    paths = _context_paths_from_env()
    if not paths:
        return cond
    device, dtype, batch_size = _infer_device_dtype_batch(cond, shape)
    context, text, mask, captions = build_context_tensors_from_paths(paths, batch_size=batch_size, device=device, dtype=dtype)
    if context is None:
        return cond
    out = dict(cond)
    out["rag_context"] = context
    out["rag_context_text_embedding"] = text
    out["rag_context_mask"] = mask
    print(f"✅ Text/Pose Context RAG attached to inference cond: units={context.shape[1]}, clip_len={context.shape[2]}, text_dim={text.shape[-1]}")
    if env_bool("EDGE_TEXT_CONTEXT_DEBUG", False):
        for i, cap in enumerate(captions[:5]):
            print(f"   context_caption[{i}]={cap}")
    return out


def _find_training_x_and_cond(args, kwargs):
    x = kwargs.get("x_start", None)
    cond = kwargs.get("cond", None)
    if torch.is_tensor(x) and isinstance(cond, dict):
        return x, cond, "kwargs", None
    cond_index = None
    for i, item in enumerate(args):
        if x is None and torch.is_tensor(item) and item.ndim == 3 and (item.shape[-1] == 151 or item.shape[1] == 151):
            x = item
        elif cond is None and isinstance(item, dict):
            cond = item
            cond_index = i
    if torch.is_tensor(x) and isinstance(cond, dict):
        return x, cond, "args", cond_index
    return None, None, "", None


def _replace_training_cond(args, kwargs, cond, location, cond_index):
    if location == "kwargs":
        kwargs = dict(kwargs)
        kwargs["cond"] = cond
        return args, kwargs
    if location == "args" and cond_index is not None:
        args = list(args)
        args[cond_index] = cond
        return tuple(args), kwargs
    return args, kwargs


def _maybe_attach_training_self_context(x_start: torch.Tensor, cond: dict) -> dict:
    if not env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
        return cond
    if not env_bool("EDGE_TEXT_CONTEXT_TRAIN_SELF", False):
        return cond
    if cond.get("rag_context", None) is not None:
        return cond
    count = env_int("EDGE_TEXT_CONTEXT_TRAIN_COUNT", 3)
    length = env_int("EDGE_RAG_CONTEXT_MAX_LEN", 45)
    context = build_self_context_from_motion(x_start, count=count, length=length)
    B, N, L, C = context.shape
    text_dim = env_int("EDGE_TEXT_CONTEXT_DIM", env_int("EDGE_TEXT_BRIDGE_FALLBACK_DIM", 384))
    text = torch.zeros((B, N, text_dim), device=x_start.device, dtype=x_start.dtype)
    mask = torch.ones((B, N), device=x_start.device, dtype=torch.bool)
    out = dict(cond)
    out["rag_context"] = context
    out["rag_context_text_embedding"] = text
    out["rag_context_mask"] = mask
    return out


def install_text_context_rag_io_patch(verbose: bool = True) -> bool:
    try:
        from model.diffusion import GaussianDiffusion
    except Exception as exc:
        if verbose:
            print(f"⚠️ Text Context RAG IO patch skipped: {exc}")
        return False
    if getattr(GaussianDiffusion, "_edge_text_context_io_patch_installed", False):
        return True
    for method_name in ("p_sample_loop", "ddim_sample"):
        original = getattr(GaussianDiffusion, method_name, None)
        if original is None:
            continue
        @wraps(original)
        def patched_sampler(self, *args, __orig=original, **kwargs):
            shape, cond, location = _infer_shape_and_cond(args, kwargs)
            if isinstance(cond, dict):
                cond2 = _maybe_attach_inference_context(cond, shape)
                args2, kwargs2 = _replace_cond(args, kwargs, cond2, location)
                return __orig(self, *args2, **kwargs2)
            return __orig(self, *args, **kwargs)
        setattr(GaussianDiffusion, method_name, patched_sampler)
    original_p_losses = getattr(GaussianDiffusion, "p_losses", None)
    if original_p_losses is not None:
        @wraps(original_p_losses)
        def patched_p_losses(self, *args, **kwargs):
            x, cond, location, cond_index = _find_training_x_and_cond(args, kwargs)
            if torch.is_tensor(x) and isinstance(cond, dict):
                cond2 = _maybe_attach_training_self_context(x, cond)
                args2, kwargs2 = _replace_training_cond(args, kwargs, cond2, location, cond_index)
                return original_p_losses(self, *args2, **kwargs2)
            return original_p_losses(self, *args, **kwargs)
        GaussianDiffusion.p_losses = patched_p_losses
    GaussianDiffusion._edge_text_context_io_patch_installed = True
    if verbose:
        print("✅ Installed Text/Pose Context RAG IO patch for training/inference.")
    return True


def install():
    return install_text_context_rag_io_patch(verbose=True)
