"""Training/inference IO patch for Text/Pose Context RAG with ablation modes.

Drop-in replacement for ``text_context_rag_io_patch.py``.

Adds inference ablation modes controlled by ``EDGE_RAG_CONTEXT_MODE``:
    normal       : use retrieved context as-is
    no_context   : skip attaching context
    shuffled     : shuffle/reverse unit order and text together
    shuffled_text: keep pose context, shuffle text embeddings
    wrong_text   : keep pose context, replace text embeddings with deterministic noise
    zero_text    : keep pose context, zero text embeddings

Optional report:
    EDGE_RAG_CONTEXT_REPORT_JSON=output/.../context_report.json
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import wraps
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from text_context_rag_utils import (
    build_context_tensors_from_paths,
    build_self_context_from_motion,
    env_bool,
    env_int,
    split_csv,
)


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


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


def _tensor_hash(t: Optional[torch.Tensor]) -> str:
    if t is None:
        return ""
    try:
        arr = t.detach().float().cpu().contiguous().numpy().tobytes()
        return "sha256:" + hashlib.sha256(arr).hexdigest()
    except Exception:
        return "hash_error"


def _paths_hash(paths) -> str:
    text = "\n".join(str(p) for p in paths)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_context_report(payload: dict) -> None:
    path = _env_str("EDGE_RAG_CONTEXT_REPORT_JSON", "")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Text/Pose Context RAG report saved: {p}")
    except Exception as exc:
        print(f"⚠️ Failed to write Text/Pose Context RAG report: {exc}")


def _deterministic_wrong_text_like(text: torch.Tensor) -> torch.Tensor:
    seed = int(float(os.environ.get("EDGE_RAG_CONTEXT_WRONG_TEXT_SEED", "1234")))
    gen = torch.Generator(device=text.device)
    try:
        gen.manual_seed(seed)
        return torch.randn(text.shape, device=text.device, dtype=text.dtype, generator=gen)
    except TypeError:
        torch.manual_seed(seed)
        return torch.randn_like(text)


def _apply_context_ablation(context, text, mask, mode: str):
    mode = str(mode or "normal").strip().lower()
    if mode in {"normal", "context", "on"}:
        return context, text, mask, mode
    if mode in {"no_context", "none", "off", "disable", "disabled"}:
        return None, None, None, "no_context"

    if context is None:
        return context, text, mask, mode

    if mode in {"shuffled", "shuffle"}:
        if context.shape[1] > 1:
            idx = torch.arange(context.shape[1] - 1, -1, -1, device=context.device)
            context = context[:, idx]
            if text is not None and text.ndim >= 3 and text.shape[1] == len(idx):
                text = text[:, idx]
            if mask is not None and mask.ndim >= 2 and mask.shape[1] == len(idx):
                mask = mask[:, idx]
        else:
            context = torch.flip(context, dims=[2])
        return context, text, mask, "shuffled"

    if mode in {"shuffled_text", "shuffle_text"}:
        if text is not None:
            if text.ndim >= 3 and text.shape[1] > 1:
                idx = torch.arange(text.shape[1] - 1, -1, -1, device=text.device)
                text = text[:, idx]
            else:
                text = -text
        return context, text, mask, "shuffled_text"

    if mode in {"wrong_text", "random_text"}:
        if text is not None:
            text = _deterministic_wrong_text_like(text)
        return context, text, mask, "wrong_text"

    if mode in {"zero_text", "text_zero"}:
        if text is not None:
            text = torch.zeros_like(text)
        return context, text, mask, "zero_text"

    print(f"⚠️ Unknown EDGE_RAG_CONTEXT_MODE={mode!r}; using normal context.")
    return context, text, mask, "normal"


def _maybe_attach_inference_context(cond: dict, shape: Optional[tuple]) -> dict:
    if not env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
        return cond
    if cond.get("rag_context", None) is not None:
        return cond

    mode = _env_str("EDGE_RAG_CONTEXT_MODE", "normal").lower()
    if mode in {"no_context", "none", "off", "disable", "disabled"}:
        out = dict(cond)
        out["_edge_rag_context_mode"] = "no_context"
        _write_context_report({"attached": False, "mode": "no_context", "reason": "ablation_mode"})
        return out

    paths = _context_paths_from_env()
    if not paths:
        if os.environ.get("EDGE_RUN_MODE", "").lower() == "formal" and env_bool("EDGE_TEXT_CONTEXT_REQUIRED", False):
            raise RuntimeError("EDGE_TEXT_CONTEXT_REQUIRED=1 but no EDGE_RAG_CONTEXT_UNIT_PATHS/EDGE_RAG_SUMMARY_UNIT_PATHS provided.")
        return cond

    device, dtype, batch_size = _infer_device_dtype_batch(cond, shape)
    context, text, mask, captions = build_context_tensors_from_paths(paths, batch_size=batch_size, device=device, dtype=dtype)
    if context is None:
        if os.environ.get("EDGE_RUN_MODE", "").lower() == "formal" and env_bool("EDGE_TEXT_CONTEXT_REQUIRED", False):
            raise RuntimeError("EDGE_TEXT_CONTEXT_REQUIRED=1 but context tensor construction returned None.")
        return cond

    context, text, mask, mode_used = _apply_context_ablation(context, text, mask, mode)
    if context is None:
        out = dict(cond)
        out["_edge_rag_context_mode"] = mode_used
        _write_context_report({"attached": False, "mode": mode_used, "paths": paths, "path_hash": _paths_hash(paths)})
        return out

    out = dict(cond)
    out["rag_context"] = context
    out["rag_context_text_embedding"] = text
    out["rag_context_mask"] = mask
    out["_edge_rag_context_mode"] = mode_used

    report = {
        "attached": True,
        "mode": mode_used,
        "unit_count": int(context.shape[1]),
        "clip_len": int(context.shape[2]),
        "feature_dim": int(context.shape[-1]),
        "text_dim": int(text.shape[-1]) if text is not None else 0,
        "paths": [str(p) for p in paths],
        "path_hash": _paths_hash(paths),
        "context_hash": _tensor_hash(context),
        "text_hash": _tensor_hash(text),
        "mask_hash": _tensor_hash(mask.float() if mask is not None else None),
        "captions_head": captions[:5] if captions else [],
    }
    _write_context_report(report)

    print(
        "✅ Text/Pose Context RAG attached to inference cond: "
        f"mode={mode_used}, units={context.shape[1]}, clip_len={context.shape[2]}, "
        f"text_dim={text.shape[-1] if text is not None else 0}"
    )
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
    if getattr(GaussianDiffusion, "_edge_text_context_io_patch_v2_installed", False):
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
    GaussianDiffusion._edge_text_context_io_patch_v2_installed = True
    if verbose:
        print("✅ Installed Text/Pose Context RAG IO patch v2 for training/inference ablations.")
    return True


def install():
    return install_text_context_rag_io_patch(verbose=True)
