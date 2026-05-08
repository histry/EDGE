"""Experiment-safety guard for EDGE V9/V10 inference and training.

Drop-in replacement for ``edge_experiment_guard.py``.

V3 additions
------------
1. Keeps the V9/V10 feature-flag and runtime-patch guard behavior.
2. Makes TextContextGradMonitor usable as a context manager so hooks are removed
   even when an ablation/training run raises an exception.
3. Uses ``.item()`` when collecting parameter/gradient norms, avoiding lingering
   tensor objects in long-running batch scripts.
4. Optionally empties CUDA cache when the monitor exits/closes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_TRUE = {"1", "true", "yes", "y", "on"}
_FALSE = {"0", "false", "no", "n", "off"}

TEXT_CONTEXT_MARKERS = (
    "text_context_pose_projection",
    "text_context_pose_encoder",
    "text_context_text_projection",
    "text_context_type_embed",
    "text_context_gate",
)

RAG_SUMMARY_MARKERS = (
    "rag_summary_projection",
    "null_rag_summary_embed",
    "rag_type_embed",
)

PATCH_SPECS: Dict[str, Tuple[str, str]] = {
    "native_trajectory": ("trajectory_native_control", "install_native_trajectory_control_patch"),
    "safety": ("edge_safety_patch", "install_edge_safety_patch"),
    "v9_rag": ("v9_rag_inference_patch", "install_v9_rag_inference_patch"),
    "full_landing": ("edge_full_landing_patch", "install_full_landing_patch"),
    "text_context_model": ("text_context_rag_model_patch", "install_text_context_rag_model_patch"),
    "text_context_io": ("text_context_rag_io_patch", "install_text_context_rag_io_patch"),
    "text_bridge": ("text_bridge_planner_patch", "install_text_bridge_planner_patch"),
    "render_contact": ("render_contact_fix_patch", "install_render_contact_fix_patch"),
}

DEFAULT_REQUIRED_BY_PROFILE: Dict[str, Tuple[str, ...]] = {
    "v9_baseline": ("native_trajectory", "safety", "v9_rag", "full_landing"),
    "baseline": ("native_trajectory", "safety", "v9_rag", "full_landing"),
    "pure_v9": ("native_trajectory", "safety", "v9_rag", "full_landing"),
    "v9": ("native_trajectory", "safety", "v9_rag", "full_landing"),
    "v10": ("native_trajectory", "safety", "v9_rag", "full_landing", "text_bridge"),
    "text_context": (
        "native_trajectory",
        "safety",
        "v9_rag",
        "full_landing",
        "text_context_model",
        "text_context_io",
        "text_bridge",
    ),
    "auto": ("native_trajectory", "safety", "v9_rag", "full_landing"),
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return bool(default)


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def env_str(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip()


def env_was_set(name: str) -> bool:
    return name in os.environ and str(os.environ.get(name, "")).strip() != ""


def _set_default(name: str, value: str) -> None:
    if not env_was_set(name):
        os.environ[name] = str(value)


def infer_checkpoint_path_from_argv(argv: Sequence[str]) -> str:
    argv = list(argv)
    for i, item in enumerate(argv):
        if item == "--checkpoint" and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith("--checkpoint="):
            return item.split("=", 1)[1]
    return env_str("CHECKPOINT", env_str("EDGE_CHECKPOINT", ""))


def _extract_state_dict(checkpoint_obj) -> Optional[dict]:
    if not isinstance(checkpoint_obj, dict):
        return None
    for key in ("ema_state_dict", "model_state_dict", "state_dict"):
        value = checkpoint_obj.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint_obj and all(hasattr(v, "shape") for v in checkpoint_obj.values()):
        return checkpoint_obj
    return None


def load_checkpoint_keys(checkpoint_path: str) -> Tuple[List[str], str]:
    if not checkpoint_path:
        return [], "no_checkpoint_path"
    path = Path(checkpoint_path)
    if not path.exists():
        return [], f"checkpoint_not_found:{checkpoint_path}"
    try:
        import torch

        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")
        state = _extract_state_dict(ckpt)
        if not isinstance(state, dict):
            return [], "no_state_dict"
        return [str(k).replace("module.", "") for k in state.keys()], "ok"
    except Exception as exc:
        return [], f"load_error:{type(exc).__name__}:{exc}"


def keys_contain_any(keys: Iterable[str], markers: Iterable[str]) -> bool:
    keys = list(keys)
    return any(any(marker in key for marker in markers) for key in keys)


def checkpoint_has_text_context(checkpoint_path: str) -> bool:
    keys, _ = load_checkpoint_keys(checkpoint_path)
    return keys_contain_any(keys, TEXT_CONTEXT_MARKERS)


def checkpoint_has_rag_summary(checkpoint_path: str) -> bool:
    keys, _ = load_checkpoint_keys(checkpoint_path)
    return keys_contain_any(keys, RAG_SUMMARY_MARKERS)


def normalize_profile(profile: Optional[str] = None) -> str:
    profile = (profile or os.environ.get("EDGE_EXPERIMENT_PROFILE") or "auto").strip().lower()
    aliases = {
        "v9-pure": "v9_baseline",
        "pure-v9": "v9_baseline",
        "clean_v9": "v9_baseline",
        "clean-v9": "v9_baseline",
        "base": "baseline",
        "tc": "text_context",
    }
    return aliases.get(profile, profile)


def _maybe_write_json_summary(summary: dict, path: str) -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"⚠️ Failed to write EDGE guard summary JSON: {exc}")


def configure_inference_feature_flags(checkpoint_path: str, profile: str = "auto", verbose: bool = True) -> Dict[str, object]:
    profile = normalize_profile(profile)
    os.environ.setdefault("EDGE_EXPERIMENT_PROFILE", profile)

    keys, key_status = load_checkpoint_keys(checkpoint_path)
    has_text_context = keys_contain_any(keys, TEXT_CONTEXT_MARKERS)
    has_rag_summary = keys_contain_any(keys, RAG_SUMMARY_MARKERS)

    explicit_text = env_was_set("EDGE_ENABLE_TEXT_CONTEXT_RAG")
    explicit_rag = env_was_set("EDGE_ENABLE_RAG_SUMMARY_TOKEN")

    if not explicit_text:
        if profile in {"v9_baseline", "baseline", "pure_v9", "v9"}:
            os.environ["EDGE_ENABLE_TEXT_CONTEXT_RAG"] = "0"
        elif profile in {"v10", "text_context"}:
            os.environ["EDGE_ENABLE_TEXT_CONTEXT_RAG"] = "1" if has_text_context else "0"
        else:
            os.environ["EDGE_ENABLE_TEXT_CONTEXT_RAG"] = "1" if has_text_context else "0"

    if not explicit_rag:
        os.environ["EDGE_ENABLE_RAG_SUMMARY_TOKEN"] = "1" if has_rag_summary else "0"

    _set_default("EDGE_RAG_SUMMARY_DIM", "7")
    _set_default("EDGE_RAG_SUMMARY_BLEND_RADIUS", "18")
    _set_default("EDGE_RAG_SUMMARY_MODE", "mean")
    _set_default("EDGE_TEXT_CONTEXT_DIM", "512")
    _set_default("EDGE_TEXT_CONTEXT_MAX_POSE_TOKENS", "64")
    _set_default("EDGE_RAG_CONTEXT_MAX_LEN", "45")
    _set_default("EDGE_TEXT_CONTEXT_DROP_PROB", "0.0")

    if not env_was_set("EDGE_TRAJECTORY_REP"):
        if profile in {"v10", "text_context"} or env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
            os.environ["EDGE_TRAJECTORY_REP"] = "relative_abs_vel"

    summary = {
        "profile": profile,
        "checkpoint_path": checkpoint_path,
        "checkpoint_key_status": key_status,
        "checkpoint_has_text_context": has_text_context,
        "checkpoint_has_rag_summary": has_rag_summary,
        "explicit_EDGE_ENABLE_TEXT_CONTEXT_RAG": explicit_text,
        "explicit_EDGE_ENABLE_RAG_SUMMARY_TOKEN": explicit_rag,
        "EDGE_ENABLE_TEXT_CONTEXT_RAG": os.environ.get("EDGE_ENABLE_TEXT_CONTEXT_RAG"),
        "EDGE_ENABLE_RAG_SUMMARY_TOKEN": os.environ.get("EDGE_ENABLE_RAG_SUMMARY_TOKEN"),
        "EDGE_TRAJECTORY_REP": os.environ.get("EDGE_TRAJECTORY_REP", ""),
    }

    if verbose:
        print("🧪 EDGE experiment feature-flag guard:")
        for k, v in summary.items():
            print(f"  {k}={v}")

    _maybe_write_json_summary(summary, env_str("EDGE_GUARD_SUMMARY_JSON", ""))
    return summary


def install_runtime_patches(
    strict: bool = False,
    required: Optional[Sequence[str]] = None,
    profile: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    profile = normalize_profile(profile)
    if required is None:
        required = DEFAULT_REQUIRED_BY_PROFILE.get(profile, DEFAULT_REQUIRED_BY_PROFILE["auto"])
        if env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False):
            required = tuple(dict.fromkeys(tuple(required) + ("text_context_model", "text_context_io")))

    results: Dict[str, str] = {}
    for patch_name, (module_name, fn_name) in PATCH_SPECS.items():
        try:
            module = __import__(module_name, fromlist=[fn_name])
            fn = getattr(module, fn_name)
            try:
                ok = fn(verbose=verbose)
            except TypeError:
                ok = fn()
            results[patch_name] = "ok" if ok is not False else "returned_false"
        except Exception as exc:
            results[patch_name] = f"missing:{type(exc).__name__}:{exc}"
            if verbose:
                print(f"⚠️ Runtime patch {patch_name} not installed: {exc}")

    missing_required = [name for name in required if not results.get(name, "").startswith("ok")]
    if strict and missing_required:
        detail = {name: results.get(name, "not_attempted") for name in missing_required}
        raise RuntimeError(
            "Required EDGE runtime patches failed to install. "
            f"profile={profile}, missing={detail}"
        )

    if verbose:
        print("🧪 EDGE runtime patch install summary:")
        for name in sorted(results):
            marker = "✅" if results[name].startswith("ok") else "⚠️"
            print(f"  {marker} {name}: {results[name]}")
    return results


def assert_inference_contract(checkpoint_path: str, profile: str = "auto", strict: Optional[bool] = None) -> None:
    profile = normalize_profile(profile)
    if strict is None:
        strict = env_bool("EDGE_STRICT_RUNTIME_PATCHES", False) or env_bool("EDGE_STRICT_EXPERIMENT_GUARD", False)

    keys, key_status = load_checkpoint_keys(checkpoint_path)
    has_text_context = keys_contain_any(keys, TEXT_CONTEXT_MARKERS)
    has_rag_summary = keys_contain_any(keys, RAG_SUMMARY_MARKERS)
    text_enabled = env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False)
    rag_enabled = env_bool("EDGE_ENABLE_RAG_SUMMARY_TOKEN", False)

    errors: List[str] = []
    warnings: List[str] = []

    if profile in {"v9_baseline", "baseline", "pure_v9", "v9"} and text_enabled:
        errors.append(
            "Clean V9/baseline profile has EDGE_ENABLE_TEXT_CONTEXT_RAG=1. "
            "This contaminates the baseline. Set EDGE_ENABLE_TEXT_CONTEXT_RAG=0."
        )

    if text_enabled and not has_text_context:
        errors.append(
            "EDGE_ENABLE_TEXT_CONTEXT_RAG=1 but checkpoint has no text_context_* weights. "
            "This would initialize Text/Pose Context RAG randomly."
        )

    if rag_enabled and checkpoint_path and key_status == "ok" and not has_rag_summary:
        warnings.append(
            "EDGE_ENABLE_RAG_SUMMARY_TOKEN=1 but checkpoint has no rag_summary_* weights. "
            "This may be intentional for new training, but is suspicious for inference."
        )

    if text_enabled:
        try:
            from model.model import DanceDecoder, DecoderLayerStack
            from model.diffusion import GaussianDiffusion

            if not getattr(DanceDecoder, "_edge_text_context_rag_patch_v2_installed", False):
                errors.append("Text/Pose Context RAG model patch v2 is not installed on DanceDecoder.")
            if not getattr(DecoderLayerStack, "_edge_text_context_rag_memory_patch_installed", False):
                errors.append("Text/Pose Context decoder-memory patch is not installed.")
            if not getattr(GaussianDiffusion, "_edge_text_context_io_patch_installed", False):
                errors.append("Text/Pose Context RAG IO patch is not installed on GaussianDiffusion.")
        except Exception as exc:
            errors.append(f"Could not import classes for Text Context patch checks: {exc}")

    if warnings:
        for msg in warnings:
            print(f"⚠️ EDGE inference contract warning: {msg}")

    if errors:
        message = "\n".join(f"- {msg}" for msg in errors)
        raise RuntimeError(f"EDGE inference contract failed:\n{message}")

    print("✅ EDGE inference contract check passed.")


def assert_model_is_clean_baseline(model, name: str = "model") -> None:
    wrapped = getattr(model, "module", model)
    enabled = bool(getattr(wrapped, "enable_text_context_rag", False))
    if enabled:
        raise RuntimeError(f"{name} has enable_text_context_rag=True; this is not a clean baseline.")
    for attr in TEXT_CONTEXT_MARKERS:
        if hasattr(wrapped, attr):
            raise RuntimeError(f"{name} unexpectedly has {attr}; this is not a clean baseline.")
    print(f"✅ {name} clean-baseline check passed: no Text/Pose Context RAG branch.")


def _param_scalar_norm(param) -> float:
    try:
        return float(param.detach().float().norm().item())
    except Exception:
        try:
            return float(param.detach().norm().item())
        except Exception:
            return 0.0


def text_context_parameter_report(model) -> Dict[str, float]:
    """Return parameter norms/trainability for text_context_* modules.

    Uses .item() to materialize Python scalars and avoid retaining tensor
    references in long-running ablation scripts.
    """
    wrapped = getattr(model, "module", model)
    report: Dict[str, float] = {}
    for name, obj in vars(wrapped).items():
        if not str(name).startswith("text_context"):
            continue
        params = []
        if hasattr(obj, "parameters"):
            params = list(obj.parameters())
        elif hasattr(obj, "numel"):
            params = [obj]
        total = sum(int(p.numel()) for p in params)
        trainable = sum(int(p.numel()) for p in params if getattr(p, "requires_grad", False))
        norm = 0.0
        for p in params:
            norm += _param_scalar_norm(p)
        report[f"{name}.total"] = float(total)
        report[f"{name}.trainable"] = float(trainable)
        report[f"{name}.norm_sum"] = float(norm)
    gate = getattr(wrapped, "text_context_gate", None)
    if gate is not None:
        try:
            report["text_context_gate.value"] = float(gate.detach().float().reshape(-1)[0].item())
        except Exception:
            pass
    return report


def write_text_context_report(model, path: str) -> None:
    if not path:
        return
    report = text_context_parameter_report(model)
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Wrote Text/Pose Context RAG report: {p}")
    except Exception as exc:
        print(f"⚠️ Failed to write Text/Pose Context report: {exc}")


def _maybe_empty_cuda_cache() -> None:
    if not env_bool("EDGE_EMPTY_CUDA_CACHE_ON_MONITOR_CLOSE", True):
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class TextContextGradMonitor:
    """Hook-based monitor for text_context_* gradient evidence.

    Use as a context manager in training/ablation scripts:

        with TextContextGradMonitor(model) as mon:
            ... training step ...
            mon.write(path)

    Hooks are removed in __exit__ even if an exception occurs.
    """

    def __init__(self, model, print_every: int = 200):
        self.print_every = max(1, int(print_every))
        self.count = 0
        self.last: Dict[str, float] = {}
        self.handles = []
        wrapped = getattr(model, "module", model)
        for name, param in wrapped.named_parameters():
            if "text_context" not in name:
                continue
            if not param.requires_grad:
                continue
            self.handles.append(param.register_hook(self._make_hook(name)))
        print(f"🧪 TextContextGradMonitor registered hooks: {len(self.handles)}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        _maybe_empty_cuda_cache()
        return False

    def _make_hook(self, name: str):
        def hook(grad):
            try:
                value = float(grad.detach().float().norm().item())
            except Exception:
                value = 0.0
            self.last[name] = value
            self.count += 1
            if value > 0 and (self.count <= 5 or self.count % self.print_every == 0):
                print(f"🧪 text_context grad evidence: {name} grad_norm={value:.6e}")
            return grad
        return hook

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles = []

    def write(self, path: str) -> None:
        if not path:
            return
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "time": time.time(),
                "hook_calls": int(self.count),
                "last_grad_norms": {str(k): float(v) for k, v in self.last.items()},
            }
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ Wrote Text/Pose Context grad monitor report: {p}")
        except Exception as exc:
            print(f"⚠️ Failed to write Text/Pose Context grad monitor report: {exc}")

    def __del__(self):  # best-effort safety for notebooks / interrupted scripts
        try:
            self.close()
        except Exception:
            pass
