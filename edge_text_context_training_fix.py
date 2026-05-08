"""Training-stage fix and evidence monitor for Text/Pose Context RAG.

Drop-in replacement for ``edge_text_context_training_fix.py``.

Fixes/diagnostics:
1. Keeps text_context_* modules trainable in train_stage="adapter".
2. Writes text_context_gate / parameter snapshots.
3. Registers gradient hooks for text_context_* parameters.
4. Optional formal hard checks:
      EDGE_TEXT_CONTEXT_REQUIRE_GRAD=1
      EDGE_TEXT_CONTEXT_MIN_GRAD_NORM=1e-10
      EDGE_TEXT_CONTEXT_REQUIRE_GATE=1
      EDGE_TEXT_CONTEXT_MIN_GATE_ABS=1e-4

Note
----
This file can only check final train-loop evidence because the original EDGE
train_loop owns epoch boundaries.  For per-epoch dashboards, set JSON paths and
let your logger or post-hook ingest them.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_TRUE = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _formal() -> bool:
    return str(os.environ.get("EDGE_RUN_MODE", "")).strip().lower() == "formal"


def _set_trainable(obj, enabled: bool) -> int:
    if obj is None:
        return 0
    if isinstance(obj, torch.nn.Parameter):
        obj.requires_grad = bool(enabled)
        return int(obj.numel())
    if hasattr(obj, "parameters"):
        count = 0
        for p in obj.parameters():
            p.requires_grad = bool(enabled)
            count += int(p.numel())
        return count
    return 0


def _unfreeze_text_context_modules(model) -> int:
    names = [
        "text_context_pose_projection",
        "text_context_pose_encoder",
        "text_context_text_projection",
        "text_context_type_embed",
        "text_context_gate",
    ]
    touched = 0
    for name in names:
        touched += _set_trainable(getattr(model, name, None), True)
    return touched


def _text_context_param_summary(model) -> str:
    names = [
        "text_context_pose_projection",
        "text_context_pose_encoder",
        "text_context_text_projection",
        "text_context_type_embed",
        "text_context_gate",
    ]
    rows = []
    total = 0
    trainable = 0
    for name in names:
        obj = getattr(model, name, None)
        if obj is None:
            continue
        if isinstance(obj, torch.nn.Parameter):
            n_total = int(obj.numel())
            n_train = n_total if obj.requires_grad else 0
        else:
            params = list(obj.parameters()) if hasattr(obj, "parameters") else []
            n_total = sum(int(p.numel()) for p in params)
            n_train = sum(int(p.numel()) for p in params if p.requires_grad)
        total += n_total
        trainable += n_train
        rows.append(f"{name}: {n_train}/{n_total}")
    if not rows:
        return "no text_context_* modules found"
    return "; ".join(rows) + f"; total={trainable}/{total}"


def _unwrap(edge_obj):
    model = getattr(edge_obj, "model", None)
    if model is None:
        return None
    accelerator = getattr(edge_obj, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "unwrap_model"):
        try:
            return accelerator.unwrap_model(model)
        except Exception:
            pass
    return getattr(model, "module", model)


def _extract_gate(model) -> float:
    gate = getattr(model, "text_context_gate", None)
    if gate is None:
        return 0.0
    try:
        return float(gate.detach().float().cpu().reshape(-1)[0])
    except Exception:
        return 0.0


def _read_monitor_summary(edge_obj) -> dict:
    monitor = getattr(edge_obj, "_edge_text_context_grad_monitor", None)
    if monitor is None:
        return {"hook_calls": 0, "last_grad_norms": {}, "max_grad_norm": 0.0, "sum_grad_norm": 0.0}
    last = dict(getattr(monitor, "last", {}) or {})
    values = [float(v) for v in last.values() if isinstance(v, (int, float))]
    return {
        "hook_calls": int(getattr(monitor, "count", 0)),
        "last_grad_norms": last,
        "max_grad_norm": float(max(values) if values else 0.0),
        "sum_grad_norm": float(sum(values) if values else 0.0),
    }


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Wrote Text/Pose Context evidence JSON: {p}")


def _maybe_attach_monitor(edge_obj) -> None:
    try:
        from edge_experiment_guard import TextContextGradMonitor, write_text_context_report
    except Exception as exc:
        print(f"⚠️ Text Context monitor skipped: cannot import edge_experiment_guard: {exc}")
        return

    unwrapped = _unwrap(edge_obj)
    if unwrapped is None:
        return
    if not bool(getattr(unwrapped, "enable_text_context_rag", False)):
        return

    report_path = os.environ.get("EDGE_TEXT_CONTEXT_REPORT_JSON", "")
    if report_path:
        write_text_context_report(unwrapped, report_path)

    if getattr(edge_obj, "_edge_text_context_grad_monitor", None) is None:
        edge_obj._edge_text_context_grad_monitor = TextContextGradMonitor(
            unwrapped,
            print_every=_env_int("EDGE_TEXT_CONTEXT_GRAD_PRINT_EVERY", 200),
        )


def _final_text_context_contract(edge_obj) -> None:
    model = _unwrap(edge_obj)
    if model is None or not bool(getattr(model, "enable_text_context_rag", False)):
        return

    gate = _extract_gate(model)
    grad = _read_monitor_summary(edge_obj)
    payload = {
        "enabled": True,
        "text_context_gate": gate,
        **grad,
        "require_grad": _env_bool("EDGE_TEXT_CONTEXT_REQUIRE_GRAD", _formal()),
        "min_grad_norm": _env_float("EDGE_TEXT_CONTEXT_MIN_GRAD_NORM", 1e-10),
        "require_gate": _env_bool("EDGE_TEXT_CONTEXT_REQUIRE_GATE", False),
        "min_gate_abs": _env_float("EDGE_TEXT_CONTEXT_MIN_GATE_ABS", 1e-4),
        "param_summary": _text_context_param_summary(model),
    }

    evidence_path = os.environ.get("EDGE_TEXT_CONTEXT_EVIDENCE_JSON", "")
    if _formal() and not evidence_path:
        evidence_path = os.environ.get("EDGE_TEXT_CONTEXT_GRAD_JSON", "logs/text_context_evidence.json")
    _write_json(evidence_path, payload)

    if payload["require_grad"]:
        if int(payload["hook_calls"]) <= 0 or float(payload["max_grad_norm"]) < float(payload["min_grad_norm"]):
            raise RuntimeError(
                "Text/Pose Context RAG appears inactive: no sufficient text_context_* gradients. "
                f"hook_calls={payload['hook_calls']}, max_grad_norm={payload['max_grad_norm']:.3e}, "
                f"threshold={payload['min_grad_norm']:.3e}."
            )

    if payload["require_gate"]:
        if abs(float(gate)) < float(payload["min_gate_abs"]):
            raise RuntimeError(
                "Text/Pose Context gate stayed near zero. "
                f"gate={gate:.3e}, threshold={payload['min_gate_abs']:.3e}."
            )


def install_edge_text_context_training_fix(EDGEClass=None, verbose: bool = True) -> bool:
    if EDGEClass is None:
        try:
            from EDGE import EDGE as EDGEClass  # type: ignore
        except Exception as exc:
            if verbose:
                print(f"⚠️ Text Context training fix skipped: cannot import EDGE: {exc}")
            return False

    if getattr(EDGEClass, "_edge_text_context_training_fix_v3_installed", False):
        return True

    original_apply_stage_freezing = EDGEClass._apply_stage_freezing
    original_init = EDGEClass.__init__
    original_train_loop = EDGEClass.train_loop

    def patched_apply_stage_freezing(self, model, train_stage, adapter_train_decoder=False):
        out = original_apply_stage_freezing(
            self,
            model,
            train_stage,
            adapter_train_decoder=adapter_train_decoder,
        )
        enabled = bool(getattr(model, "enable_text_context_rag", False)) or _env_bool("EDGE_ENABLE_TEXT_CONTEXT_RAG", False)
        if str(train_stage) == "adapter" and enabled:
            touched = _unfreeze_text_context_modules(model)
            if verbose:
                print(
                    "🧩 train_stage=adapter: Text/Pose Context RAG trainable fix applied; "
                    f"params_touched={touched}; {_text_context_param_summary(model)}"
                )
        return out

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            _maybe_attach_monitor(self)
        except Exception as exc:
            print(f"⚠️ Text Context monitor attach failed after EDGE.__init__: {exc}")

    def patched_train_loop(self, opt):
        try:
            _maybe_attach_monitor(self)
        except Exception as exc:
            print(f"⚠️ Text Context monitor attach failed before train_loop: {exc}")
        try:
            return original_train_loop(self, opt)
        finally:
            monitor = getattr(self, "_edge_text_context_grad_monitor", None)
            grad_path = os.environ.get("EDGE_TEXT_CONTEXT_GRAD_JSON", "")
            if monitor is not None and grad_path:
                try:
                    monitor.write(grad_path)
                except Exception as exc:
                    print(f"⚠️ Failed to write Text Context grad monitor report: {exc}")
            report_path = os.environ.get("EDGE_TEXT_CONTEXT_REPORT_JSON", "")
            if report_path:
                try:
                    from edge_experiment_guard import write_text_context_report

                    unwrapped = _unwrap(self)
                    if unwrapped is not None:
                        write_text_context_report(unwrapped, report_path)
                except Exception as exc:
                    print(f"⚠️ Failed final Text Context report: {exc}")
            _final_text_context_contract(self)

    EDGEClass._apply_stage_freezing = patched_apply_stage_freezing
    EDGEClass.__init__ = patched_init
    EDGEClass.train_loop = patched_train_loop
    EDGEClass._edge_original_apply_stage_freezing = original_apply_stage_freezing
    EDGEClass._edge_original_init_for_text_context_fix = original_init
    EDGEClass._edge_original_train_loop_for_text_context_fix = original_train_loop
    EDGEClass._edge_text_context_training_fix_installed = True
    EDGEClass._edge_text_context_training_fix_v2_installed = True
    EDGEClass._edge_text_context_training_fix_v3_installed = True

    if verbose:
        print("✅ Installed Text/Pose Context RAG training fix v3: adapter unfreeze + grad/gate contract.")
    return True


def install():
    return install_edge_text_context_training_fix(verbose=True)
