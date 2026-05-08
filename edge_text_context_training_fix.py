"""Training-stage fix and evidence monitor for Text/Pose Context RAG.

Drop this file into the EDGE project root.

Fixes/diagnostics:
1. Keeps text_context_* modules trainable in train_stage="adapter".
2. Prints and optionally writes text_context_gate / parameter snapshots.
3. Registers gradient hooks for text_context_* parameters, so a smoke/full run
   can prove the branch is not merely instantiated but actually receives grads.

Useful env vars:
    EDGE_TEXT_CONTEXT_REPORT_JSON=logs/text_context_report.json
    EDGE_TEXT_CONTEXT_GRAD_JSON=logs/text_context_grad.json
    EDGE_TEXT_CONTEXT_GRAD_PRINT_EVERY=200
"""

from __future__ import annotations

import os
import torch


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, default)))
    except Exception:
        return int(default)


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


def _maybe_attach_monitor(edge_obj) -> None:
    try:
        from edge_experiment_guard import TextContextGradMonitor, write_text_context_report
    except Exception as exc:
        print(f"⚠️ Text Context monitor skipped: cannot import edge_experiment_guard: {exc}")
        return

    model = getattr(edge_obj, "model", None)
    if model is None:
        return

    unwrapped = getattr(getattr(edge_obj, "accelerator", None), "unwrap_model", lambda x: getattr(x, "module", x))(model)
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


def install_edge_text_context_training_fix(EDGEClass=None, verbose: bool = True) -> bool:
    if EDGEClass is None:
        try:
            from EDGE import EDGE as EDGEClass  # type: ignore
        except Exception as exc:
            if verbose:
                print(f"⚠️ Text Context training fix skipped: cannot import EDGE: {exc}")
            return False

    if getattr(EDGEClass, "_edge_text_context_training_fix_v2_installed", False):
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

        enabled = bool(getattr(model, "enable_text_context_rag", False)) or _env_bool(
            "EDGE_ENABLE_TEXT_CONTEXT_RAG", False
        )
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
                monitor.write(grad_path)
            report_path = os.environ.get("EDGE_TEXT_CONTEXT_REPORT_JSON", "")
            if report_path:
                try:
                    from edge_experiment_guard import write_text_context_report
                    unwrapped = getattr(getattr(self, "accelerator", None), "unwrap_model", lambda x: getattr(x, "module", x))(self.model)
                    write_text_context_report(unwrapped, report_path)
                except Exception as exc:
                    print(f"⚠️ Failed final Text Context report: {exc}")

    EDGEClass._apply_stage_freezing = patched_apply_stage_freezing
    EDGEClass.__init__ = patched_init
    EDGEClass.train_loop = patched_train_loop
    EDGEClass._edge_original_apply_stage_freezing = original_apply_stage_freezing
    EDGEClass._edge_original_init_for_text_context_fix = original_init
    EDGEClass._edge_original_train_loop_for_text_context_fix = original_train_loop
    EDGEClass._edge_text_context_training_fix_installed = True
    EDGEClass._edge_text_context_training_fix_v2_installed = True

    if verbose:
        print("✅ Installed Text/Pose Context RAG training fix v2: adapter unfreeze + grad evidence monitor.")
    return True


def install():
    return install_edge_text_context_training_fix(verbose=True)
